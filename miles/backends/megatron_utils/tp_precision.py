"""Opt-in FP32 local TP sums, shared by native and streaming A/B runs.

Uses Megatron's existing all-to-all + FP32-sum primitive. Scoped bindings are
restored even on failure; DP collectives and torch.distributed are not patched.
"""

import importlib
from contextlib import contextmanager

import torch
import torch.distributed as dist
from megatron.core import parallel_state


class _PendingSum:
    def __init__(self, work, source, output):
        self.work, self.source, self.output = work, source, output
        self.done = False

    def wait(self):
        if self.done:
            return
        self.work.wait()
        # The sum is enqueued on the waiting stream. Preserve cross-stream
        # storage lifetime without adding a device-wide synchronization.
        if self.output.is_cuda:
            stream = torch.cuda.current_stream(self.output.device)
            for tensor in (self.source, self.output, getattr(self.work, "all_to_all_output_tensor", None)):
                if isinstance(tensor, torch.Tensor):
                    tensor.record_stream(stream)
        self.done = True
        self.source = None

    def __getattr__(self, name):
        return getattr(self.work, name)


def _sum(output, source, *, group, async_op=False):
    # Optional Megatron primitive: default-off runs do not require this versioned API.
    from megatron.core.distributed.reduce_scatter_with_fp32_accumulation import (
        reduce_scatter_with_fp32_accumulation,
    )

    contiguous = source.contiguous()
    work = reduce_scatter_with_fp32_accumulation(
        output, contiguous, op=dist.ReduceOp.SUM, group=group, async_op=async_op
    )
    return _PendingSum(work, contiguous, output) if async_op else work


def _output(source, group):
    if source.shape[0] % group.size():
        raise ValueError("TP reduce-scatter needs divisible sequence length")
    return torch.empty((source.shape[0] // group.size(), *source.shape[1:]), device=source.device, dtype=source.dtype)


def _te_wrapper(original, target):
    def reduce(source, tp_group=None, async_op=False):
        if tp_group is not target or source.dtype != torch.bfloat16:
            return original(source, tp_group, async_op=async_op)
        if tp_group.size() == 1:
            return source, None
        output = _output(source, tp_group)
        return output, _sum(output, source, group=tp_group, async_op=async_op)

    return reduce


def _mapping_wrapper(original, groups):
    def reduce(source, group, input_split_sizes=None, use_global_buffer=False):
        if not any(group is target for target in groups) or not (
            source.dtype == torch.bfloat16 or (group is not groups[0] and source.dtype == torch.float32)
        ):
            return original(source, group, input_split_sizes=input_split_sizes, use_global_buffer=use_global_buffer)
        if group.size() == 1:
            return source
        if input_split_sizes is None:
            output = _output(source, group)
            _sum(output, source, group=group)
        else:
            # MoE expert TP may have unequal token counts per destination.
            if len(input_split_sizes) != group.size() or sum(input_split_sizes) != source.shape[0]:
                raise ValueError("Invalid expert TP split sizes")
            parts = torch.split(source, input_split_sizes, dim=0)
            output = torch.empty_like(parts[group.rank()])
            width = output.numel()
            received = torch.empty(group.size() * width, device=source.device, dtype=source.dtype)
            dist.all_to_all_single(
                received,
                source.contiguous().view(-1),
                group=group,
                output_split_sizes=[width] * group.size(),
                input_split_sizes=[part.numel() for part in parts],
            )
            output.copy_(received.view(group.size(), width).sum(0, dtype=torch.float32).view_as(output))
        return output

    return reduce


def _direct_wrapper(original, target):
    def reduce(output, input, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if group is not target or input.dtype != torch.bfloat16:
            return original(output, input, op=op, group=group, async_op=async_op)
        if op != dist.ReduceOp.SUM:
            raise ValueError("TP precision adapter only supports SUM")
        return _sum(output, input, group=group, async_op=async_op)

    return reduce


@contextmanager
def tp_precision_context(args):
    if not args.tp_reduce_scatter_fp32:
        yield
        return
    tp = parallel_state.get_tensor_model_parallel_group()
    groups = [tp]
    if args.num_experts:
        groups.append(parallel_state.get_expert_tensor_parallel_group())
    specs = [
        ("transformer_engine.pytorch.distributed", "reduce_scatter_along_first_dim", _te_wrapper, tp),
        ("transformer_engine.pytorch.module.linear", "reduce_scatter_along_first_dim", _te_wrapper, tp),
        ("transformer_engine.pytorch.module.layernorm_linear", "reduce_scatter_along_first_dim", _te_wrapper, tp),
        ("megatron.core.tensor_parallel.mappings", "_reduce_scatter_along_first_dim", _mapping_wrapper, groups),
        ("megatron.core.tensor_parallel.layers", "dist_reduce_scatter_func", _direct_wrapper, tp),
    ]
    bindings = []
    try:
        for path, name, factory, target in specs:
            module = importlib.import_module(path)
            original = getattr(module, name)  # Reject incompatible TE/Megatron versions explicitly.
            bindings.append((module, name, original))
            setattr(module, name, factory(original, target))
        yield
    finally:
        for module, name, original in reversed(bindings):
            setattr(module, name, original)
