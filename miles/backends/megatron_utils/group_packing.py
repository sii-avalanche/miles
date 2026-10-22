"""Complete groups -> token windows -> one native optimizer update.

Only the backward schedule is split. Clipping, Adam and LR stepping stay in
train_one_step, after every group has arrived and the producer has acknowledged it.
"""

import logging
from contextlib import ExitStack
from copy import copy

import torch
import torch.distributed as dist
from megatron.core.utils import get_model_config
from miles.backends.megatron_utils.group_packing_plan import (
    plan_microbatches,
    select_ready,
)
from miles.backends.training_utils.data.rollout import (
    get_data_iterator,
    get_rollout_data,
)
from miles.backends.training_utils.loss.objective import compute_advantages_and_returns
from miles.utils.data import RolloutDataPack
from miles.utils.gradient_accumulation import DeferredFinalize
from miles.utils.group_stream import GroupStream
from miles.utils.timer import Timer

logger = logging.getLogger(__name__)


class StreamingGroups:
    """Owns one batch's consumption and host object-store leases."""

    def __init__(self, args, first, rollout_id):
        self.args = args
        self.stream = GroupStream(first["group_stream"], timeout=args.group_packing_timeout)
        self.batch = self.stream.batch
        if self.batch != dict(rollout_id=rollout_id, count=args.rollout_batch_size, size=args.n_samples_per_prompt):
            raise ValueError("Stream descriptor does not match this optimizer update")
        self.first = first
        self.rollout_data = first  # Native train loop inspects this attribute.
        self.used = 0
        self.lengths = []
        self.resources = ExitStack()
        # CPU control does not synchronize CUDA or make other ranks wait in NCCL
        # while a slow rollout is still running. DP1/PP1/CP1 is validated at launch.
        self.control_group = dist.new_group(backend="gloo")

    def reset(self):
        if self.used:
            raise RuntimeError("A streaming batch can only be trained once")

    def close(self):
        self.resources.close()
        dist.destroy_process_group(self.control_group)

    def load(self, row):
        if row["index"] != self.used:
            raise ValueError("Duplicate or reordered streamed group")
        if self.used == 0:
            data = self.first
        else:
            pack = RolloutDataPack.model_validate(row["pack"])
            data, lease = get_rollout_data(self.args, pack.data_ref)
            self.resources.enter_context(lease)
            compute_advantages_and_returns(self.args, data)
        if data["total_lengths"] != row["lengths"] or len(data["tokens"]) != self.batch["size"]:
            raise ValueError("Published lengths do not match the loaded group")
        self.used += 1
        self.lengths.extend(row["lengths"])
        return {key: value for key, value in data.items() if key != "group_stream"}

    def windows(self):
        count = self.batch["count"]
        while self.used < count:
            # All ranks wait outside collectives; only rank zero selects the window.
            rows = self.stream.wait(self.used)
            packet = [None]
            if dist.get_rank() == 0:
                try:
                    selected = select_ready(
                        rows,
                        remaining=count - self.used,
                        budget=self.args.group_packing_window_tokens or 4 * self.args.max_tokens_per_gpu,
                    )
                    lengths = [length for row in selected for length in row["lengths"]]
                    bins = plan_microbatches(
                        lengths,
                        capacity=self.args.max_tokens_per_gpu,
                        native_layout=len(selected) == count - self.used,
                    )
                    packet[0] = dict(selected=selected, bins=bins)
                except Exception as error:
                    packet[0] = dict(error=repr(error))
            dist.broadcast_object_list(packet, src=0, group=self.control_group)
            if "error" in packet[0]:
                raise ValueError(packet[0]["error"])
            selected, bins = packet[0]["selected"], packet[0]["bins"]
            start = self.used
            groups = [self.load(row) for row in selected]
            data = merge_groups(groups)
            data["micro_batch_indices"] = bins
            data["num_microbatches"] = [len(bins)]
            logger.info(
                "group packing: groups=%s:%s/%s tokens=%s microbatches=%s",
                start,
                self.used,
                count,
                sum(data["total_lengths"]),
                len(bins),
            )
            yield data
            # Leases for later groups are needed only through their backward pass.
            self.resources.close()
            del data, groups
        self.stream.wait_finished()
        Timer().seq_lens = self.lengths


def merge_groups(groups):
    sizes = [len(group["tokens"]) for group in groups]
    keys = set(groups[0])
    if any(set(group) != keys for group in groups):
        raise ValueError("Group schemas differ within an optimizer update")
    result = {}
    for key in keys:
        values = [group[key] for group in groups]
        if all(value is None for value in values):
            result[key] = None
        elif all(isinstance(value, list) and len(value) == size for value, size in zip(values, sizes, strict=True)):
            result[key] = [item for value in values for item in value]
        elif all(
            isinstance(value, torch.Tensor) and value.ndim and len(value) == size
            for value, size in zip(values, sizes, strict=True)
        ):
            result[key] = torch.cat(values)
        else:
            raise ValueError(f"Unsupported streamed field: {key}")
    return result


def run_group_windows(args, dumper, source, model, num_rollouts, native_backward):
    config = get_model_config(model[0])
    losses = []
    try:
        with DeferredFinalize(config) as finalizer:
            for data in source.windows():
                view = copy(args)
                view.global_batch_size = len(data["tokens"])
                iterators, counts = get_data_iterator(view, model, data)
                before = finalizer.calls
                losses.extend(native_backward(args, dumper, iterators, model, counts[0], num_rollouts))
                if finalizer.calls != before + 1:
                    raise RuntimeError("Native schedule did not finalize exactly once per window")
                del iterators, data
        # source.windows checks producer completion before we normalize/commit.
        finalizer.finish()
    except BaseException as error:
        source.stream.fail(error)
        raise
    return losses
