"""Standalone CPU regressions; no Ray/model/sandbox initialization required."""

import ast
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[4]
MEGATRON_ROOT = Path(os.environ.get("MEGATRON_ROOT", ROOT.parent / "Megatron-LM"))


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


primitive = load_file("chunked_head", ROOT / "miles/backends/training_utils/chunked_lm_head.py")
adapter = load_file("chunked_head_adapter", ROOT / "miles/backends/megatron_utils/chunked_lm_head.py")


def load_existing_function(name, filename, **namespace):
    """Exercise the exact repository function without importing the GPU stack."""
    tree = ast.parse(filename.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    scope = {"torch": torch, **namespace}
    exec(compile(ast.fix_missing_locations(module), str(filename), "exec"), scope)
    return scope[name]


dppo = load_existing_function("compute_dppo_loss", ROOT / "miles/backends/training_utils/loss/hub/math_utils.py")


@pytest.mark.parametrize("chunk", [1, 4, 32])
@pytest.mark.parametrize("temperature", [0.7, 1.0])
def test_logprobs_loss_gradients_and_one_optimizer_step(chunk, temperature):
    torch.manual_seed(42)
    h = torch.randn(13, 5, dtype=torch.float64, requires_grad=True)
    w = torch.randn(11, 5, dtype=torch.float64, requires_grad=True)
    hr = h.detach().clone().requires_grad_()
    wr = w.detach().clone().requires_grad_()
    targets = torch.randint(11, (13,))
    reference = ((hr @ wr.t()) / temperature).log_softmax(-1).gather(1, targets[:, None]).squeeze(1)
    actual = primitive.chunked_logprobs(h, w, targets, chunk, temperature=temperature)
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
    old = reference.detach() + torch.linspace(-2, 2, 13)
    advantages = torch.linspace(-1, 1, 13)
    mask = torch.tensor([0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0], dtype=torch.float64)
    a, blocked, _ = dppo(old, actual, advantages, 0.1, "tv")
    b, blocked_ref, _ = dppo(old, reference, advantages, 0.1, "tv")
    torch.testing.assert_close(blocked, blocked_ref)
    ((a * mask).sum() / mask.sum()).backward()
    ((b * mask).sum() / mask.sum()).backward()
    torch.testing.assert_close(h.grad, hr.grad, atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(w.grad, wr.grad, atol=1e-12, rtol=1e-11)
    torch.optim.AdamW([h, w], lr=1e-6).step()
    torch.optim.AdamW([hr, wr], lr=1e-6).step()
    torch.testing.assert_close(h, hr, atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(w, wr, atol=1e-12, rtol=1e-11)


def test_gradcheck_and_no_saved_vocab_activations():
    torch.manual_seed(2)
    h = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    w = torch.randn(19, 3, dtype=torch.float64, requires_grad=True)
    t = torch.arange(7)
    assert torch.autograd.gradcheck(lambda a, b: primitive.chunked_logprobs(a, b, t, 3), (h, w))
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda x: saved.append(tuple(x.shape)) or x, lambda x: x):
        primitive.chunked_logprobs(h, w, t, 3).sum().backward()
    assert saved == [(7, 3), (19, 3), (7,)]


def test_existing_bf16_backward_rounding():
    torch.manual_seed(5)
    h = torch.randn(9, 4, requires_grad=True)
    w = torch.randn(17, 4, requires_grad=True)
    t = torch.arange(9)
    upstream = torch.randn(9)
    logits = h.detach() @ w.detach().t()
    grad = logits.softmax(-1)
    grad[torch.arange(9), t] -= 1
    grad = (grad * -upstream[:, None]).to(torch.bfloat16).float()
    primitive.chunked_logprobs(h, w, t, 3, backward_dtype=torch.bfloat16).backward(upstream)
    torch.testing.assert_close(h.grad, grad @ w.detach(), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(w.grad, grad.t() @ h.detach(), atol=1e-6, rtol=1e-5)


def test_response_boundaries_use_existing_slicer():
    state = SimpleNamespace(cp=SimpleNamespace(size=1))
    slicer = load_existing_function(
        "get_responses",
        ROOT / "miles/backends/training_utils/loss/hub/logit_processors.py",
        _iter_response_chunks=load_existing_function(
            "_iter_response_chunks",
            ROOT / "miles/backends/training_utils/loss/hub/logit_processors.py",
            get_parallel_state=lambda: state,
        ),
    )
    args = SimpleNamespace(qkv_format="thd", true_on_policy_mode=False, rollout_temperature=0.7)
    values = torch.arange(12, dtype=torch.float32).reshape(1, 12, 1)
    # Two packed samples, prompt=2 and prompt=1; extra positions are padding.
    result = list(
        slicer(
            values,
            args=args,
            unconcat_tokens=[torch.arange(5), torch.arange(4)],
            total_lengths=[5, 4],
            response_lengths=[3, 3],
        )
    )
    assert [x[:, 0].tolist() for x, _ in result] == [[1, 2, 3], [5, 6, 7]]


@pytest.mark.parametrize(
    "overrides",
    [
        {"entropy_coef": 0.1},
        {"context_parallel_size": 2},
        {"pipeline_model_parallel_size": 2},
        {"true_on_policy_mode": True},
        {"untie_embeddings_and_output_weights": False},
        {"lora_rank": 8},
        {"use_rollout_entropy": True},
        {"observe_training_entropy": True},
        {"use_sampling_support_replay": True},
        {"multi_lora": True},
        {"enable_mtp_training": True},
        {"mtp_num_layers": 1},
        {"loss_type": "sft_loss"},
        {"qkv_format": "bshd"},
        {"fp8": "hybrid"},
        {"fp4": "nvfp4"},
        {"defer_embedding_wgrad_compute": True},
        {"train_backend": "fsdp"},
        {"use_critic": True},
        {"enable_witness": True},
        {"use_opd": True},
        {"fp16": True},
        {"cuda_graph_impl": "local"},
        {"chunked_lm_head_size": -1},
        {"rollout_temperature": 0},
        {"rollout_temperature": float("nan")},
    ],
)
def test_unsupported_fails_closed(overrides):
    settings = dict(chunked_lm_head_size=4, rollout_temperature=1.0, untie_embeddings_and_output_weights=True)
    settings.update(overrides)
    with pytest.raises(ValueError):
        adapter.chunked_head_kwargs(SimpleNamespace(**settings))


def test_disabled_is_unchanged():
    assert adapter.chunked_head_kwargs(SimpleNamespace()) == {}


def test_bf16_hidden_and_masked_padding():
    torch.manual_seed(17)
    h = torch.randn(8, 4, dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(23, 4, requires_grad=True)
    out = primitive.chunked_logprobs(h, w, torch.arange(8), 3)
    out[:5].sum().backward()
    assert h.grad.dtype == torch.bfloat16
    assert h.grad[5:].count_nonzero() == 0
    assert torch.isfinite(w.grad).all()


def test_output_processor_contract():
    hidden = torch.ones(5, 1, 3)
    layer = object()
    ids = torch.arange(5).reshape(1, 5)
    settings = SimpleNamespace(
        chunked_lm_head_size=4, rollout_temperature=0.7, untie_embeddings_and_output_weights=True
    )
    processor = adapter.chunked_head_kwargs(settings)["output_processor"]
    with patch.object(adapter, "_project", return_value=hidden) as project:
        assert (
            processor(
                hidden_states=hidden,
                output_layer=layer,
                output_weight=None,
                input_ids=ids,
                labels=None,
                inference_context=None,
                config=SimpleNamespace(mtp_num_layers=None),
            )
            is hidden
        )
        project.assert_called_once_with(hidden, layer, None, ids, chunk_size=4, temperature=0.7)


def _tp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, world_size=2, rank=rank)
    try:
        torch.manual_seed(11)
        h = torch.randn(9, 4, dtype=torch.float64, requires_grad=True)
        full = torch.randn(14, 4, dtype=torch.float64, requires_grad=True)
        w = full.detach().chunk(2)[rank].clone().requires_grad_()
        t = torch.tensor([0, 13, 6, 7, 1, 11, 4, 8, 2])
        actual = primitive.chunked_logprobs(h, w, t, 4, group=dist.group.WORLD)
        hr = h.detach().clone().requires_grad_()
        ref = (hr @ full.t()).log_softmax(-1).gather(1, t[:, None]).squeeze(1)
        torch.testing.assert_close(actual, ref)
        actual.square().sum().backward()
        ref.square().sum().backward()
        dist.all_reduce(h.grad)  # Same sum that the TP-copy/SP-gather backward performs.
        torch.testing.assert_close(h.grad, hr.grad)
        torch.testing.assert_close(w.grad, full.grad.chunk(2)[rank])

        # Exercise Megatron's mapping math with Gloo. Its allocation device is
        # hard-coded to CUDA: redirect ONLY that allocation selector to CPU.
        # This is not an NCCL/GPU integration test.
        sys.path[:0] = [str(ROOT), str(MEGATRON_ROOT)]
        for sequence_parallel in (False, True):
            torch.manual_seed(13)
            hidden_full = torch.randn(10, 1, 4, requires_grad=True)
            weight_full = torch.randn(14, 4, requires_grad=True)
            ids = torch.arange(10).reshape(1, 10)
            hidden = (
                (hidden_full.detach().chunk(2)[rank] if sequence_parallel else hidden_full.detach())
                .clone()
                .requires_grad_()
            )
            weight = weight_full.detach().chunk(2)[rank].clone().requires_grad_()
            layer = SimpleNamespace(
                bias=None, weight=weight, tp_group=dist.group.WORLD, sequence_parallel=sequence_parallel
            )
            with patch("torch.cuda.current_device", return_value=torch.device("cpu")):
                actual = adapter._project(hidden, layer, None, ids, chunk_size=4, temperature=1.0)
                dense = hidden_full[:, 0] @ weight_full.t()
                # Existing fused CE rounds the logit gradient through BF16.
                dense.register_hook(lambda g: g.to(torch.bfloat16).float())
                ref = dense.log_softmax(-1).gather(1, ids.reshape(-1).roll(-1)[:, None]).reshape(1, -1, 1)
                torch.testing.assert_close(actual, ref, atol=2e-6, rtol=2e-6)
                actual.sum().backward()
                ref.sum().backward()
            expected = hidden_full.grad.chunk(2)[rank] if sequence_parallel else hidden_full.grad
            torch.testing.assert_close(hidden.grad, expected, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(weight.grad, weight_full.grad.chunk(2)[rank], atol=2e-5, rtol=2e-5)
    finally:
        dist.destroy_process_group()


def test_two_rank_vocabulary_parallel(tmp_path):
    mp.spawn(_tp_worker, args=(f"file://{tmp_path}/tp-rendezvous",), nprocs=2, join=True)


def load_megatron_method(class_name, method_name, relative_path, **namespace):
    path = MEGATRON_ROOT / relative_path
    if not path.is_file():
        pytest.fail("Set MEGATRON_ROOT to the supported Megatron checkout; integration tests cannot be skipped")
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    node.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    scope = {"torch": torch, **namespace}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope[method_name]


def test_upstream_megatron_calls_public_processor_before_projection():
    postprocess = load_megatron_method(
        "GPTModel",
        "_postprocess",
        "megatron/core/models/gpt/gpt_model.py",
        InferenceMode=SimpleNamespace(is_active=lambda: False),
    )

    def unexpected_projection(*args, **kwargs):
        raise AssertionError("Full-vocabulary output layer must not run")

    model = SimpleNamespace(
        config=SimpleNamespace(mtp_num_layers=None, use_mup=False),
        share_embeddings_and_output_weights=False,
        post_process=True,
        output_layer=unexpected_projection,
        compute_language_model_loss=None,
        _scale_logits=None,
    )
    hidden = torch.randn(5, 1, 3)
    ids = torch.arange(5).reshape(1, 5)
    settings = SimpleNamespace(
        chunked_lm_head_size=2, rollout_temperature=0.7, untie_embeddings_and_output_weights=True
    )
    with patch.object(adapter, "_project", return_value=hidden) as project:
        out = postprocess(model, hidden, ids, None, None, None, None, None, **adapter.chunked_head_kwargs(settings))
        assert out is hidden
        project.assert_called_once_with(hidden, model.output_layer, None, ids, chunk_size=2, temperature=0.7)


def test_main_grad_accumulation_and_clipped_optimizer_update():
    make_hook = load_megatron_method(
        "DistributedDataParallel",
        "_make_backward_post_hook",
        "megatron/core/distributed/distributed_data_parallel.py",
        is_graph_capturing=lambda: False,
    )
    torch.manual_seed(23)
    w = torch.nn.Parameter(torch.randn(11, 4))
    wr = torch.nn.Parameter(w.detach().clone())
    w.main_grad = torch.zeros_like(w)
    w.grad_added_to_main_grad = False
    ddp = SimpleNamespace(param_to_bucket_group={w: None}, ddp_config=SimpleNamespace(overlap_grad_reduce=False))
    handle = w.register_post_accumulate_grad_hook(make_hook(ddp, w))
    for _ in range(2):
        h = torch.randn(7, 4)
        t = torch.arange(7)
        primitive.chunked_logprobs(h, w, t, 3).sum().backward()
        (h @ wr.t()).log_softmax(-1).gather(1, t[:, None]).sum().backward()
    assert w.grad is None
    torch.testing.assert_close(w.main_grad, wr.grad)
    handle.remove()
    w.grad = w.main_grad.clone()
    torch.nn.utils.clip_grad_norm_([w], 1.0)
    torch.nn.utils.clip_grad_norm_([wr], 1.0)
    torch.optim.AdamW([w], lr=1e-4).step()
    torch.optim.AdamW([wr], lr=1e-4).step()
    torch.testing.assert_close(w, wr)


def test_precomputed_response_logprobs_are_not_scaled_twice():
    path = ROOT / "miles/backends/training_utils/loss/hub/logit_processors.py"
    state = SimpleNamespace(cp=SimpleNamespace(size=1))
    chunks = load_existing_function("_iter_response_chunks", path, get_parallel_state=lambda: state)
    responses = load_existing_function("get_responses", path, _iter_response_chunks=chunks)
    process = load_existing_function("get_log_probs_and_entropy", path, get_responses=responses)
    args = SimpleNamespace(
        chunked_lm_head_size=3, qkv_format="thd", true_on_policy_mode=False, rollout_temperature=0.7
    )
    values = -torch.arange(12, dtype=torch.float32).reshape(1, 12, 1)
    kwargs = dict(
        args=args, unconcat_tokens=[torch.arange(5), torch.arange(4)], total_lengths=[5, 4], response_lengths=[3, 3]
    )
    result = process(values, **kwargs)
    assert [part.tolist() for part in result["log_probs"]] == [[-1, -2, -3], [-5, -6, -7]]
    with pytest.raises(ValueError, match="entropy"):
        process(values, with_entropy=True, **kwargs)
    with pytest.raises(ValueError, match="sampling"):
        process(values, rollout_sampling_mask=[], **kwargs)
    for bad in (torch.zeros(12), torch.zeros(12, 1), torch.zeros(2, 12, 1), torch.zeros(1, 12, 5)):
        with pytest.raises(ValueError, match="precomputed"):
            process(bad, **kwargs)


@pytest.mark.parametrize(
    "changes",
    [
        {"labels": torch.tensor([1])},
        {"inference_context": object()},
        {"config": SimpleNamespace(mtp_num_layers=1)},
        {"config": SimpleNamespace(mtp_num_layers=None, use_mup=True)},
        {"config": SimpleNamespace(mtp_num_layers=None, defer_embedding_wgrad_compute=True)},
    ],
)
def test_processor_rejects_unsupported_runtime(changes):
    options = dict(
        hidden_states=None,
        output_layer=None,
        output_weight=None,
        input_ids=None,
        labels=None,
        inference_context=None,
        config=SimpleNamespace(mtp_num_layers=None),
        chunk_size=3,
        temperature=1.0,
    )
    with pytest.raises(ValueError):
        adapter._process_output(**(options | changes))


def test_autocast_rejected():
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        primitive.chunked_logprobs(torch.ones(2, 3), torch.ones(4, 3), torch.tensor([0, 1]), 1)
