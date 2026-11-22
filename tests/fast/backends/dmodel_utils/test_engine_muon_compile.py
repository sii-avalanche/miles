"""Native optimizer/compile tests with CPU-only engine construction boundaries.

Model assembly, parallelism and CUDA rank discovery are replaced by a tiny CPU
fixture. The engine, freeze contract, optimizer builder, Muon compile policy,
torch.compile and scheduler are real. This is not CUDA/Inductor performance or
FSDP/offload validation.
"""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

import dmodel.operators as operators
import dmodel.parallelism.compile as compile_policy
import dmodel.train as native_train
import dmodel.training.optimizer.muon as native_muon
from dmodel.training.chunked_loss import ChunkedCrossEntropyOutputHead
from dmodel.training.optimizer import CombinedOptimizer
from dmodel.utils.config import Config
from miles.backends.dmodel_utils import engine as engine_module


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.Module()
        layer.use_moe = True
        layer.moe = nn.Module()
        layer.moe.experts = nn.Parameter(torch.arange(24).reshape(2, 4, 3).float() / 37)
        layer.projection = nn.Linear(3, 4)
        self.layers = nn.ModuleList([layer])
        self.output = ChunkedCrossEntropyOutputHead(4, 5, bias=False)


@pytest.fixture
def construct(monkeypatch):
    assert not torch.cuda.is_available(), "Run with CUDA hidden; this fixture must remain CPU-only"
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(operators, "load_dispatch", lambda *args, **kwargs: None)
    monkeypatch.setattr(operators, "refresh_report", lambda: None)
    monkeypatch.setattr(native_train, "resolve_model_config", lambda config: ("cpu-fixture", config.model))
    monkeypatch.setattr(native_train, "apply_config_overrides", lambda config, model_config: model_config)
    monkeypatch.setattr(native_train, "build_meta_model", lambda *args: _Model())
    monkeypatch.setattr(native_train, "apply_parallelism", lambda model, *args: (model, SimpleNamespace(tp_enabled=False)))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(native_train, "materialise_and_init_weights", lambda *args: None)
    calls = []
    real_compile = native_muon.compile_callable

    def recording_compile(function, **kwargs):
        compiled = real_compile(function, **kwargs)
        invocations = []

        def counted(*args, **call_kwargs):
            invocations.append(1)
            return compiled(*args, **call_kwargs)

        calls.append((function, kwargs, counted, invocations, compiled))
        return counted

    monkeypatch.setattr(native_muon, "compile_callable", recording_compile)

    def no_pretraining_loss(*args, **kwargs):
        pytest.fail("The RL engine must not install pretraining CE loss")

    monkeypatch.setattr(compile_policy, "maybe_compile_loss", no_pretraining_loss)

    def build(*, enabled=True, components=("muon",), dynamic=None, backend="aot_eager", optimizer="muon"):
        config = OmegaConf.structured(Config)
        config.compile.enable = enabled
        config.compile.components = list(components)
        config.compile.dynamic = dynamic
        config.compile.backend = backend
        config.optimizer.optimizer_type = optimizer
        config.optimizer.decay_ratio = 0
        config.optimizer.warmup_steps = 0
        config.optimizer.muon_ns_steps = 2
        config.parallelism.fsdp_cpu_offload = False
        monkeypatch.setattr(engine_module, "load_dmodel_config", lambda args: config)
        args = SimpleNamespace(dmodel_freeze_router_gate=False, dmodel_freeze_expert_bias=False)
        engine = engine_module.DModelEngine(args)
        assert all(p.device.type == "cpu" for p in engine.model.parameters())
        assert not torch.cuda.is_initialized()
        return engine, calls

    return build


@pytest.mark.parametrize("dynamic", [None, False, True])
def test_engine_forwards_native_muon_compile_policy(construct, dynamic):
    engine, calls = construct(dynamic=dynamic, backend="inductor")
    assert isinstance(engine.optimizer, CombinedOptimizer)
    muon = engine.optimizer.inner_optimizers()["muon"]
    assert muon._compile_backend == "inductor"
    assert calls[0][0] is native_muon._ns_orthogonalize_batched
    assert calls[0][1] == dict(backend="inductor", fullgraph=True, dynamic=dynamic)
    assert calls[0][2] is muon._compiled_batched_ns
    assert len(calls) == 2  # CPU construction skips the CUDA owner-kernel slot.
    assert engine.scheduler.optimizer is engine.optimizer


@pytest.mark.parametrize("enabled,components,optimizer", [
    (False, ("muon",), "muon"),
    (True, ("model", "loss"), "muon"),
    (True, ("muon",), "adamw"),
])
def test_disabled_unselected_and_adamw_keep_native_default(construct, enabled, components, optimizer):
    engine, calls = construct(enabled=enabled, components=components, optimizer=optimizer)
    assert calls == []
    if optimizer == "muon":
        muon = engine.optimizer.inner_optimizers()["muon"]
        assert muon._compiled_batched_ns is None and muon._compile_backend is None
    else:
        assert isinstance(engine.optimizer, torch.optim.AdamW)


def test_native_cpu_steps_reuse_compiled_callable_and_optimizer(construct):
    engine, calls = construct(backend="aot_eager")
    optimizer, scheduler = engine.optimizer, engine.scheduler
    muon = optimizer.inner_optimizers()["muon"]
    compiled = muon._compiled_batched_ns
    before = engine.model.layers[0].moe.experts.detach().clone()
    for index in range(3):
        optimizer.zero_grad(set_to_none=True)
        for parameter in engine.model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.full_like(parameter, 0.1 + index * 0.03)
        optimizer.step()
        scheduler.step()
        compile_policy.maybe_compile_muon(optimizer, engine.config.compile)
        assert muon._compiled_batched_ns is compiled
        assert engine.optimizer is optimizer and engine.scheduler is scheduler
        assert len(calls) == 1
        assert len(calls[0][3]) == index + 1
        assert calls[0][4]._torchdynamo_orig_callable is native_muon._ns_orthogonalize_batched
    assert not torch.equal(before, engine.model.layers[0].moe.experts)
    assert scheduler.last_epoch == 3
    assert muon.state and all("momentum_buffer" in state for state in muon.state.values())


def test_staged_bucket_preserves_compiled_callables(construct):
    """Exercise the compiled bucket routine with a CPU staging fixture."""

    engine, calls = construct(backend="inductor")
    muon = engine.optimizer.inner_optimizers()["muon"]
    assert muon._owner_compiled_ns is None and muon._cublaslt_fn is None
    # GPU-only slots get opaque CPU markers; the two other slots are real wrappers.
    muon._owner_compiled_ns = lambda: None
    muon._cublaslt_fn = lambda: None
    compiled = {name: getattr(muon, name) for name in (
        "_compiled_batched_ns", "_compiled_bucket_updates", "_owner_compiled_ns", "_cublaslt_fn"
    )}
    assert all(callable(value) for value in compiled.values())
    original = (muon.param_groups, muon._param_names, muon._3d_param_ids)
    group = muon.param_groups[0]
    shadow = nn.Parameter(group["params"][0].detach().clone())
    buffer = torch.zeros_like(shadow)
    caches = ("_2d_plan_cache", "_owner_execution_plans", "_owner_arenas",
              "_packed_tables", "_writeback_pointer_tables")
    muon._owner_chunk_sizes["shape-only"] = 2
    original_state_keys = set(muon.state)
    for name in caches:
        getattr(muon, name)["cpu-fixture"] = object()
    muon.step_staged_bucket(group, [shadow], [torch.ones_like(shadow)], [buffer])
    assert all(getattr(muon, name) is value for name, value in compiled.items())
    assert muon.param_groups is original[0]
    assert muon._param_names is original[1] and muon._3d_param_ids is original[2]
    assert shadow not in muon.state and set(muon.state) == original_state_keys
    assert all(not getattr(muon, name) for name in caches)
    assert muon._owner_chunk_sizes["shape-only"] == 2
    assert muon.owner_workspace_bytes == 0
    assert any(call[3] for call in calls)
    assert not torch.cuda.is_initialized()
