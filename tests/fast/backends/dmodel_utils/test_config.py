from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from miles.backends.dmodel_utils.arguments import bind_dmodel_hf_args, validate_dmodel_args
from miles.backends.dmodel_utils.config import _resolve_relative_path, load_dmodel_config, validate_dmodel_config


def _args(**overrides):
    values = dict(
        dmodel_config="dmodel.yaml",
        debug_train_only=True,
        colocate=True, offload_rollout=True,
        loss_type="policy_loss",
        use_critic=False,
        use_opd=False,
        ref_update_interval=None,
        qkv_format="thd",
        fp16=False,
        bf16=True,
        optimizer="adam",
        use_sampling_support_replay=False,
        use_routing_replay=False,
        use_rollout_routing_replay=False,
        fsdp_cpu_offload=False,
        dmodel_gpu_update_offload=False,
    )
    values.update(overrides)
    return Namespace(**values)


def _config(**overrides):
    values = {
        "parallelism": {
            "tp": 1,
            "pp": 1,
            "cp_ring": 1,
            "cp_ulysses": 1,
            "fsdp_cpu_offload": False,
        },
        "model": {
            "attention_backend": "varlen",
            "chunked_cross_entropy": True,
            "num_hash_layers": None,
        },
    }
    for dotted, value in overrides.items():
        section, key = dotted.split("__", 1)
        values[section][key] = value
    return OmegaConf.create(values)


def test_argument_contract_accepts_initial_policy_backend():
    validate_dmodel_args(_args())


def test_online_requires_cpu_optimizer_state_before_model_construction():
    with pytest.raises(ValueError, match="CPU optimizer state"):
        validate_dmodel_args(_args(debug_train_only=False))


def test_argument_contract_accepts_muon_without_cpu_offload():
    validate_dmodel_args(_args(optimizer="muon"))


def test_argument_contract_accepts_muon_gpu_update_offload():
    validate_dmodel_args(_args(optimizer="muon", fsdp_cpu_offload=True, dmodel_gpu_update_offload=True))


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_argument_contract_rejects_invalid_gpu_bucket_budget(budget):
    with pytest.raises(ValueError, match="bucket"):
        validate_dmodel_args(_args(dmodel_gpu_update_bucket_mb=budget))


def test_resident_muon_config_preserves_native_optimizer_settings():
    config_path = Path(__file__).resolve().parents[4] / "examples/dmodel/deepseek_v4_flash_rl_resident_4k_muon_rs1_b2048_pool4.yaml"
    args = _args(
        dmodel_config=str(config_path),
        optimizer="muon",
        hf_checkpoint="/tmp/hf-model",
        lr=1e-6,
        weight_decay=0.01,
        adam_beta1=0.5,
        adam_beta2=0.5,
        adam_eps=1e-6,
        lr_warmup_iters=0,
        clip_grad=1.0,
        global_batch_size=64,
        micro_batch_size=8,
    )
    config = load_dmodel_config(args)
    assert config.optimizer.optimizer_type == "muon"
    assert config.optimizer.muon_lr == 4e-4
    assert config.optimizer.learning_rate == 4e-4
    assert config.optimizer.warmup_steps == 800
    assert config.training.gradient_accumulation_sync_grad is True
    assert config.parallelism.fsdp_reduce_scatter_max_input_buffers == 1
    assert config.optimizer.muon_2d_bucket_max_peak_mb == 2048
    assert config.operators.config_path == str(config_path.parent / "operator_backends_dsv4_flash_rl_resident_4k_pool4.yaml")


def test_muon_cli_rejects_adamw_config(tmp_path):
    config_path = tmp_path / "adamw.yaml"
    config_path.write_text("optimizer:\n  optimizer_type: adamw\n")
    args = _args(dmodel_config=str(config_path), optimizer="muon", hf_checkpoint="/tmp/hf-model")
    with pytest.raises(ValueError, match="optimizer.optimizer_type=muon"):
        load_dmodel_config(args)


def test_argument_contract_accepts_gpu_update_with_cpu_offload():
    validate_dmodel_args(_args(fsdp_cpu_offload=True, dmodel_gpu_update_offload=True))


def test_argument_contract_accepts_optimizer_only_offload():
    validate_dmodel_args(_args(fsdp_cpu_offload=False, dmodel_gpu_update_offload=True))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("loss_type", "score_centering"),
        ("calculate_per_token_loss", True),
        ("use_critic", True),
        ("use_opd", True),
        ("qkv_format", "bshd"),
        ("fp16", True),
        ("optimizer", "sgd"),
        ("use_sampling_support_replay", True),
        ("use_routing_replay", True),
        ("use_rollout_routing_replay", True),
        ("use_indexer_replay", True),
        ("use_rollout_indexer_replay", True),
        ("offload_train", True),
    ],
)
def test_argument_contract_rejects_unsupported_modes(name, value):
    with pytest.raises(ValueError):
        validate_dmodel_args(_args(**{name: value}))


def test_runtime_config_accepts_ep_and_fsdp_only():
    validate_dmodel_config(_config())


def test_runtime_config_accepts_native_fsdp_cpu_offload():
    validate_dmodel_config(_config(parallelism__fsdp_cpu_offload=True))


def test_runtime_config_accepts_deepseek_ring_context_parallel():
    validate_dmodel_config(_config(parallelism__cp_ring=2))


def test_relative_operator_config_is_resolved_from_dmodel_config(tmp_path):
    config_path = tmp_path / "recipe" / "seta.yaml"
    assert _resolve_relative_path(config_path, "operators.yaml") == str(
        (config_path.parent / "operators.yaml").resolve()
    )
    absolute = tmp_path / "shared" / "operators.yaml"
    assert _resolve_relative_path(config_path, str(absolute)) == str(absolute)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("parallelism__tp", 2),
        ("parallelism__pp", 2),
        ("parallelism__cp_ulysses", 2),
        ("model__attention_backend", "sdpa"),
        ("model__chunked_cross_entropy", False),
    ],
)
def test_runtime_config_rejects_unsupported_modes(name, value):
    with pytest.raises(ValueError):
        validate_dmodel_config(_config(**{name: value}))


def online(**changes):
    options = dict(debug_train_only=False, fsdp_cpu_offload=True, dmodel_gpu_update_offload=True, optimizer="muon")
    validate_dmodel_args(_args(**(options | changes)))


def test_online_cpu_state_and_native_inference_offload_allowed():
    online()


@pytest.mark.parametrize("key", ["colocate", "offload_rollout"])
def test_missing_online_residency_policy_rejected(key):
    with pytest.raises(ValueError, match="online training requires"):
        online(**{key: False})


def test_optimizer_state_only_offload():
    online(fsdp_cpu_offload=False)
    with pytest.raises(ValueError, match="CPU optimizer state"):
        online(fsdp_cpu_offload=False, dmodel_gpu_update_offload=False)


def test_generic_model_to_offload_stays_rejected():
    with pytest.raises(ValueError, match="native residency"):
        online(offload_train=True)


def test_online_lora_not_silently_dropped():
    with pytest.raises(ValueError, match="LoRA"):
        online(lora_rank=8)


@pytest.mark.parametrize("size", [0, -1, True, 1.5, "8192"])
def test_publication_source_ceiling_is_explicit_positive_integer(size):
    with pytest.raises(ValueError, match="publish-max-tensor"):
        online(dmodel_publish_max_tensor_mb=size)


def test_training_only_backend_does_not_require_rollout_workers():
    online(debug_train_only=True, colocate=False, offload_rollout=False)


def test_bind_dmodel_hf_args_exposes_router_topk() -> None:
    args = Namespace()

    bind_dmodel_hf_args(args, SimpleNamespace(num_experts_per_tok=6))

    assert args.moe_router_topk == 6


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_bind_dmodel_hf_args_rejects_invalid_router_topk(value) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        bind_dmodel_hf_args(Namespace(), SimpleNamespace(num_experts_per_tok=value))


def test_dense_hf_config_needs_no_router_setting():
    args = Namespace()
    bind_dmodel_hf_args(args, SimpleNamespace())
    assert not hasattr(args, "moe_router_topk")
