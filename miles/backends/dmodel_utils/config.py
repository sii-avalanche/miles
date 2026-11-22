from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def _resolve_relative_path(config_path: Path, value: str | None) -> str | None:
    if not value:
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((config_path.parent / path).resolve())


def load_dmodel_config(args) -> DictConfig:
    """Load a strict DModel config and bind Miles-owned RL settings."""
    from dmodel.utils.config import Config

    path = Path(args.dmodel_config)
    if not path.is_file():
        raise FileNotFoundError(f"DModel config does not exist: {path}")

    schema = OmegaConf.structured(Config)
    config = OmegaConf.merge(schema, OmegaConf.load(path))
    OmegaConf.set_struct(config, True)

    # Miles owns the policy checkpoint; DModel owns the selected optimizer recipe.
    config.model.path = args.hf_checkpoint
    config.model.name = None
    config.model.chunked_cross_entropy = True
    config.data.tokenizer_path = args.hf_checkpoint
    config.data.train_files = "miles://external-rollout"
    config.operators.config_path = _resolve_relative_path(path, config.operators.config_path)
    if args.optimizer == "adam":
        config.optimizer.optimizer_type = "adamw"
        config.optimizer.learning_rate = args.lr
        config.optimizer.weight_decay = args.weight_decay
        config.optimizer.beta1 = args.adam_beta1
        config.optimizer.beta2 = args.adam_beta2
        config.optimizer.eps = args.adam_eps
        config.optimizer.warmup_steps = args.lr_warmup_iters
        config.optimizer.decay_ratio = 0.0
    elif args.optimizer == "muon":
        if config.optimizer.optimizer_type != "muon":
            raise ValueError("--optimizer muon requires optimizer.optimizer_type=muon in DModel config")
    else:
        raise ValueError(f"unsupported DModel optimizer: {args.optimizer}")
    config.training.param_mixed_precision = "bfloat16"
    config.training.reduce_mixed_precision = "float32"
    config.training.max_grad_norm = args.clip_grad
    config.training.max_steps = None
    config.training.global_batch_size = args.global_batch_size
    config.training.micro_batch_size = args.micro_batch_size
    config.system.save_final_checkpoint = False
    config.parallelism.fsdp_cpu_offload = args.fsdp_cpu_offload

    config_obj = OmegaConf.to_object(config)
    assert isinstance(config_obj, Config)
    config_obj.validate()
    validate_dmodel_config(config)
    return config


def validate_dmodel_config(config: DictConfig) -> None:
    """Reject DModel features not represented by the Miles runner."""
    parallel = config.parallelism
    unsupported = {
        "parallelism.tp": parallel.tp,
        "parallelism.pp": parallel.pp,
        "parallelism.cp_ulysses": parallel.cp_ulysses,
    }
    invalid = {name: value for name, value in unsupported.items() if value != 1}
    if invalid:
        rendered = ", ".join(f"{name}={value}" for name, value in sorted(invalid.items()))
        raise ValueError(f"the DModel RL backend requires TP=PP=CP-ULYSSES=1; got {rendered}")
    if config.model.attention_backend != "varlen":
        raise ValueError("the DModel RL backend requires model.attention_backend=varlen")
    if not config.model.chunked_cross_entropy:
        raise ValueError("the DModel RL backend requires model.chunked_cross_entropy=true")
