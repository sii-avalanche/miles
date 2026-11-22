from dataclasses import dataclass

from miles.backends.fsdp_utils.arguments import FSDPArgs, build_dataclass_parser, load_args_from_parser


@dataclass
class DModelArgs(FSDPArgs):
    """Miles arguments that select an external DModel runtime configuration."""

    dmodel_config: str | None = None
    dmodel_cache_dir: str | None = None
    dmodel_freeze_router_gate: bool = False
    dmodel_freeze_expert_bias: bool = False
    dmodel_gpu_update_offload: bool = False
    dmodel_gpu_update_bucket_mb: int = 512
    dmodel_muon_prefetch: bool = False
    dmodel_publish_max_tensor_mb: int = 8192


def build_dmodel_parser(extra_args_provider=None):
    return build_dataclass_parser(DModelArgs, "DModel RL Training (miles)", extra_args_provider)


def load_dmodel_args(extra_args_provider=None):
    return load_args_from_parser(build_dmodel_parser(extra_args_provider))


def bind_dmodel_hf_args(args, hf_config) -> None:
    """Expose DModel architecture fields consumed by shared Miles rollout code."""
    getter = getattr(hf_config, "get_text_config", None)
    text_config = (getter() if callable(getter) else getattr(hf_config, "text_config", None)) or hf_config
    topk = getattr(text_config, "num_experts_per_tok", None)
    if topk is None:
        return
    if isinstance(topk, bool) or not isinstance(topk, int) or topk < 1:
        raise ValueError(
            "DModel requires a positive integer num_experts_per_tok in the HF config "
            "to normalize rollout routing replay"
        )
    args.moe_router_topk = topk


def validate_dmodel_args(args) -> None:
    if not args.dmodel_config:
        raise ValueError("--dmodel-config is required with --train-backend dmodel")
    if not args.debug_train_only:
        if not (args.fsdp_cpu_offload or args.dmodel_gpu_update_offload) or not args.colocate or not args.offload_rollout:
            raise ValueError("DModel online training requires CPU optimizer state and colocated inference offload")
        if getattr(args, "lora_rank", 0) > 0:
            raise ValueError("DModel online publication does not implement LoRA")
    if args.loss_type != "policy_loss":
        raise ValueError("the initial DModel backend supports --loss-type policy_loss only")
    if getattr(args, "calculate_per_token_loss", False):
        raise ValueError("DModel currently implements sample-mean loss, not accumulated global token-mean loss")
    if args.use_critic:
        raise ValueError("the initial DModel backend does not support a critic/value model")
    if args.use_opd:
        raise ValueError("the initial DModel backend does not support on-policy distillation")
    if args.ref_update_interval is not None:
        raise ValueError("the initial DModel backend does not support --ref-update-interval")
    if args.qkv_format != "thd":
        raise ValueError("the DModel backend requires --qkv-format thd")
    if args.fp16 or not args.bf16:
        raise ValueError("the initial DModel backend requires bf16 training")
    if args.optimizer not in {"adam", "muon"}:
        raise ValueError("the DModel backend requires --optimizer adam or muon")
    if args.use_sampling_support_replay:
        raise ValueError("the initial DModel backend does not support sampling-support replay")
    if args.use_routing_replay or args.use_rollout_routing_replay:
        raise ValueError("DModel routing replay is not wired yet; disable routing replay for the first canary")
    if getattr(args, "use_indexer_replay", False) or getattr(args, "use_rollout_indexer_replay", False):
        raise ValueError("DModel indexer replay has no consumer yet")
    if getattr(args, "offload_train", False):
        raise ValueError(
            "DModel phase offload needs a native residency adapter; --fsdp-cpu-offload is a separate policy"
        )
    bucket = getattr(args, "dmodel_gpu_update_bucket_mb", 512)
    if isinstance(bucket, bool) or not isinstance(bucket, int) or bucket <= 0:
        raise ValueError("--dmodel-gpu-update-bucket-mb must be a positive integer")
    if getattr(args, "dmodel_muon_prefetch", False) and (
        not args.dmodel_gpu_update_offload or getattr(args, "fsdp_cpu_offload", False) or args.optimizer != "muon"
    ):
        raise ValueError("--dmodel-muon-prefetch requires Muon, --dmodel-gpu-update-offload and CUDA parameters")
    publish = getattr(args, "dmodel_publish_max_tensor_mb", 8192)
    if type(publish) is not int or publish <= 0:
        raise ValueError("--dmodel-publish-max-tensor-mb must be a positive integer")
