"""Run-wide constraints for continuous SEC; disabled runs keep the old paths."""

import math
from dataclasses import dataclass

from miles.backends.megatron_utils.megatron_config import resolve_megatron_config
from miles.utils.environ import use_legacy_rollout_v1


@dataclass(frozen=True)
class SECConfig:
    alpha: float = 0.02
    bandwidth: float = 0.10
    temperature: float = 0.5
    exploration: float = 0.10
    grid_size: int = 101

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.alpha, self.bandwidth, self.temperature, self.exploration)):
            raise ValueError("SEC parameters must be finite")
        if not 0 < self.alpha <= 1 or self.bandwidth <= 0 or self.temperature <= 0:
            raise ValueError("SEC requires 0 < alpha <= 1, bandwidth > 0, temperature > 0")
        if not 0 <= self.exploration <= 1 or self.grid_size < 2:
            raise ValueError("SEC requires 0 <= exploration <= 1 and grid_size >= 2")

    @classmethod
    def from_args(cls, args):
        return cls(**{name: getattr(args, f"curriculum_{name}") for name in cls.__dataclass_fields__})


def add_curriculum_arguments(parser):
    group = parser.add_argument_group("Continuous SEC curriculum")
    group.add_argument(
        "--curriculum-enabled", action="store_true", help="Use continuous SEC; rollout cadence follows --fully-async."
    )
    group.add_argument("--curriculum-strategy", choices=["continuous_sec"], default="continuous_sec")
    for name, default in (("alpha", 0.02), ("bandwidth", 0.10), ("temperature", 0.5), ("exploration", 0.10)):
        group.add_argument(f"--curriculum-{name}", type=float, default=default)
    group.add_argument(
        "--curriculum-grid-size", type=int, default=101, help="Numerical interpolation nodes, not sampling buckets."
    )
    group.add_argument("--curriculum-id-field", default="id", help="Stable unique question ID in each input record.")
    group.add_argument("--curriculum-initial-probability-field", default="p_i_initial")
    group.add_argument(
        "--curriculum-initial-probability-scale",
        type=float,
        default=100.0,
        help="Use 1 for existing p_correct fractions.",
    )
    return parser


def validate_curriculum_args(args):
    if not getattr(args, "curriculum_enabled", False):
        return
    if args.fully_async:
        raise ValueError("Minimal SEC uses the original batch rollout; set fully_async=false")
    SECConfig.from_args(args)
    if args.curriculum_strategy != "continuous_sec":
        raise ValueError("Only continuous_sec is supported")
    scale = args.curriculum_initial_probability_scale
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("curriculum_initial_probability_scale must be finite and positive")
    if args.n_samples_per_prompt != 16:
        raise ValueError("Continuous SEC requires complete 16-sample groups")
    if args.advantage_estimator != "grpo" or not args.rewards_normalization:
        raise ValueError("Continuous SEC requires GRPO with rewards_normalization")
    if args.normalize_advantages or args.use_opd or getattr(args, "overlong_reward_mode", None) not in (None, "none"):
        raise ValueError("SEC completion feedback cannot use batch advantage whitening, OPD or shaped rewards")
    incompatible = (
        "custom_reward_post_process_path",
        "custom_convert_samples_to_train_data_path",
        "custom_async_data_buffer_path",
        "custom_async_data_buffer_path_per_model",
        "buffer_filter_path",
        "load_debug_rollout_data",
        "rollout_sample_filter_path",
        "multi_lora",
        "partial_rollout",
        "colocate",
        "debug_train_only",
        "debug_rollout_only",
        "debug_skip_weight_update",
        "indep_dp",
    )
    for name in incompatible:
        if getattr(args, name, None):
            raise ValueError(f"Continuous SEC does not support {name}")
    if use_legacy_rollout_v1() or not args.rollout_global_dataset or resolve_megatron_config(args).is_multi_policy:
        raise ValueError("Continuous SEC requires the class-based single-policy global-dataset pipeline")
    if args.data_source_path != "miles.rollout.data_source.RolloutDataSourceWithBuffer":
        raise ValueError("Continuous SEC requires the built-in RolloutDataSourceWithBuffer")
