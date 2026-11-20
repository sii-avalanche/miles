"""Run-wide contract for early group backward; disabled leaves native behavior intact."""

import math
import os
from pathlib import Path


def add_group_packing_arguments(parser):
    group = parser.add_argument_group("Streaming group packing")
    group.add_argument(
        "--streaming-group-packing",
        action="store_true",
        help="Pack complete ready GRPO groups and accumulate one optimizer update.",
    )
    group.add_argument(
        "--group-packing-directory",
        type=str,
        default=None,
        help="Shared control directory; AvaTrain supplies a per-run path.",
    )
    group.add_argument(
        "--group-packing-window-tokens",
        type=int,
        default=None,
        help="Soft window budget; default 4 * max_tokens_per_gpu. Groups stay whole.",
    )
    group.add_argument(
        "--group-packing-timeout",
        type=float,
        default=1800.0,
        help="Maximum wait in seconds for group publication or producer completion.",
    )
    group.add_argument(
        "--tp-reduce-scatter-fp32",
        action="store_true",
        help="Use local FP32 TP sums in both native and streaming numerical comparisons.",
    )


def validate_group_packing(args):
    enabled = getattr(args, "streaming_group_packing", False)
    precision = getattr(args, "tp_reduce_scatter_fp32", False)
    if not enabled and not precision:
        return
    required = {
        "train_backend": "megatron",
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "bf16": True,
        "sequence_parallel": True,
    }
    forbidden = ["tp_comm_overlap", "fp16", "fp8", "indep_dp", "offload_train", "optimizer_cpu_offload", "multi_lora"]
    if enabled:
        required.update(
            calculate_per_token_loss=True,
            use_dynamic_batch_size=True,
            use_rollout_logprobs=True,
            compute_advantages_and_returns=True,
            advantage_estimator="grpo",
            normalize_advantages=False,
            rewards_normalization=True,
            kl_coef=0.0,
            kl_loss_coef=0.0,
            entropy_coef=0.0,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            loss_type="policy_loss",
            ppo_epochs=1,
            balance_data=False,
            delay_split_train_data_by_dp=True,
            tp_reduce_scatter_fp32=True,
            qkv_format="thd",
            rollout_global_dataset=True,
        )
        forbidden += [
            "gradient_accumulation_fusion",
            "overlap_grad_reduce",
            "overlap_param_gather",
            "use_dynamic_global_batch_size",
            "fully_async",
            "colocate",
            "partial_rollout",
            "use_critic",
            "use_opd",
            "keep_old_actor",
            "enable_mtp_training",
            "enable_witness",
            "use_tis",
            "get_mismatch_metrics",
            "use_rollout_routing_replay",
            "use_kl_loss",
            "rollout_data_postprocess_path",
            "custom_loss_function_path",
            "lora_rank",
            "custom_reward_post_process_path",
            "custom_convert_samples_to_train_data_path",
            "rollout_sample_filter_path",
            "rollout_all_samples_process_path",
            "snr_filter_keep_ratio",
            "recompute_logprobs_via_prefill",
            "load_debug_rollout_data",
            "debug_disable_optimizer",
            "save_debug_train_data",
            "save_debug_rollout_data",
            "mtp_num_layers",
            "ref_load",
            "teacher_load",
            "multi_lora_n_adapters",
            "megatron_config",
            "ref_update_interval",
            "custom_megatron_before_train_step_hook_path",
            "ci_inject_rollout_data_path",
        ]
    for key, expected in required.items():
        if getattr(args, key, None) != expected:
            raise ValueError(f"Group packing/TP precision requires {key}={expected!r}")
    for key in forbidden:
        if getattr(args, key, False):
            raise ValueError(f"Group packing/TP precision does not support {key}")
    world = args.actor_num_nodes * args.actor_num_gpus_per_node
    if world != args.tensor_model_parallel_size:
        raise ValueError("Initial group-packing release supports DP1/CP1/PP1 only")
    if getattr(args, "expert_model_parallel_size", 1) != 1:
        raise ValueError("Expert parallelism beyond EP1 needs separate validation")
    if getattr(args, "num_experts", None):
        if getattr(args, "moe_aux_loss_coeff", None) != 0:
            raise ValueError("Streaming MoE currently requires moe_aux_loss_coeff=0")
        if args.expert_tensor_parallel_size != args.tensor_model_parallel_size:
            raise ValueError("Streaming MoE requires expert TP equal to dense TP")
    if not enabled:
        return
    if os.environ.get("MILES_USE_LEGACY_ROLLOUT_V1", "0") != "0":
        raise ValueError("Streaming packing requires the current inference rollout API")
    if args.rollout_function_path != "miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn":
        raise ValueError("Streaming packing currently requires InferenceRolloutFn (AvaCore generation is supported)")
    if args.global_batch_size != args.rollout_batch_size * args.n_samples_per_prompt:
        raise ValueError("One streamed rollout must equal exactly one global_batch_size")
    if args.num_steps_per_rollout not in (None, 1):
        raise ValueError("Streaming groups require one optimizer update per rollout")
    if not args.group_packing_directory or not Path(args.group_packing_directory).is_absolute():
        raise ValueError("group_packing_directory must be an absolute path shared by rollout and training")
    if args.max_tokens_per_gpu is None or args.max_tokens_per_gpu <= 0:
        raise ValueError("Streaming groups require a positive max_tokens_per_gpu")
    if args.group_packing_window_tokens is not None and args.group_packing_window_tokens <= 0:
        raise ValueError("group_packing_window_tokens must be positive")
    if not math.isfinite(args.group_packing_timeout) or args.group_packing_timeout <= 0:
        raise ValueError("group_packing_timeout must be positive")
