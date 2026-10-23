import logging
import os
from copy import deepcopy

import wandb
from wandb.sdk.lib.runid import generate_id

from miles.utils.env_report.launcher_report import read_launcher_report

logger = logging.getLogger(__name__)

# Shared-mode workers do not upload the run's metric configuration. Globs only
# expand after a local log call, so declare common worker metrics on the primary
# even though it never logs their values. Keep globs below for custom metrics.
_WORKER_METRICS = {
    "train": """
        loss pg_loss entropy_loss kl_loss grad_norm ppo_kl pg_clipfrac pg_clipfrac_lower
        pg_clipfrac_pos pg_clipfrac_neg train_rollout_kl train_rollout_logprob_abs_diff
        tis tis_abs tis_clipfrac ois ess_ratio opsm_clipfrac opd_reverse_kl
        lld_ppo_kl_pos lld_ppo_kl_neg mtp_loss mtp_1_loss lr-pg_0 lr-pg_1 lr-pg_2
        critic-loss critic-value_loss critic-value_clipfrac critic-grad_norm critic-lr-pg_0
    """,
    "rollout": """
        raw_reward raw_reward_unfiltered rewards response_lengths total_lengths
        repetition_frac truncated_ratio group_reward_variance_mean group_reward_variance_max
        zero_std/all_zero_percentage zero_std/all_one_percentage
        log_probs ref_log_probs rollout_log_probs teacher_log_probs entropy correct_entropy
        advantages returns values logp_preferred logp_dispreferred frac_preferred
        weight_version/mixed_version_ratio prefix_cache_hit_rate
        dynamic_filter_drop_group_has_aborted dynamic_filter_drop_zero_std_0.0 dynamic_filter_drop_zero_std_1.0
    """,
    "perf": """
        rollout_time actor_train_time train_time train_wait_time update_weights_time step_time wait_time_ratio
        log_probs_time ref_log_probs_time log_probs_tflops ref_log_probs_tflops
        tokens_per_gpu_per_sec effective_tokens_per_gpu_per_sec actor_train_tok_per_s
        actor_train_mfu actor_train_tflops mfu_peak_tflops
    """,
}


def _is_offline_mode(args) -> bool:
    """Detect whether W&B should run in offline mode.

    Priority order:
    1) args.wandb_mode if provided
    2) WANDB_MODE environment variable
    """
    if args.wandb_mode:
        return args.wandb_mode == "offline"
    return os.environ.get("WANDB_MODE") == "offline"


def _wandb_settings(**kwargs):
    return wandb.Settings(init_timeout=300.0, **kwargs)


def init_wandb_primary(args):
    if not args.use_wandb:
        args.wandb_run_id = None
        return

    # Set W&B mode if specified (overrides WANDB_MODE env var)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_mode == "offline":
            logger.info("W&B offline mode enabled. Data will be saved locally.")
        elif args.wandb_mode == "disabled":
            logger.info("W&B disabled mode enabled. No data will be logged.")
        elif args.wandb_mode == "online":
            logger.info("W&B online mode enabled. Data will be uploaded to cloud.")

    offline = _is_offline_mode(args)

    # Only perform explicit login when NOT offline
    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Prepare wandb init parameters
    # add random 6 length string with characters
    if args.wandb_random_suffix:
        group = args.wandb_group + "_" + generate_id()
        run_name = f"{group}-RANK_{args.rank}"
    else:
        group = args.wandb_group
        run_name = args.wandb_group

    # Prepare wandb init parameters
    init_kwargs = {
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "group": group,
        "name": run_name,
        "config": _compute_config_for_logging(args),
    }
    if args.wandb_run_id:
        init_kwargs |= {"id": args.wandb_run_id, "resume": "allow"}

    # Configure settings based on offline/online mode
    if offline:
        init_kwargs["settings"] = _wandb_settings(mode="offline")
    else:
        init_kwargs["settings"] = _wandb_settings(mode="shared", x_primary=True)

    # Add custom directory if specified
    if args.wandb_dir:
        # Ensure directory exists to avoid backend crashes
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir
        logger.info(f"W&B logs will be stored in: {args.wandb_dir}")

    wandb.init(**init_kwargs)

    _init_wandb_common()

    # Set wandb_run_id in args for easy access throughout the training process
    args.wandb_run_id = wandb.run.id


def _compute_config_for_logging(args):
    output = deepcopy(args.__dict__)

    whitelist_env_vars = [
        "SLURM_JOB_ID",
        # We may insert more default values here, and may also allow users to configure a whitelist
    ]
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}

    if env_report_raw := args.env_report:
        if launcher_report := read_launcher_report(env_report_raw):
            output["launcher_env_report"] = launcher_report

    return output


# https://docs.wandb.ai/guides/track/log/distributed-training/#track-all-processes-to-a-single-run
def init_wandb_secondary(args, router_addr=None):
    wandb_run_id = getattr(args, "wandb_run_id", None)
    if wandb_run_id is None:
        return

    # Set W&B mode if specified (same as primary)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    offline = _is_offline_mode(args)

    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Configure settings based on offline/online mode
    if offline:
        settings_kwargs = dict(mode="offline")
    else:
        settings_kwargs = dict(
            mode="shared",
            x_primary=False,
            x_update_finish_state=False,
        )

    if args.sglang_enable_metrics and router_addr is not None and getattr(args, "perf_monitor_interval", 0) <= 0:
        logger.info(f"Forward SGLang metrics at {router_addr} to WandB.")
        settings_kwargs |= dict(
            x_stats_open_metrics_endpoints={
                "sgl_engine": f"{router_addr}/engine_metrics",
            },
            x_stats_open_metrics_filters={
                "sgl_engine.*": {},
            },
        )

    init_kwargs = {
        "id": wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "config": args.__dict__,
        "resume": "allow",
        "reinit": True,
        "settings": _wandb_settings(**settings_kwargs),
    }

    # Add custom directory if specified
    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    wandb.init(**init_kwargs)

    _init_wandb_common()


def _init_wandb_common():
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("rollout/step")
    wandb.define_metric("rollout/*", step_metric="rollout/step")
    wandb.define_metric("multi_turn/*", step_metric="rollout/step")
    wandb.define_metric("passrate/*", step_metric="rollout/step")
    wandb.define_metric("mopd/*", step_metric="rollout/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    wandb.define_metric("perf/*", step_metric="rollout/step")
    for prefix, names in _WORKER_METRICS.items():
        step_metric = "train/step" if prefix == "train" else "rollout/step"
        for name in names.split():
            wandb.define_metric(f"{prefix}/{name}", step_metric=step_metric, step_sync=False)
    for prefix in ("response_len", "weight_version", "log_prob/token", "log_prob/sample_mean"):
        for statistic in ("min", "mean", "median", "max"):
            wandb.define_metric(f"rollout/{prefix}/{statistic}", step_metric="rollout/step", step_sync=False)
