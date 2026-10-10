"""CLI arguments and configuration plumbing for the miles dashboard."""

from __future__ import annotations

import logging
import math

from miles.dashboard.collector import CollectorConfig
from miles.dashboard.sglang_scraper import DEFAULT_METRIC_WHITELIST

logger = logging.getLogger(__name__)

# curated subset of args persisted into meta.json for the dashboard header
_SNAPSHOT_KEYS = (
    "wandb_group",
    "wandb_team",
    "wandb_project",
    "wandb_run_id",
    "wandb_host",
    "colocate",
    "num_gpus_per_node",
    "actor_num_nodes",
    "actor_num_gpus_per_node",
    "rollout_num_gpus",
    "rollout_num_gpus_per_engine",
    "rollout_batch_size",
    "n_samples_per_prompt",
    "rollout_max_response_len",
    "hf_checkpoint",
    "sglang_max_running_requests",
    "sglang_mem_fraction_static",
    "sglang_load_balance_method",
    "use_miles_router",
    "router_dp_aware",
    "router_policy",
    "sglang_router_policy",
    "router_assignment_mode",
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "context_parallel_size",
    "expert_model_parallel_size",
    "global_batch_size",
    "async_max_concurrent_samples",
    "async_data_buffer_capacity_factor",
    "max_weight_staleness",
    "perf_monitor_interval",
    "perf_monitor_expert_layer",
    "perf_monitor_workload_key",
    "perf_monitor_workload_targets",
)


def add_dashboard_arguments(parser) -> None:
    group = parser.add_argument_group("miles dashboard")
    group.add_argument(
        "--use-miles-dashboard",
        action="store_true",
        default=False,
        help="Collect phases, GPU utilization and engine metrics. Use --dashboard-dir for telemetry only, "
        "or --dump-details for telemetry plus sample dumps. View with `python -m miles.dashboard.serve`.",
    )
    group.add_argument(
        "--dashboard-dir",
        type=str,
        default=None,
        help="Telemetry directory without full rollout/train tensor dumps. Overrides the dashboard location in --dump-details.",
    )
    group.add_argument("--dashboard-flush-interval", type=float, default=5.0, help="collector disk flush cadence (s)")
    group.add_argument("--dashboard-gpu-sample-interval", type=float, default=1.0, help="NVML sampling cadence (s)")
    group.add_argument("--dashboard-sglang-scrape-interval", type=float, default=2.0, help="engine scrape cadence (s)")
    group.add_argument(
        "--dashboard-sglang-scrape-mode",
        type=str,
        choices=["auto", "router", "direct"],
        default="auto",
        help="auto scrapes each engine's /metrics; router scrapes {router}/engine_metrics",
    )
    group.add_argument(
        "--dashboard-sglang-metrics",
        type=str,
        default=None,
        help="comma-separated override of the scraped sglang metric whitelist",
    )
    group.add_argument(
        "--dashboard-forward-prometheus",
        action="store_true",
        default=False,
        help="also push dashboard gauges to the --use-prometheus collector for external Grafana",
    )


def validate_dashboard_args(args) -> None:
    if not args.use_miles_dashboard:
        return
    assert (
        args.dump_details is not None or getattr(args, "dashboard_dir", None) is not None
    ), "--use-miles-dashboard requires --dashboard-dir (telemetry only) or --dump-details (including sample dumps)"
    for name in ("dashboard_flush_interval", "dashboard_gpu_sample_interval", "dashboard_sglang_scrape_interval"):
        assert (
            math.isfinite(getattr(args, name)) and getattr(args, name) > 0
        ), f"--{name.replace('_', '-')} must be finite and positive"
    if args.dump_details is not None and not args.use_rollout_entropy:
        logger.warning(
            "--use-miles-dashboard without --use-rollout-entropy: per-token entropy "
            "will be missing from the dashboard token view"
        )


def collector_config_from_args(args, *, start_ts: float) -> CollectorConfig:
    if args.dashboard_sglang_metrics is not None:
        whitelist = tuple(m for m in args.dashboard_sglang_metrics.split(",") if m)
        assert whitelist, f"empty --dashboard-sglang-metrics: {args.dashboard_sglang_metrics!r}"
    else:
        whitelist = DEFAULT_METRIC_WHITELIST
    snapshot = {key: getattr(args, key) for key in _SNAPSHOT_KEYS if hasattr(args, key)}
    return CollectorConfig(
        # Explicit telemetry location wins; dump_details retains its existing
        # meaning and never gets enabled merely for performance monitoring.
        dashboard_dir=getattr(args, "dashboard_dir", None) or f"{args.dump_details}/dashboard",
        run_name=args.wandb_group or "miles-run",
        start_ts=start_ts,
        args_snapshot=snapshot,
        flush_interval_seconds=args.dashboard_flush_interval,
        gpu_sample_interval_seconds=args.dashboard_gpu_sample_interval,
        scrape_interval_seconds=args.dashboard_sglang_scrape_interval,
        scrape_mode=args.dashboard_sglang_scrape_mode,
        metric_whitelist=whitelist,
        forward_prometheus=args.dashboard_forward_prometheus,
    )
