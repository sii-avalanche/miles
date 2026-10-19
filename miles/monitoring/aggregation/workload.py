"""Bounded workload labels shared by buffer and rollout telemetry."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from miles.utils.types import Sample


def workload_targets(args) -> dict[str, float]:
    return getattr(args, "perf_monitor_workload_targets", None) or {"all": 1.0}


def workload_of(sample, args) -> str:
    if (targets := getattr(args, "perf_monitor_workload_targets", None)) is None:
        return "all"
    # Only explicitly configured labels enter metric names. Unknown metadata
    # maps to one bucket, keeping cardinality bounded and payloads private.
    value = (sample.metadata or {}).get(args.perf_monitor_workload_key)
    return value if isinstance(value, str) and value in targets else "other"


def expert_load_metrics(samples: Iterable[Sample], *, layer: int, num_experts: int) -> dict[str, float]:
    """Count served-policy routing assignments, including prompts, before training."""
    # Routing diagnostics are opt-in; core runtime probes need no NumPy.
    import numpy as np

    counts = np.zeros(num_experts, dtype=np.int64)
    samples_seen = samples_counted = 0
    for sample in samples:
        samples_seen += 1
        routed = sample.rollout_routed_experts
        if routed is None:
            continue
        if routed.ndim != 3 or routed.shape[1] < layer:
            continue
        samples_counted += 1
        # Only one layer; bound temporary int64 allocations even for 250k-token traces.
        for start in range(0, routed.shape[0], 8192):
            ids = np.asarray(routed[start : start + 8192, layer - 1, :]).reshape(-1)
            if ids.size and (ids.min() < 0 or ids.max() >= num_experts):
                raise ValueError("routing replay contains out-of-range expert ids")
            counts += np.bincount(ids.astype(np.int64, copy=False), minlength=num_experts)
    prefix = f"moe/rollout_layer_{layer}/"
    result = {
        f"{prefix}sample_coverage": samples_counted / samples_seen if samples_seen else 0,
        f"{prefix}assignments": int(counts.sum()),
    }
    mean = float(counts.mean())
    if mean > 0:
        result.update(
            {
                f"{prefix}cv": float(counts.std()) / mean,
                f"{prefix}max_over_mean": int(counts.max()) / mean,
                f"{prefix}cold_experts_fraction": float(np.mean(counts < 0.1 * mean)),
            }
        )
    return result


def producer_metrics(
    *,
    live,
    pending_count,
    group_budget,
    samples_per_prompt,
    paused,
    samples_in_flight,
    last_completed,
    last_heartbeat,
    targets,
    now,
):
    metrics = {
        "in_flight_groups": len(live),
        "completed_pending_groups": pending_count - len(live),
        "oldest_in_flight_seconds": max((now - t for t, _ in live), default=0.0),
        "group_submission_budget": group_budget,
        "sample_budget": group_budget * samples_per_prompt,
        "paused": int(paused),
    }
    if samples_in_flight is not None:
        metrics["in_flight_samples"] = samples_in_flight
    if last_completed is not None:
        metrics["seconds_since_group_completed"] = now - last_completed
    if last_heartbeat is not None:
        metrics["event_loop/heartbeat_age_seconds"] = now - last_heartbeat
    targets = targets | {"other": targets.get("other", 0)} if targets else {}
    for name, target in targets.items():
        metrics[f"workload/{name}/target_fraction"] = target
        metrics[f"workload/{name}/in_flight_group_fraction"] = (
            sum(workload == name for _, workload in live) / len(live) if live else 0
        )
    # Workload gauges live alongside the buffer and recorder, not under producer/.
    return {
        key if key.startswith(("workload/", "event_loop/")) else f"producer/{key}": value
        for key, value in metrics.items()
    }


def collection_metrics(prefix, rollout_id, counts, *, target, targets):
    collected = sum(counts.values())
    return {
        f"{prefix}/rollout_id": rollout_id,
        f"{prefix}/target_groups": target,
        f"{prefix}/collected_groups": collected,
        f"{prefix}/progress_fraction": collected / target,
        **{
            f"workload/{name}/collection_progress_fraction": counts.get(name, 0) / (target * fraction)
            for name, fraction in targets.items()
        },
    }


def batch_details(groups, *, current_version, trainer_model_id, workload_counts):
    versions = Counter()
    workload_versions: dict[str, Counter] = {}
    mixed = 0
    for oldest, newest, workload in groups:
        version = str(oldest) if oldest is not None else "unknown"
        versions[version] += 1
        workload_versions.setdefault(workload, Counter())[version] += 1
        mixed += int(oldest != newest)
    return dict(
        current_version=current_version,
        trainer_model_id=trainer_model_id,
        groups=len(groups),
        versions=dict(versions),
        mixed_groups=mixed,
        workload_versions={name: dict(counter) for name, counter in workload_versions.items()},
        groups_before_batch_filter=sum(workload_counts.values()),
        groups_removed_by_batch_filter=max(0, sum(workload_counts.values()) - len(groups)),
        workload_groups_before_batch_filter=workload_counts,
    )
