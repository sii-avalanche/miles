"""Buffer distribution and per-step selection statistics; no delivery or scheduling."""

import threading
from collections import Counter, deque

from miles.rollout.filter_hub.base_types import MetricGatherer
from miles.rollout.filter_hub.common_filters import GroupWeightVersionStats


class BufferStatistics:
    def __init__(self, *, workloads, track_workloads):
        self.current_version = None
        self.reference_is_published = False
        self._workload_version_counts = {name: Counter() for name in workloads}
        self._track_workloads = track_workloads
        if track_workloads:
            self._workload_version_counts["other"] = Counter()
        self._queued_version_stats = deque()
        self._telemetry_lock = threading.Lock()

    def snapshot(self, capacity) -> dict[str, float]:
        with self._telemetry_lock:
            version = self.current_version
            workloads = {name: counter.copy() for name, counter in self._workload_version_counts.items()}
        total = Counter()
        for counter in workloads.values():
            total.update(counter)
        metrics = self._version_snapshot(total, version, capacity)
        for name, counter in workloads.items() if self._track_workloads else ():
            metrics.update(
                {
                    f"workload/{name}/{key}": value
                    for key, value in self._version_snapshot(counter, version, capacity).items()
                }
            )
        return metrics

    def _version_snapshot(self, counts: Counter, version: int | None, capacity: int) -> dict[str, float]:
        queued = sum(counts.values())
        metrics = {
            "queued_groups": queued,
            "capacity_groups": capacity,
            "fill_fraction": queued / capacity,
            "versions/unknown_groups": 0,
            "versions/mixed_groups": sum(n for (_, mixed), n in counts.items() if mixed),
        }
        if version is not None:
            metrics["current_weight_version"] = version
            buckets = [*(f"lag_{lag}" for lag in range(4)), "older", "future"]
            metrics.update({f"versions/{bucket}_groups": 0 for bucket in buckets})
        for (oldest, _), count in counts.items():
            if oldest is None:
                bucket = "unknown"
            elif version is None:
                continue
            else:
                lag = version - oldest
                bucket = f"lag_{lag}" if 0 <= lag < 4 else "older" if lag >= 4 else "future"
            metrics[f"versions/{bucket}_groups"] += count
        return metrics

    def set_version(self, version, *, published=False):
        with self._telemetry_lock:
            if published or not self.reference_is_published:
                self.current_version = version
            self.reference_is_published |= published

    def enqueue(self, started, workload, key):
        with self._telemetry_lock:
            self._queued_version_stats.append((started, workload, key))
            self._workload_version_counts[workload][key] += 1

    def dequeue(self):
        with self._telemetry_lock:
            started, workload, key = self._queued_version_stats.popleft()
            counter = self._workload_version_counts[workload]
            counter[key] -= 1
            if not counter[key]:
                del counter[key]
        return started


class BufferWindow:
    def __init__(self):
        self._metric_gatherer = MetricGatherer()
        self._metric_aborted_groups = 0
        self._metric_stale_groups = 0
        self._metric_consumed_staleness: list[int] = []
        self._metric_selected_newest_lag: list[int] = []
        self._metric_selected_version_span: list[int] = []
        self._metric_selected_token_lag_sum = 0.0
        self._metric_selected_versioned_tokens = 0
        self._metric_selected_samples = 0
        self._metric_selected_versioned_samples = 0

    def aborted(self):
        self._metric_aborted_groups += 1

    def stale(self):
        self._metric_stale_groups += 1

    def consumed_staleness(self, value):
        self._metric_consumed_staleness.append(value)

    @property
    def filter_metrics(self):
        return self._metric_gatherer

    def selected(
        self,
        stats: GroupWeightVersionStats,
        current_version: int | None,
    ) -> None:
        self._metric_selected_samples += stats.sample_count
        self._metric_selected_versioned_samples += stats.versioned_sample_count

        if stats.oldest_version is not None and stats.newest_version is not None:
            self._metric_selected_version_span.append(stats.newest_version - stats.oldest_version)

        newest_lag = stats.newest_lag(current_version)
        if newest_lag is not None:
            self._metric_selected_newest_lag.append(newest_lag)

        token_weighted_lag = stats.token_weighted_lag(current_version)
        if token_weighted_lag is not None:
            self._metric_selected_token_lag_sum += token_weighted_lag * stats.versioned_token_count
            self._metric_selected_versioned_tokens += stats.versioned_token_count

    def collect(self, *, queue_size, buffered_staleness) -> dict[str, float]:
        prefix = "rollout/fully_async/"
        metrics = {
            f"{prefix}queue_size": queue_size,
            f"{prefix}aborted_groups_filtered": self._metric_aborted_groups,
            f"{prefix}stale_groups_filtered": self._metric_stale_groups,
            **self._metric_gatherer.collect(),
        }
        if consumed := self._metric_consumed_staleness:
            metrics[f"{prefix}avg_staleness"] = sum(consumed) / len(consumed)
            metrics[f"{prefix}max_staleness"] = max(consumed)
        if newest_lag := self._metric_selected_newest_lag:
            metrics[f"{prefix}avg_post_generation_staleness"] = sum(newest_lag) / len(newest_lag)
            metrics[f"{prefix}max_post_generation_staleness"] = max(newest_lag)
        if version_span := self._metric_selected_version_span:
            metrics[f"{prefix}avg_generation_version_span"] = sum(version_span) / len(version_span)
            metrics[f"{prefix}max_generation_version_span"] = max(version_span)
        if self._metric_selected_versioned_tokens:
            metrics[f"{prefix}token_weighted_staleness"] = (
                self._metric_selected_token_lag_sum / self._metric_selected_versioned_tokens
            )
        if self._metric_selected_samples:
            metrics[f"{prefix}weight_version_sample_coverage"] = (
                self._metric_selected_versioned_samples / self._metric_selected_samples
            )
        buffered = buffered_staleness
        if buffered:
            metrics[f"{prefix}buffer_avg_staleness"] = sum(buffered) / len(buffered)
            metrics[f"{prefix}buffer_max_staleness"] = max(buffered)

        self._metric_gatherer = MetricGatherer()
        self._metric_consumed_staleness = []
        self._metric_selected_newest_lag = []
        self._metric_selected_version_span = []
        self._metric_selected_token_lag_sum = 0.0
        self._metric_selected_versioned_tokens = 0
        self._metric_selected_samples = 0
        self._metric_selected_versioned_samples = 0
        self._metric_aborted_groups = self._metric_stale_groups = 0
        return metrics
