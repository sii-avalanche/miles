"""Buffer instrumentation adapters. Observe decisions without owning the queue."""

import time

from miles.monitoring.aggregation.buffer import BufferStatistics, BufferWindow
from miles.monitoring.aggregation.workload import workload_of, workload_targets
from miles.rollout.filter_hub.common_filters import group_staleness, group_weight_version_stats
from miles.rollout.group_identity import group_identity


class BufferTelemetry:
    def __init__(self, args, monitor, *, capacity, trainer_model_id=None):
        self._args = args
        self._monitor = monitor
        self._capacity = capacity
        self._trainer_model_id = trainer_model_id
        self._perf_prefix = "buffer" if trainer_model_id is None else f"buffer/{trainer_model_id}"
        self._track_workloads = getattr(args, "perf_monitor_workload_targets", None) is not None
        self.stats = BufferStatistics(workloads=workload_targets(args), track_workloads=self._track_workloads)
        self.window = BufferWindow()
        for name in ("stale_rejected", "aborted", "missing_reward", "dynamic_rejected", "drop", "retry"):
            self.increment(f"{name}_groups_total", 0)
        monitor.register(self._perf_prefix, self.snapshot)

    def snapshot(self):
        return self.stats.snapshot(self._capacity())

    def increment(self, name, value=1):
        self._monitor.increment(f"{self._perf_prefix}/{name}", value)

    def wait(self, name):
        return self._monitor.wait(f"{self._perf_prefix}/{name}")

    def seen(self, entry):
        self.increment("groups_seen_total")
        self._workload_increment(entry, "groups_seen_total")

    def consumed(self, entry):
        self.increment("groups_consumed_total")
        self._workload_increment(entry, "groups_consumed_total")

    def _workload_increment(self, entry, name):
        if self._monitor.enabled and self._track_workloads:
            self._monitor.increment(f"workload/{workload_of(entry.prompt_group[0], self._args)}/{name}")

    def enqueued(self, entry):
        if not self._monitor.enabled:
            return
        stats = group_weight_version_stats(entry.group)
        key = (stats.oldest_version, stats.oldest_version != stats.newest_version)
        self.stats.enqueue(time.monotonic(), workload_of(entry.prompt_group[0], self._args), key)
        self.increment("groups_enqueued_total")

    def dequeued(self):
        if self._monitor.enabled:
            self._monitor.observe(f"{self._perf_prefix}/residence", time.monotonic() - self.stats.dequeue())

    def collect(self, entries):
        buffered = [
            s for entry in entries if (s := group_staleness(entry.group, self.stats.current_version)) is not None
        ]
        return self.window.collect(queue_size=len(entries), buffered_staleness=buffered)

    def rejected(self, entry, reason: str, *, reusable: bool, queued_groups: int) -> None:
        if not self._monitor.enabled:
            return
        counter = {"dynamic_filter": "dynamic_rejected", "stale": "stale_rejected"}.get(reason, reason)
        self._monitor.increment(f"{self._perf_prefix}/{counter}_groups_total")
        stats = group_weight_version_stats(entry.group)
        action = (
            "retry" if reusable and getattr(self._args, "async_unused_samples_handler", "drop") == "retry" else "drop"
        )
        self._monitor.increment(f"{self._perf_prefix}/{action}_groups_total")
        workload = workload_of(entry.prompt_group[0], self._args)
        if self._track_workloads:
            self._monitor.increment(f"{self._perf_prefix}/workload/{workload}/{reason}_rejected_groups_total")
            self._monitor.increment(f"{self._perf_prefix}/workload/{workload}/{action}_groups_total")
        self._monitor.event(
            "buffer_reject",
            details=dict(
                **group_identity(entry.prompt_group),
                reason=reason,
                action=action,
                groups=1,
                trainer_model_id=self._trainer_model_id,
                oldest_version=stats.oldest_version,
                newest_version=stats.newest_version,
                current_version=self.stats.current_version,
                queued_groups=queued_groups,
                workload=workload,
            ),
        )
