"""Fully-async rollout probes; scheduling and queue ownership stay with RL."""

import asyncio
import time

from miles.monitoring.aggregation.workload import (
    batch_details,
    collection_metrics,
    producer_metrics,
    workload_of,
)
from miles.rollout.filter_hub.common_filters import group_weight_version_stats


class RolloutTelemetry:
    def __init__(self, args, monitor, *, group_budget, scheduler, resumed):
        self.args = args
        self.monitor = monitor
        self.group_budget = group_budget
        self.scheduler = scheduler
        self.resumed = resumed
        self.pending = {}
        self.last_completed = None
        self.last_heartbeat = None
        self.targets = dict(getattr(args, "perf_monitor_workload_targets", None) or {})
        monitor.register("", self.snapshot)

    def snapshot(self):
        pending = self.pending.copy()
        return producer_metrics(
            live=[started for task, started in pending.items() if not task.done()],
            pending_count=len(pending),
            group_budget=self.group_budget(),
            samples_per_prompt=self.args.n_samples_per_prompt,
            paused=not self.resumed.is_set(),
            samples_in_flight=getattr(self.scheduler, "samples_in_flight", None),
            last_completed=self.last_completed,
            last_heartbeat=self.last_heartbeat,
            targets=self.targets,
            now=time.monotonic(),
        )

    def submitted(self, task, prompt_group):
        if self.monitor.enabled:
            self.pending[task] = (time.monotonic(), workload_of(prompt_group[0], self.args))
        self.monitor.increment("producer/groups_submitted_total")

    def generated(self, started):
        self.last_completed = time.monotonic()
        self.monitor.observe("producer/group_generation", self.last_completed - started)
        self.monitor.increment("producer/groups_completed_total")

    async def heartbeat(self):
        self.last_heartbeat = time.monotonic()
        while True:
            started = time.monotonic()
            await asyncio.sleep(1)
            self.last_heartbeat = time.monotonic()
            self.monitor.update({"event_loop/lag_seconds": max(0, self.last_heartbeat - started - 1)})

    def batch(self, rollout_id, *, trainer_model_id):
        return BatchTelemetry(
            self.args, self.monitor, rollout_id, trainer_model_id=trainer_model_id, targets=self.targets
        )


class BatchTelemetry:
    def __init__(self, args, monitor, rollout_id, *, trainer_model_id, targets):
        self.args = args
        self.monitor = monitor
        self.rollout_id = rollout_id
        self.trainer_model_id = trainer_model_id
        self.targets = targets
        self.counts = {}
        self.group_workloads = {}
        self.prefix = "drain" if trainer_model_id is None else f"drain/{trainer_model_id}"
        self._report()

    def collected(self, entry):
        if not self.monitor.enabled:
            return
        workload = workload_of(entry.prompt_group[0], self.args)
        self.counts[workload] = self.counts.get(workload, 0) + 1
        self.group_workloads[id(entry.group)] = workload
        self._report()

    def _report(self):
        if self.monitor.enabled:
            self.monitor.update(
                collection_metrics(
                    self.prefix,
                    self.rollout_id,
                    self.counts,
                    target=self.args.rollout_batch_size,
                    targets=self.targets,
                )
            )

    def ready(self, data, *, current_version):
        if self.monitor.enabled:
            groups = []
            for group in data:
                stats = group_weight_version_stats(group)
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                workload = self.group_workloads.get(id(group), workload_of(sample, self.args))
                groups.append((stats.oldest_version, stats.newest_version, workload))
            self.monitor.event(
                "batch_ready",
                rollout_id=self.rollout_id,
                details=batch_details(
                    groups,
                    current_version=current_version,
                    trainer_model_id=self.trainer_model_id,
                    workload_counts=self.counts,
                ),
            )
