"""Observe the valid groups collected by a batch rollout, without retaining samples."""

from argparse import Namespace

from miles.utils.perf_monitor import get_monitor


class BatchPoolTelemetry:
    def __init__(self, args: Namespace, rollout_id: int):
        self._monitor = get_monitor(args)
        self._rollout_id = rollout_id
        self._target = args.rollout_batch_size
        self._collected = 0

    def __enter__(self):
        for name in ("completed", "selected", "rejected"):
            self._monitor.increment(f"batch/groups_{name}_total", 0)
        self._publish(status="collecting", active=True)
        return self

    def observe_group(self, *, keep: bool, reason: str | None) -> None:
        self._monitor.increment("batch/groups_completed_total")
        if not keep:
            self._monitor.increment("batch/groups_rejected_total")
            self._monitor.event(
                "batch_reject",
                rollout_id=self._rollout_id,
                details={"groups": 1, "reason": reason, "action": "drop"},
            )

    def selected(self, groups: int) -> None:
        self._monitor.increment("batch/groups_selected_total", groups)

    def accept_group(self) -> None:
        self._collected += 1
        self._publish(status="ready" if self._collected == self._target else "collecting", active=True)

    def __exit__(self, error_type, error, traceback):
        # The local collection is handed off on return; failures must not leave a full pool gauge behind.
        self._collected = 0
        status = "released" if error_type is None else "cancelled" if error_type.__name__ == "CancelledError" else "failed"
        self._publish(status=status, active=False)

    def _publish(self, *, status: str, active: bool) -> None:
        if not self._monitor.enabled:
            return
        self._monitor.update(
            {
                "batch/collected_groups": self._collected,
                "batch/target_groups": self._target,
                "batch/rollout_id": self._rollout_id,
                "batch/active": int(active),
            }
        )
        # Exact transitions preserve short collections that periodic sampling could miss.
        self._monitor.event(
            "batch_pool",
            rollout_id=self._rollout_id,
            details={"collected_groups": self._collected, "target_groups": self._target, "status": status},
        )
