"""Background runtime delivery to JSONL, Tracking and the existing Ray collector."""

import json
import logging
import os
import socket
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class RuntimeWriter:
    def __init__(self, monitor, *, sink=None, path=None):
        self.monitor = monitor
        self.enabled = monitor.enabled
        self.args = monitor.args
        self.role = monitor.role
        self.interval = monitor.interval
        self._sink = sink
        self._path = path
        self._stop = threading.Event()
        self._thread = None
        self._closed = False
        self._last_warning = 0.0

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        if self._path is None:
            directory = os.environ.get("LOG_DIR") or getattr(self.args, "dashboard_dir", None)
            if directory is None and getattr(self.args, "dump_details", None):
                directory = self.args.dump_details
            if directory:
                self._path = Path(directory) / "perf" / f"{self.role}-{socket.gethostname()}-{os.getpid()}.jsonl"
        self._thread = threading.Thread(target=self._run, name=f"perf-{self.role}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        # Avoid two simultaneous writers when a tracking backend is stuck.
        if self._thread is None or not self._thread.is_alive():
            self.emit()

    def emit(self) -> None:
        if not self.enabled:
            return
        try:
            metrics = self.monitor.snapshot()
            metrics.update(self.monitor._stats._rates(metrics))
            events = self.monitor._stats.drain_events()
            if self._path is not None:
                try:
                    _append_records(self._path, [*events, dict(kind="metrics", ts=time.time(), metrics=metrics)])
                except Exception:
                    self._warn("performance log write failed; events dropped")
            if events and getattr(self.args, "use_miles_dashboard", False):
                try:
                    self.monitor._publish_events(events)
                except Exception:
                    self._warn("performance event push failed; other tracking continues")
            # A separate backend failure cannot prevent local records from landing.
            if self._sink is not None:
                self._sink(metrics)
            else:
                from miles.utils.tracking_utils import tracking  # optional backends; breaks import cycle

                tracking.log(self.args, metrics, step_key="runtime/time_s")
        except Exception:
            self._warn("performance tracking failed; next interval will retry")

    def _publish_events(self, events: list[dict]) -> None:
        # Optional backend; never resolve an actor on the hot path.
        from miles.monitoring.collection.records import RuntimeEvent
        from miles.monitoring.transport.backend import current_collector

        if (handle := current_collector()) is not None:
            handle.push_runtime_events.remote(
                [
                    RuntimeEvent(ts=event["t1"] or event["t0"], **{k: v for k, v in event.items() if k != "kind"})
                    for event in events
                ]
            )

    def _warn(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_warning > 60:
            logger.warning(message, exc_info=True)
            self._last_warning = now

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.emit()


def _append_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")


def _fleet_snapshot() -> dict[str, float]:
    import ray  # optional distributed dashboard backend

    from miles.monitoring.transport.backend import current_collector

    if (handle := current_collector()) is None:
        return {}
    return ray.get(handle.runtime_snapshot.remote(), timeout=2)


def configure_tracking(monitor, *, primary):
    if not primary or not monitor.enabled:
        return
    # Tracking imports the instrumentation API; initialize it only at run startup.
    from miles.utils.tracking_utils import tracking

    tracking.define_step_key_metric_group("runtime", "runtime/time_s")
    if getattr(monitor.args, "perf_monitor_expert_layer", None) is not None:
        tracking.define_step_key_metric_group("moe", "rollout/step")
    if getattr(monitor.args, "use_miles_dashboard", False):
        monitor.register("fleet", _fleet_snapshot)
