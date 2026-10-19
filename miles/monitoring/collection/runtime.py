"""Opt-in process instrumentation. Business code only records observations.

Statistics owns bounded state; RuntimeWriter owns delivery and its existing
background thread. No filesystem or tracking work happens in a hot-path hook.
"""

from __future__ import annotations

import atexit
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from miles.monitoring.aggregation.runtime import RuntimeStatistics
from miles.monitoring.aggregation.workload import workload_of as workload_of
from miles.monitoring.aggregation.workload import workload_targets as workload_targets
from miles.monitoring.collection.host import _host_memory_metrics
from miles.monitoring.transport.config import add_perf_monitor_arguments as add_perf_monitor_arguments
from miles.monitoring.transport.config import validate_perf_monitor_args as validate_perf_monitor_args
from miles.monitoring.transport.runtime import RuntimeWriter, configure_tracking

_monitors: dict[str, RuntimeMonitor] = {}


class RuntimeMonitor:
    """Lightweight phase/event API and raw-gauge provider registry."""

    def __init__(self, *, args=None, role="disabled", sink=None, path: Path | None = None):
        self.enabled = args is not None and getattr(args, "perf_monitor_interval", 0) > 0
        self.role = role
        self.args = args
        self.interval = getattr(args, "perf_monitor_interval", 0)
        self.start_ts = getattr(args, "perf_monitor_start_ts", time.time())
        self._stats = RuntimeStatistics(enabled=self.enabled)
        self._lock = self._stats._lock
        self._providers = {}
        self._writer = RuntimeWriter(self, sink=sink, path=path)

    def start(self):
        self._writer.start()

    def stop(self):
        self._writer.stop()

    def emit(self):
        self._writer.emit()

    def update(self, values):
        self._stats.update(values)

    def increment(self, name, value=1):
        self._stats.increment(name, value)

    def observe(self, name, seconds):
        self._stats.observe(name, seconds)

    def _event(self, event):
        self._stats._event(event)

    @property
    def _events(self):
        return self._stats._events

    def _rates(self, values):
        return self._stats._rates(values)

    def _publish_events(self, events):
        self._writer._publish_events(events)

    def register(self, name: str, provider: Callable[[], Mapping[str, float]]) -> None:
        if self.enabled:
            with self._lock:
                self._providers[name] = provider

    @contextmanager
    def wait(self, name: str) -> Iterator[None]:
        """Time a real blocking wait, including cancellation; expose ongoing waits."""
        if not self.enabled:
            yield
            return
        started = time.monotonic()
        self._stats.begin_wait(name, started)
        try:
            yield
        finally:
            self._stats.end_wait(name, started)

    @contextmanager
    def phase(self, name: str, *, rollout_id: int | None = None) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        t0 = time.time()
        event = dict(kind="phase", role=self.role, name=name, rollout_id=rollout_id, t0=t0, t1=None)
        self._event(event | dict(status="running"))
        status, error_type = "completed", None
        try:
            with self.wait(f"phase/{name}"):
                yield
        except BaseException as error:
            error_type = type(error).__name__
            status = "cancelled" if error_type == "CancelledError" else "failed"
            self.increment(f"phase/{name}/{status}_total")
            raise
        finally:
            self._event(event | dict(t1=time.time(), status=status, error_type=error_type))

    def event(self, name: str, *, rollout_id: int | None = None, details: dict | None = None) -> None:
        """Record an instantaneous telemetry decision using explicit, non-sensitive fields."""
        if not self.enabled:
            return
        stamp = time.time()
        self._event(
            dict(
                kind="phase",
                role=self.role,
                name=name,
                rollout_id=rollout_id,
                t0=stamp,
                t1=stamp,
                status="completed",
                details=details,
            )
        )

    def completed_step(self, rollout_id: int, *, started: float) -> None:
        if not self.enabled:
            return
        ended = time.time()
        self.observe("step", ended - started)
        self._event(
            dict(
                kind="phase",
                role=self.role,
                name="step",
                rollout_id=rollout_id,
                t0=started,
                t1=ended,
                status="completed",
                error_type=None,
            )
        )

    def snapshot(self) -> dict[str, float]:
        values = self._stats.snapshot()
        with self._lock:
            providers = dict(self._providers)
        for name, provider in providers.items():
            try:
                values.update({f"{name}/{key}" if name else key: value for key, value in provider().items()})
            except Exception:
                self._writer._warn("performance gauge provider failed; leaving a gap")
        return {f"runtime/{self.role}/{key}": value for key, value in values.items()} | {
            "runtime/time_s": max(0.0, time.time() - self.start_ts)
        }


_disabled = RuntimeMonitor()


def get_monitor(args, role: str = "rollout") -> RuntimeMonitor:
    if getattr(args, "perf_monitor_interval", 0) <= 0:
        return _disabled
    if role not in _monitors:
        monitor = _monitors[role] = RuntimeMonitor(args=args, role=role)
        monitor.register("host", _host_memory_metrics)
    return _monitors[role]


def start_monitor(args, *, primary: bool, router_addr: str | None = None) -> None:
    role = "driver" if primary else "rollout" if router_addr is not None else "train"
    monitor = get_monitor(args, role)
    configure_tracking(monitor, primary=primary)
    monitor.start()


def stop_monitors() -> None:
    for monitor in list(_monitors.values()):
        monitor.stop()
    _monitors.clear()


atexit.register(stop_monitors)
