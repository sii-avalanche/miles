"""Opt-in, wall-clock telemetry. Never waits on the training or rollout path.

Counters are cumulative; duration percentiles describe the last 256 completed
operations. Live waits include operations that have not returned yet. Snapshots
go through the existing tracking backends and to a bounded local event stream.
No sample text, exception messages, environment values or credentials are saved.
"""

from __future__ import annotations

import atexit
import json
import logging
import math
import os
import re
import socket
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from miles.utils.perf_monitor_io import append_records, host_memory_metrics

logger = logging.getLogger(__name__)
_monitors: dict[str, RuntimeMonitor] = {}


def add_perf_monitor_arguments(parser) -> None:
    group = parser.add_argument_group("runtime performance monitoring")
    group.add_argument(
        "--perf-monitor-interval",
        type=float,
        default=0.0,
        help="Wall-clock telemetry interval (s); 0 disables it. Writes runtime/* to tracking and LOG_DIR/perf/.",
    )
    group.add_argument(
        "--perf-monitor-expert-layer",
        type=int,
        default=None,
        help="Optional 1-based MoE layer to count from rollout routing replay. Requires --use-rollout-routing-replay.",
    )
    group.add_argument(
        "--perf-monitor-workload-key",
        type=str,
        default=None,
        help="Optional Sample.metadata field identifying a workload (e.g. source). Requires workload targets.",
    )
    group.add_argument(
        "--perf-monitor-workload-targets",
        type=json.loads,
        default=None,
        help='Reference group fractions as JSON, e.g. {"math":0.7,"code":0.3}. Monitoring only; does not change scheduling.',
    )


def validate_perf_monitor_args(args) -> None:
    interval = args.perf_monitor_interval
    assert math.isfinite(interval) and interval >= 0, "--perf-monitor-interval must be finite and nonnegative"
    layer = args.perf_monitor_expert_layer
    targets = args.perf_monitor_workload_targets
    key = args.perf_monitor_workload_key
    assert (targets is None) == (key is None), "workload key and targets must be configured together"
    if targets is not None:
        assert interval > 0, "workload telemetry requires --perf-monitor-interval > 0"
        assert isinstance(targets, dict) and 1 <= len(targets) <= 32, "workload targets must have 1 to 32 entries"
        assert all(re.fullmatch(r"[A-Za-z0-9_]{1,48}", name) for name in targets), "workload names must be short slugs"
        assert all(isinstance(v, (int, float)) and math.isfinite(v) and 0 < v <= 1 for v in targets.values())
        assert math.isclose(sum(targets.values()), 1, abs_tol=1e-6), "workload target fractions must sum to 1"
    if layer is not None:
        assert interval > 0, "--perf-monitor-expert-layer requires --perf-monitor-interval > 0"
        assert args.use_rollout_routing_replay, "expert telemetry requires --use-rollout-routing-replay"
        assert 1 <= layer <= args.num_layers, "expert layer must be in [1, num_layers]"
        assert args.num_experts > 0, "expert telemetry requires num_experts > 0"


class RuntimeMonitor:
    """One process owns its state and background writer; providers only read gauges."""

    def __init__(self, *, args=None, role: str = "disabled", sink=None, path: Path | None = None):
        self.enabled = args is not None and getattr(args, "perf_monitor_interval", 0) > 0
        self.role = role
        self.args = args
        self.interval = getattr(args, "perf_monitor_interval", 0)
        self.start_ts = getattr(args, "perf_monitor_start_ts", time.time())
        self._sink = sink
        self._path = path
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._counters = defaultdict(float)
        self._gauges = {}
        self._durations = defaultdict(lambda: deque(maxlen=256))
        self._open: dict[str, list[float]] = defaultdict(list)
        self._providers: dict[str, Callable[[], Mapping[str, float]]] = {}
        self._events = deque(maxlen=2048)
        self._last_warning = 0.0
        self._last_rate_time = time.monotonic()
        self._last_rate_values = {}
        self._closed = False

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

    def update(self, values: Mapping[str, float]) -> None:
        if self.enabled:
            with self._lock:
                self._gauges.update(values)

    def increment(self, name: str, value: float = 1) -> None:
        if self.enabled:
            with self._lock:
                self._counters[name] += value

    def observe(self, name: str, seconds: float) -> None:
        if self.enabled:
            with self._lock:
                self._counters[f"{name}/count_total"] += 1
                self._counters[f"{name}/seconds_total"] += seconds
                self._durations[name].append(seconds)

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
        with self._lock:
            self._open[name].append(started)
        try:
            yield
        finally:
            with self._lock:
                self._open[name].remove(started)
            self.observe(name, time.monotonic() - started)

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

    def _event(self, event: dict) -> None:
        with self._lock:
            if len(self._events) == self._events.maxlen:
                self._counters["telemetry/events_dropped_total"] += 1
            self._events.append(event)

    def completed_step(self, rollout_id: int, *, started: float) -> None:
        if not self.enabled:
            return
        ended = time.time()
        self.increment("steps_completed_total")
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

    def _rates(self, values: dict[str, float]) -> dict[str, float]:
        now = time.monotonic()
        elapsed = max(now - self._last_rate_time, 1e-6)
        rates = {}
        for key, value in values.items():
            if key.endswith("_total") or key.endswith("/seconds_including_open"):
                if key not in self._last_rate_values and ("/host/" in key or "/pool/" in key):
                    # External counters can predate the monitor. Never interpret
                    # their first observed value as work/failures in this interval.
                    continue
                suffix = "_total" if key.endswith("_total") else "_including_open"
                rate_key = key.removesuffix(suffix) + "_per_s"
                rates[rate_key] = max(0, value - self._last_rate_values.get(key, 0)) / elapsed
        # Open waits take precedence over completed-only totals of the same wait.
        self._last_rate_values = dict(values)
        self._last_rate_time = now
        return rates

    def snapshot(self) -> dict[str, float]:
        now = time.monotonic()
        with self._lock:
            values = dict(self._counters) | self._gauges
            providers = dict(self._providers)
            for name, started in self._open.items():
                values[f"{name}/waiting"] = len(started)
                values[f"{name}/oldest_wait_seconds"] = max((now - t for t in started), default=0.0)
                values[f"{name}/seconds_including_open"] = values.get(f"{name}/seconds_total", 0) + sum(
                    now - t for t in started
                )
            for name, durations in self._durations.items():
                ordered = sorted(durations)
                values[f"{name}/window_count"] = len(ordered)
                for quantile in (50, 95):
                    values[f"{name}/p{quantile}_seconds"] = ordered[math.ceil(quantile / 100 * len(ordered)) - 1]
                values[f"{name}/max_seconds"] = ordered[-1]
        for name, provider in providers.items():
            try:
                values.update({f"{name}/{key}" if name else key: value for key, value in provider().items()})
            except Exception:
                self._warn("performance gauge provider failed; leaving a gap")
        return {f"runtime/{self.role}/{key}": value for key, value in values.items()} | {
            "runtime/time_s": max(0.0, time.time() - self.start_ts)
        }

    def emit(self) -> None:
        if not self.enabled:
            return
        try:
            metrics = self.snapshot()
            metrics.update(self._rates(metrics))
            with self._lock:
                events, self._events = list(self._events), deque(maxlen=2048)
            if self._path is not None:
                try:
                    append_records(self._path, [*events, dict(kind="metrics", ts=time.time(), metrics=metrics)])
                except Exception:
                    self._warn("performance log write failed; events dropped")
            if events and getattr(self.args, "use_miles_dashboard", False):
                try:
                    self._publish_events(events)
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
        from miles.dashboard.backend import current_collector
        from miles.dashboard.store import RuntimeEvent

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


_disabled = RuntimeMonitor()


def workload_targets(args) -> dict[str, float]:
    return getattr(args, "perf_monitor_workload_targets", None) or {"all": 1.0}


def workload_of(sample, args) -> str:
    if (targets := getattr(args, "perf_monitor_workload_targets", None)) is None:
        return "all"
    # Only explicitly configured labels enter metric names. Unknown metadata
    # maps to one bucket, keeping cardinality bounded and payloads private.
    value = (sample.metadata or {}).get(args.perf_monitor_workload_key)
    return value if isinstance(value, str) and value in targets else "other"


def get_monitor(args, role: str = "rollout") -> RuntimeMonitor:
    if getattr(args, "perf_monitor_interval", 0) <= 0:
        return _disabled
    if role not in _monitors:
        monitor = _monitors[role] = RuntimeMonitor(args=args, role=role)
        monitor.register("host", host_memory_metrics)
    return _monitors[role]


def start_monitor(args, *, primary: bool, router_addr: str | None = None) -> None:
    role = "driver" if primary else "rollout" if router_addr is not None else "train"
    monitor = get_monitor(args, role)
    if primary and monitor.enabled:
        from miles.utils.tracking_utils import tracking  # tracking imports this module

        tracking.define_step_key_metric_group("runtime", "runtime/time_s")
        if getattr(args, "perf_monitor_expert_layer", None) is not None:
            tracking.define_step_key_metric_group("moe", "rollout/step")
        if getattr(args, "use_miles_dashboard", False):
            monitor.register("fleet", _fleet_snapshot)
    monitor.start()


def _fleet_snapshot() -> dict[str, float]:
    import ray  # optional distributed dashboard backend

    from miles.dashboard.backend import current_collector

    if (handle := current_collector()) is None:
        return {}
    return ray.get(handle.runtime_snapshot.remote(), timeout=2)


def stop_monitors() -> None:
    for monitor in list(_monitors.values()):
        monitor.stop()
    _monitors.clear()


atexit.register(stop_monitors)
