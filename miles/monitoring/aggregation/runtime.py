"""Bounded runtime counters, gauges, waits, events and rate windows; no I/O."""

import math
import threading
import time
from collections import Counter, defaultdict, deque
from collections.abc import Mapping

from miles.monitoring.collection.records import MetricsRecord

_P95_STAGES = frozenset({"sample/generate", "sample/reward", "record/persist", "producer/group_generation"})
_RATE_COUNTERS = frozenset(
    {
        "groups_seen_total",
        "groups_enqueued_total",
        "groups_consumed_total",
        "groups_completed_total",
        "groups_selected_total",
        "groups_rejected_total",
        "stale_rejected_groups_total",
        "dynamic_rejected_groups_total",
        "aborted_groups_total",
        "missing_reward_groups_total",
        "process_cpu_seconds_total",
    }
)


class RuntimeStatistics:
    def __init__(self, *, enabled):
        self.enabled = enabled
        self._lock = threading.RLock()
        self._counters = defaultdict(float)
        self._gauges = {}
        self._durations = defaultdict(lambda: deque(maxlen=256))
        self._open = defaultdict(list)
        self._events = deque(maxlen=2048)
        self._last_rate_time = time.monotonic()
        self._last_rate_values = {}

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
                if name in _P95_STAGES or (name.startswith("buffer/") and name.endswith("/residence")):
                    self._durations[name].append(seconds)

    def _event(self, event: dict) -> None:
        with self._lock:
            if len(self._events) == self._events.maxlen:
                self._counters["telemetry/events_dropped_total"] += 1
            self._events.append(event)

    def _rates(self, values: dict[str, float]) -> dict[str, float]:
        now = time.monotonic()
        elapsed = max(now - self._last_rate_time, 1e-6)
        rates = {}
        for key, value in values.items():
            if key.rsplit("/", 1)[-1] in _RATE_COUNTERS:
                if key not in self._last_rate_values and "/host/" in key:
                    # External counters can predate the monitor. Never interpret
                    # their first observed value as work/failures in this interval.
                    continue
                rate_key = key.removesuffix("_total") + "_per_s"
                rates[rate_key] = max(0, value - self._last_rate_values.get(key, 0)) / elapsed
        self._last_rate_values = {
            key: value for key, value in values.items() if key.rsplit("/", 1)[-1] in _RATE_COUNTERS
        }
        self._last_rate_time = now
        return rates

    def snapshot(self) -> dict[str, float]:
        now = time.monotonic()
        with self._lock:
            values = dict(self._counters) | self._gauges
            for name, started in self._open.items():
                values[f"{name}/waiting"] = len(started)
                values[f"{name}/oldest_wait_seconds"] = max((now - t for t in started), default=0.0)
            for name, durations in self._durations.items():
                ordered = sorted(durations)
                values[f"{name}/p95_seconds"] = ordered[math.ceil(0.95 * len(ordered)) - 1]
        return values

    def begin_wait(self, name, started):
        with self._lock:
            self._open[name].append(started)

    def end_wait(self, name, started):
        with self._lock:
            self._open[name].remove(started)
        self.observe(name, time.monotonic() - started)

    def drain_events(self):
        with self._lock:
            events = list(self._events)
            self._events.clear()
        return events


_ROLLOUT = "runtime/rollout/"
_DRIVER = "runtime/driver/"
_STAGES = (("generate", "sample/generate"), ("reward", "sample/reward"), ("persist", "record/persist"))


def _number(value) -> float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _latest(records: list[MetricsRecord], prefix: str) -> MetricsRecord | None:
    return max((r for r in records if any(k.startswith(prefix) for k in r.metrics)), key=lambda r: r.ts, default=None)


def _freshness(record: MetricsRecord | None, now: float, max_age: float) -> dict:
    age = max(0, now - record.ts) if record else None
    return dict(observed_at=record.ts if record else None, age_seconds=age, fresh=age is not None and age <= max_age)


def runtime_status(records: list[MetricsRecord], *, interval_s: float = 10, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    max_age = max(30, 3 * (_number(interval_s) or 10))
    rollout = _latest(records, _ROLLOUT)
    driver = _latest(records, _DRIVER)
    metrics = rollout.metrics if rollout else {}

    def value(key: str) -> float | None:
        return _number(metrics.get(_ROLLOUT + key))

    stages = [
        dict(
            id=name,
            active=value(prefix + "/waiting"),
            oldest_seconds=value(prefix + "/oldest_wait_seconds"),
            finished_total=value(prefix + "/count_total"),
            p95_seconds=value(prefix + "/p95_seconds"),
        )
        for name, prefix in _STAGES
    ]
    stages.append(
        dict(
            id="database",
            active=value("record/in_flight_writes"),
            pending=value("record/pending_writes"),
            oldest_seconds=value("record/oldest_pending_seconds"),
            finished_total=value("record/success_total"),
        )
    )
    driver_metrics = driver.metrics if driver else {}
    active_phases = [
        dict(
            name=key[len(_DRIVER + "phase/") : -len("/waiting")],
            active=v,
            oldest_seconds=_number(driver_metrics.get(key.removesuffix("waiting") + "oldest_wait_seconds")),
        )
        for key, v in driver_metrics.items()
        if key.startswith(_DRIVER + "phase/") and key.endswith("/waiting") and (_number(v) or 0) > 0
    ]
    return dict(
        server_time=now,
        max_age_seconds=max_age,
        rollout={**_freshness(rollout, now, max_age), "stages": stages},
        driver={
            **_freshness(driver, now, max_age),
            "active_phases": active_phases,
            "rollout_id": _number(driver_metrics.get(_DRIVER + "rollout_id")),
        },
    )


def _identity(row: dict) -> tuple | None:
    # A group keeps its identity across retries; the same prompt in another epoch
    # has a different group_index and must not be confused with this retry chain.
    if row.get("group_index") is None:
        return None
    return row.get("trainer_model_id"), row["group_index"]


def _final_events(events: list[dict], *, order_by="t0") -> list[dict]:
    final = {}
    for event in events:
        key = (event["role"], event.get("rollout_id"), event["name"], event["t0"])
        if key not in final or event.get("t1") is not None:
            final[key] = event
    return sorted(final.values(), key=lambda event: event[order_by])


def _training_status(selection: dict, events: list[dict]) -> dict:
    matches = [
        event
        for event in events
        if event["name"] == "train"
        and event.get("rollout_id") == selection["batch_id"]
        and event["t0"] >= selection["ts"]
        and (event.get("details") or {}).get("trainer_model_id") == selection.get("trainer_model_id")
    ]
    event = matches[-1] if matches else None
    status = "selected"
    if event:
        status = (
            "training_completed"
            if event.get("t1") is not None and event["status"] == "completed"
            else ("training_failed" if event["status"] == "failed" else "training_running")
        )
    return {
        "status": status,
        "batch_id": selection["batch_id"],
        "generation_attempt": selection.get("generation_attempt"),
        "selected_at": selection["ts"],
        "completed_at": event.get("t1") if event else None,
    }


def group_history(events: list[dict], legacy: dict) -> dict:
    events = _final_events(events)
    recovered = {entry["ts"]: entry for entry in legacy.get("rejections", [])}
    rows, selections = [], []
    for event in events:
        details = event.get("details") or {}
        if event["name"] == "buffer_reject":
            row = {
                **recovered.get(event["t0"], {}),
                **details,
                "ts": event["t0"],
                "event": "buffer_reject",
                "timestamp_kind": "decision",
                "identity_source": "runtime" if "group_index" in details else "unknown",
            }
            if row.get("group_index") is not None and "group_index" not in details:
                row["identity_source"] = "recovered_fifo"
            rows.append(row)
        elif event["name"] == "training_batch_selected":
            selections.extend(
                {
                    **identity,
                    "ts": event["t0"],
                    "event": "training_batch_selected",
                    "batch_id": event["rollout_id"],
                    "trainer_model_id": details.get("trainer_model_id"),
                    "timestamp_kind": "decision",
                    "identity_source": "runtime",
                }
                for identity in details.get("groups", [])
            )
    selections.extend(legacy.get("selections", []))
    rows.extend(legacy.get("unbound_dynamic_groups", []))
    selected_by_group = {}
    for selection in sorted(selections, key=lambda entry: entry["ts"]):
        selected_by_group.setdefault(_identity(selection), []).append(selection)
    totals = Counter(_identity(row) for row in rows if row.get("reason") == "stale" and _identity(row) is not None)
    stale_seen = Counter()
    for row in sorted(rows + selections, key=lambda entry: entry["ts"]):
        key = _identity(row)
        if row.get("reason") == "stale" and key is not None:
            stale_seen[key] += 1
        row["stale_count_at_event"] = stale_seen[key] if key is not None else None
        row["stale_count_total"] = totals[key] if key is not None else None
        later = next((selection for selection in selected_by_group.get(key, []) if selection["ts"] >= row["ts"]), None)
        row["training"] = _training_status(later, events) if later else {"status": "unknown"}
    counts = Counter(row.get("reason") for row in rows if row.get("timestamp_kind") == "decision")
    return {
        "rows": sorted(rows + selections, key=lambda row: row["ts"], reverse=True),
        "counts": {
            "stale": counts["stale"],
            "dynamic_filter": counts["dynamic_filter"],
            "identified_stale": sum(row.get("reason") == "stale" and _identity(row) is not None for row in rows),
            "identified_dynamic": sum(
                row.get("reason") == "dynamic_filter" and _identity(row) is not None for row in rows
            ),
            "selected_groups": len(selections),
        },
        "coverage_note": legacy.get(
            "coverage_note", "Older decision events may not include prompt or group identities."
        ),
        "last_ts": max((event["t0"] for event in events), default=None),
    }


def event_window(records, *, t0=None, t1=None, limit=500):
    if t0 is None and t1 is None:
        return dict(events=[record.to_dict() for record in records[-limit:]], truncated=len(records) > limit)
    # Resolve open/closed twins before filtering, shared with group-history queries.
    source = _final_events([record.to_dict() for record in records], order_by="ts")
    events = [
        event
        for event in source
        if (t1 is None or event["t0"] <= t1) and (t0 is None or event["t1"] is None or event["t1"] >= t0)
    ]
    return dict(events=events[-limit:], truncated=len(events) > limit)
