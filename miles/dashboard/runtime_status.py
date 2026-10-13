"""Read current rollout stages from existing telemetry, independently of the viewport."""

from __future__ import annotations

import math
import time
import tomllib
from pathlib import Path

from miles.dashboard.store import MetricsRecord

_ROLLOUT = "runtime/rollout/"
_DRIVER = "runtime/driver/"
_STAGES = (("generate", "sample/generate"), ("reward", "sample/reward"), ("persist", "record/persist"))


def reward_info(config_path: Path) -> dict:
    """Expose only judge identity and limits; never return endpoints, keys or env."""
    try:
        with config_path.open("rb") as file:
            reward = tomllib.load(file).get("reward", {})
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    if not isinstance(reward, dict):
        return {}
    result = {"kind": reward.get("kind")} if isinstance(reward.get("kind"), str) else {}
    if result.get("kind") != "lmaaj":
        return result
    model = reward.get("model", {})
    if not isinstance(model, dict):
        return result
    name = model.get("name")
    if isinstance(name, str) and "${" not in name:
        result["model_name"] = name
    for key in ("concurrency", "timeout", "max_retries"):
        value = _number(model.get(key))
        if value is not None:
            result[key] = value
    return result


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
