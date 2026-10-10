import asyncio
import json
import time
from argparse import ArgumentParser, Namespace

import pytest

from miles.utils.perf_monitor import (
    RuntimeMonitor,
    add_perf_monitor_arguments,
    validate_perf_monitor_args,
    workload_of,
)


def monitor(tmp_path=None, sink=None):
    return RuntimeMonitor(
        args=Namespace(perf_monitor_interval=10.0, perf_monitor_start_ts=time.time()),
        role="test",
        sink=sink or (lambda metrics: None),
        path=tmp_path / "events.jsonl" if tmp_path else None,
    )


def test_live_waits_are_visible_before_completion():
    m = monitor()
    with m.wait("buffer/empty_wait"):
        time.sleep(0.01)
        snapshot = m.snapshot()
        assert snapshot["runtime/test/buffer/empty_wait/waiting"] == 1
        assert snapshot["runtime/test/buffer/empty_wait/oldest_wait_seconds"] >= 0.008
        assert "runtime/test/buffer/empty_wait/count_total" not in snapshot
    assert m.snapshot()["runtime/test/buffer/empty_wait/waiting"] == 0
    assert m.snapshot()["runtime/test/buffer/empty_wait/count_total"] == 1


async def test_cancelled_wait_releases_live_gauge():
    m = monitor()

    async def block():
        with m.wait("blocked"):
            await asyncio.Event().wait()

    task = asyncio.create_task(block())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert m.snapshot()["runtime/test/blocked/waiting"] == 0
    assert m.snapshot()["runtime/test/blocked/count_total"] == 1


def test_duration_memory_is_bounded_and_snapshot_does_not_reset_counters():
    m = monitor()
    for value in range(1000):
        m.observe("write", value)
    first = m.snapshot()
    second = m.snapshot()
    assert first["runtime/test/write/count_total"] == second["runtime/test/write/count_total"] == 1000
    assert first["runtime/test/write/window_count"] == 256
    assert first["runtime/test/write/p95_seconds"] == 987


def test_backend_failure_still_saves_observed_stage_without_error_payload(tmp_path):
    def broken_sink(metrics):
        raise RuntimeError("backend is down")

    m = monitor(tmp_path, sink=broken_sink)
    with pytest.raises(ValueError):
        with m.phase("train", rollout_id=3):
            raise ValueError("sensitive-payload")
    m.emit()
    raw = (tmp_path / "events.jsonl").read_text()
    rows = [json.loads(line) for line in raw.splitlines()]
    failure = next(row for row in rows if row.get("status") == "failed")
    assert failure["rollout_id"] == 3 and failure["error_type"] == "ValueError"
    assert "sensitive-payload" not in raw
    assert rows[-1]["kind"] == "metrics"


def test_completed_steps_use_wall_time_and_are_separate_from_optimizer_steps(tmp_path):
    m = monitor(tmp_path)
    m.completed_step(5, started=time.time() - 2)
    m.emit()
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert rows[0]["name"] == "step" and rows[0]["rollout_id"] == 5
    assert rows[0]["t1"] - rows[0]["t0"] >= 2
    assert rows[-1]["metrics"]["runtime/test/steps_completed_total"] == 1


def test_workload_labels_have_bounded_cardinality():
    args = Namespace(perf_monitor_workload_key="source", perf_monitor_workload_targets={"math": 0.7, "code": 0.3})
    assert workload_of(Namespace(metadata={"source": "math"}), args) == "math"
    assert workload_of(Namespace(metadata={"source": "private/problem/text"}), args) == "other"
    assert workload_of(Namespace(metadata={}), args) == "other"


def test_launch_validation_rejects_invalid_interval_and_inconsistent_workloads():
    parser = ArgumentParser()
    add_perf_monitor_arguments(parser)
    args = parser.parse_args(["--perf-monitor-interval", "nan"])
    with pytest.raises(AssertionError, match="finite"):
        validate_perf_monitor_args(args)
    args = parser.parse_args(["--perf-monitor-workload-key", "source"])
    with pytest.raises(AssertionError, match="together"):
        validate_perf_monitor_args(args)
