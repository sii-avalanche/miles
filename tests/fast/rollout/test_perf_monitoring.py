import asyncio
import time
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest

from miles.rollout import fully_async_data_buffer as buffers
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput, DefaultDataBuffer
from miles.rollout.moe_metrics import expert_load_metrics
from miles.utils.perf_monitor import RuntimeMonitor
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall


def make_buffer(monkeypatch):
    args = Namespace(
        async_data_buffer_capacity_factor=1,
        rollout_batch_size=1,
        dynamic_sampling_filter_path=None,
        max_weight_staleness=None,
        reward_key=None,
    )
    monitor = RuntimeMonitor(args=Namespace(perf_monitor_interval=10), role="rollout", sink=lambda _: None)
    monkeypatch.setattr(buffers, "get_monitor", lambda args: monitor)
    buffer = DefaultDataBuffer(DataBufferConstructorInput(args=args, unused_handler_fn=lambda _: None))
    return buffer, monitor


def entry(index):
    sample = Sample(index=index, reward=1.0, status=Sample.Status.COMPLETED)
    return DataBufferInput(prompt_group=[sample], group=[sample])


def versioned_entry(index, *versions, workload="math"):
    item = entry(index)
    item.group[0].weight_versions = [
        WeightVersionsPerCall(spans=[WeightVersionSpan(version=str(v), abs_start=0, abs_end=1)]) for v in versions
    ]
    item.prompt_group[0].metadata = {"source": workload}
    return item


async def test_pool_version_workload_counts_follow_publication_consumption_and_rejection(monkeypatch):
    buffer, monitor = make_buffer(monkeypatch)
    buffer._capacity = 10
    buffer._args.perf_monitor_workload_key = "source"
    buffer._args.perf_monitor_workload_targets = {"math": 0.5, "code": 0.5}
    # Configure before constructing the buffer, as a real run does.
    buffer = DefaultDataBuffer(DataBufferConstructorInput(args=buffer._args, unused_handler_fn=lambda _: None))
    buffer._capacity = 10
    buffer.set_weight_version(4)
    entries = [
        versioned_entry(0, 4),
        versioned_entry(1, 3, workload="code"),
        versioned_entry(2, 2, 4),
        versioned_entry(3, 1),
        versioned_entry(4, 0),
        entry(5),
    ]
    for item in entries:
        await buffer.put(item)
    values = monitor.snapshot()
    prefix = "runtime/rollout/buffer/"
    assert values[prefix + "queued_groups"] == 6
    assert [values[prefix + f"versions/lag_{lag}_groups"] for lag in range(4)] == [1, 1, 1, 1]
    assert values[prefix + "versions/older_groups"] == 1
    assert values[prefix + "versions/unknown_groups"] == 1
    assert values[prefix + "versions/mixed_groups"] == 1
    assert values[prefix + "workload/code/versions/lag_1_groups"] == 1
    assert values[prefix + "workload/other/queued_groups"] == 1
    assert await buffer.get(current_version=4) is entries[0]
    buffer.set_weight_version(5)
    values = monitor.snapshot()
    assert values[prefix + "current_weight_version"] == 5
    assert [values[prefix + f"versions/lag_{lag}_groups"] for lag in range(4)] == [0, 0, 1, 1]
    assert values[prefix + "versions/older_groups"] == 2
    buffer._args.max_weight_staleness = 1
    buffer._args.async_unused_samples_handler = "retry"
    assert await buffer.get(current_version=5) is entries[-1]
    values = monitor.snapshot()
    assert values[prefix + "queued_groups"] == 0
    assert values[prefix + "retry_groups_total"] == 4
    assert values[prefix + "versions/mixed_groups"] == 0
    rejects = [e for e in monitor._events if e["name"] == "buffer_reject"]
    assert len(rejects) == 4
    assert rejects[0]["details"] == dict(
        reason="stale",
        action="retry",
        groups=1,
        trainer_model_id=None,
        oldest_version=3,
        newest_version=3,
        current_version=5,
        queued_groups=4,
        workload="code",
    )
    assert all(e["t0"] == e["t1"] for e in rejects)


async def test_rejection_events_distinguish_drop_from_retry(monkeypatch):
    buffer, monitor = make_buffer(monkeypatch)
    buffer._args.async_unused_samples_handler = "retry"
    aborted = entry(0)
    aborted.group[0].status = Sample.Status.ABORTED
    missing = entry(1)
    missing.group[0].reward = None
    await buffer.put(aborted)
    await buffer.put(missing)
    decisions = [e["details"] for e in monitor._events if e["name"] == "buffer_reject"]
    assert [(d["reason"], d["action"]) for d in decisions] == [("aborted", "retry"), ("missing_reward", "drop")]
    assert monitor.snapshot()["runtime/rollout/buffer/queued_groups"] == 0


async def test_full_buffer_wait_and_cancellation_do_not_change_contents(monkeypatch):
    buffer, monitor = make_buffer(monkeypatch)
    first, second = entry(0), entry(1)
    await buffer.put(first)
    blocked = asyncio.create_task(buffer.put(second))
    await asyncio.sleep(0)
    snapshot = monitor.snapshot()
    assert snapshot["runtime/rollout/buffer/fill_fraction"] == 1
    assert snapshot["runtime/rollout/buffer/full_wait/waiting"] == 1
    blocked.cancel()
    with pytest.raises(asyncio.CancelledError):
        await blocked
    assert await buffer.get() is first
    assert monitor.snapshot()["runtime/rollout/buffer/groups_consumed_total"] == 1
    assert monitor.snapshot()["runtime/rollout/buffer/full_wait/waiting"] == 0


async def test_empty_wait_is_timed_and_telemetry_does_not_reset_rollout_reward(monkeypatch):
    buffer, monitor = make_buffer(monkeypatch)
    waiting = asyncio.create_task(buffer.get())
    await asyncio.sleep(0)
    assert monitor.snapshot()["runtime/rollout/buffer/empty_wait/waiting"] == 1
    first = entry(0)
    await buffer.put(first)
    monitor.emit()
    assert await waiting is first
    assert buffer.get_metrics()["rollout/raw_reward_unfiltered"] == 1
    snapshot = monitor.snapshot()
    assert snapshot["runtime/rollout/buffer/empty_wait/waiting"] == 0
    assert snapshot["runtime/rollout/buffer/residence/count_total"] == 1


def test_expert_metrics_match_known_assignments_and_ignore_missing_replay():
    samples = [Sample(rollout_routed_experts=np.array([[[0, 1]], [[0, 1]]], dtype=np.int32)), Sample()]
    values = expert_load_metrics(samples, layer=1, num_experts=4)
    prefix = "moe/rollout_layer_1/"
    assert values[prefix + "cv"] == 1
    assert values[prefix + "max_over_mean"] == 2
    assert values[prefix + "cold_experts_fraction"] == 0.5
    assert values[prefix + "sample_coverage"] == 0.5
    assert values[prefix + "assignments"] == 4


def test_no_expert_assignments_does_not_fabricate_balance():
    values = expert_load_metrics([Sample()], layer=1, num_experts=4)
    assert "moe/rollout_layer_1/cv" not in values
    assert values["moe/rollout_layer_1/sample_coverage"] == 0


async def test_record_timeout_is_counted_and_does_not_fail_rollout(monkeypatch):
    from psycopg_pool import PoolTimeout

    from miles.rollout.avacore_rollout import AvaCoreRollout

    class Run:
        async def create_rollout(self, **kwargs):
            raise PoolTimeout("pool has no connections")

    adapter = object.__new__(AvaCoreRollout)
    adapter.args = Namespace(n_samples_per_prompt=16)
    adapter.failed_writes = 0
    adapter._monitor = RuntimeMonitor(args=Namespace(perf_monitor_interval=10), role="rollout", sink=lambda _: None)
    adapter.run = asyncio.get_running_loop().create_future()
    adapter.run.set_result(Run())
    sample = Sample(epoch=0, index=0, metadata={"_index": 0})
    trace = SimpleNamespace(last_assistant=lambda: SimpleNamespace(metadata={"weight_version": 1}))
    await adapter.record({}, sample, trace, None, {})
    metrics = adapter._monitor.snapshot()
    assert adapter.failed_writes == metrics["runtime/rollout/record/pool_timeout_total"] == 1
    assert metrics["runtime/rollout/record/end_to_end/count_total"] == 1
    assert "runtime/rollout/record/success_total" not in metrics


async def test_fully_async_collection_tracks_workload_targets_without_rescheduling(monkeypatch):
    from tests.fast.rollout.test_fully_async_rollout import FakeDataSource, make_args, make_fn, make_group

    from miles.rollout import fully_async_rollout
    from miles.rollout.base_types import RolloutFnTrainInput

    groups = [make_group(1), make_group(2)]
    for group in groups:
        for sample in group:
            sample.metadata["source"] = "math"
    args = make_args(rollout_batch_size=2)
    args.perf_monitor_workload_key = "source"
    args.perf_monitor_workload_targets = {"math": 0.5, "code": 0.5}
    monitor = RuntimeMonitor(args=Namespace(perf_monitor_interval=10), role="rollout", sink=lambda _: None)
    monkeypatch.setattr(fully_async_rollout, "get_monitor", lambda args: monitor)
    monkeypatch.setattr(buffers, "get_monitor", lambda args: monitor)
    fn = make_fn(monkeypatch, args, FakeDataSource(scripted=groups))
    try:
        fn.set_weight_version(3)
        output = await fn(RolloutFnTrainInput(rollout_id=0, weight_version=3))
        assert len(output.samples) == 2
        values = monitor.snapshot()
        assert values["runtime/rollout/drain/progress_fraction"] == 1
        assert values["runtime/rollout/workload/math/collection_progress_fraction"] == 2
        assert values["runtime/rollout/workload/code/collection_progress_fraction"] == 0
        assert values["runtime/rollout/workload/math/target_fraction"] == 0.5
        batch = next(e for e in monitor._events if e["name"] == "batch_ready")
        assert batch["details"]["current_version"] == 3
        assert batch["details"]["versions"] == {"unknown": 2}
        assert batch["details"]["workload_versions"] == {"math": {"unknown": 2}}
        fn.set_weight_version(4)
        assert monitor.snapshot()["runtime/rollout/buffer/current_weight_version"] == 4
        fn._last_heartbeat = time.monotonic() - 10
        assert monitor.snapshot()["runtime/rollout/event_loop/heartbeat_age_seconds"] >= 10
    finally:
        await fn.dispose()
