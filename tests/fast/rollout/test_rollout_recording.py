import asyncio
import copy
import gzip
import json
import threading
from argparse import Namespace

import pytest

from miles.utils.recording import outbox as recording_outbox
from miles.utils.recording.outbox import DurableRolloutOutbox
from miles.utils.types import Sample


@pytest.mark.parametrize("version", ["default", "007", 3, None])
async def test_avacore_record_preserves_original_weight_version(tmp_path, version):
    from ava_core.core import ChatMessage, Trace
    from ava_core.generate.core import Sample as Row
    from ava_core.rewards.core import Reward

    from miles.rollout.avacore_rollout import AvaCoreRollout
    from miles.utils.perf_monitor import RuntimeMonitor

    saved = []

    async def write(value):
        saved.append(value)

    async def prepared(_params):
        pass

    adapter = object.__new__(AvaCoreRollout)
    adapter.args = Namespace(n_samples_per_prompt=2)
    adapter._monitor = RuntimeMonitor(args=Namespace(perf_monitor_interval=10), role="rollout", sink=lambda _: None)
    adapter._prepare_recording = prepared
    adapter._outbox = outbox(tmp_path, write)
    trace = Trace(message=ChatMessage(role="assistant", content="answer", metadata={"weight_version": version}))
    sample = Sample(index=0, group_index=0, metadata={"_index": 7})
    await adapter._outbox.start()
    try:
        await adapter.record(Row({"prompt": "question"}, key=lambda _: "7"), sample, (trace,), Reward(score=1), {}, trial_id=0)
        assert await adapter._outbox.flush(timeout=2)
        assert sample.recording_source == str(tmp_path)
        assert sample.recording_trial_id == 0
        assert sample.metadata == {"_index": 7}
        assert saved[0]["metadata"]["weight_version"] == version
        assert type(saved[0]["metadata"]["weight_version"]) is type(version)
        committed = next((tmp_path / "committed").glob("*.json.gz"))
        with gzip.open(committed, "rt") as stream:
            assert json.load(stream)["metadata"]["weight_version"] == version
    finally:
        await adapter._outbox.close(timeout=2)


def record(trial_id=0):
    return {
        "query_id": "prompt-7",
        "trial_id": trial_id,
        "instance": {"prompt": "question"},
        "trace": {"messages": [{"role": "assistant", "content": "answer"}], "tokens": [1, 2]},
        "reward": {"score": 1.0},
        "status": "completed",
        "metadata": {"weight_version": 3, "generation_attempt": 0},
    }


def outbox(path, write, *, workers=1):
    return DurableRolloutOutbox(
        path,
        write,
        workers=workers,
        retry_initial_seconds=0.001,
        retry_max_seconds=0.005,
    )


@pytest.mark.parametrize("matches", [False, True])
async def test_pinned_database_run_is_checked_before_opening_or_creating(matches, monkeypatch):
    from contextlib import AsyncExitStack, asynccontextmanager
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from miles.rollout import avacore_rollout

    existing = SimpleNamespace(kind="rl", model="model" if matches else "wrong", run="old",
                               collection="data", model_step=None, collection_version=None)
    run = SimpleNamespace(update=AsyncMock())

    @asynccontextmanager
    async def opened_run(**kwargs):
        assert kwargs["resume"] is True
        assert kwargs["run"] == "old"
        yield run

    store = SimpleNamespace(pool=SimpleNamespace(), get_run=AsyncMock(return_value=existing),
                            rl_run=Mock(side_effect=opened_run))

    @asynccontextmanager
    async def backend(*args, **kwargs):
        yield store

    monkeypatch.setattr(avacore_rollout, "PostgresBackend", backend)
    adapter = object.__new__(avacore_rollout.AvaCoreRollout)
    adapter.recording = {"postgres": "unused", "model": "model", "run": "old", "expected_run_id": 6907}
    adapter.args = Namespace(prompt_data="data.jsonl")
    adapter.config = {}
    adapter._pool_min = adapter._pool_max = 1
    adapter._pool_timeout = 2
    async with AsyncExitStack() as adapter.stack:
        if matches:
            assert await adapter.open({}) is run
            store.rl_run.assert_called_once()
        else:
            with pytest.raises(ValueError, match="expected_run_id"):
                await adapter.open({})
            store.rl_run.assert_not_called()
        store.get_run.assert_awaited_once_with(6907)


async def test_outage_keeps_durable_snapshot_until_database_recovers(tmp_path):
    calls = []
    attempted = asyncio.Event()
    healthy = asyncio.Event()

    async def write(value):
        calls.append(copy.deepcopy(value))
        attempted.set()
        if not healthy.is_set():
            raise ConnectionError("database unavailable")

    recorder = outbox(tmp_path, write)
    await recorder.start()
    original = record()
    expected = copy.deepcopy(original)
    try:
        await recorder.enqueue(original)
        original["metadata"]["generation_attempt"] = 8
        original["trace"]["tokens"].clear()
        await asyncio.wait_for(attempted.wait(), timeout=1)
        assert not await recorder.flush(timeout=0.01)
        pending = list((tmp_path / "pending").glob("*.json.gz"))
        assert len(pending) == 1
        with gzip.open(pending[0], "rt") as stream:
            assert json.load(stream) == expected
        healthy.set()
        assert await recorder.flush(timeout=1)
        assert calls and all(value == expected for value in calls)
        assert not list((tmp_path / "pending").glob("*.json.gz"))
        assert len(list((tmp_path / "committed").glob("*.json.gz"))) == 1
    finally:
        await recorder.close(timeout=1)


async def test_pending_records_replay_after_shutdown_and_restart(tmp_path):
    attempted = asyncio.Event()

    async def unavailable(value):
        attempted.set()
        raise ConnectionError("database unavailable")

    first = outbox(tmp_path, unavailable)
    await first.start()
    await first.enqueue(record(0))
    await first.enqueue(record(1))
    await asyncio.wait_for(attempted.wait(), timeout=1)
    await first.close(timeout=0.01)
    assert len(list((tmp_path / "pending").glob("*.json.gz"))) == 2

    saved = []

    async def recovered(value):
        saved.append(value)

    second = outbox(tmp_path, recovered)
    await second.start()
    try:
        assert await second.flush(timeout=1)
        assert sorted(value["trial_id"] for value in saved) == [0, 1]
        assert not list((tmp_path / "pending").glob("*.json.gz"))
    finally:
        await second.close(timeout=1)


async def test_capture_before_workers_start_is_delivered_once(tmp_path):
    saved = []

    async def write(value):
        saved.append(value)

    recorder = outbox(tmp_path, write)
    expected = record()
    path = await recorder.enqueue(expected)
    assert path.exists()
    assert saved == []
    await recorder.start()
    try:
        assert await recorder.flush(timeout=1)
        assert saved == [expected]
        assert not path.exists()
        assert (recorder.committed / path.name).exists()
    finally:
        await recorder.close(timeout=1)


async def test_database_writers_have_bounded_concurrency(tmp_path):
    gate = asyncio.Event()
    reached_limit = asyncio.Event()
    active = 0
    peak = 0
    saved = []

    async def write(value):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == 3:
            reached_limit.set()
        try:
            await gate.wait()
            saved.append(value["trial_id"])
        finally:
            active -= 1

    recorder = outbox(tmp_path, write, workers=3)
    await recorder.start()
    try:
        for trial_id in range(12):
            await recorder.enqueue(record(trial_id))
        await asyncio.wait_for(reached_limit.wait(), timeout=1)
        assert peak == 3
        gate.set()
        assert await recorder.flush(timeout=1)
        assert sorted(saved) == list(range(12))
        assert peak == 3
    finally:
        gate.set()
        await recorder.close(timeout=1)


async def test_ambiguous_commit_retries_the_same_identity_and_payload(tmp_path):
    database = {}
    calls = []

    async def write(value):
        calls.append(copy.deepcopy(value))
        database[value["query_id"], value["trial_id"]] = copy.deepcopy(value)
        if len(calls) == 1:
            raise ConnectionError("connection lost after server committed")

    recorder = outbox(tmp_path, write)
    await recorder.start()
    try:
        expected = record(4)
        await recorder.enqueue(expected)
        assert await recorder.flush(timeout=1)
        assert calls == [expected, expected]
        assert database == {("prompt-7", 4): expected}
    finally:
        await recorder.close(timeout=1)


async def test_cancellation_after_commit_preserves_replayable_record(tmp_path):
    database = {}
    committed = asyncio.Event()
    blocked = asyncio.Event()

    async def commit_then_hang(value):
        database[value["query_id"], value["trial_id"]] = copy.deepcopy(value)
        committed.set()
        await blocked.wait()

    first = outbox(tmp_path, commit_then_hang)
    await first.start()
    await first.enqueue(record(2))
    await asyncio.wait_for(committed.wait(), timeout=1)
    await first.close(timeout=0.01)
    assert len(list((tmp_path / "pending").glob("*.json.gz"))) == 1

    async def recover(value):
        database[value["query_id"], value["trial_id"]] = copy.deepcopy(value)

    second = outbox(tmp_path, recover)
    await second.start()
    try:
        assert await second.flush(timeout=1)
        assert database == {("prompt-7", 2): record(2)}
    finally:
        await second.close(timeout=1)


async def test_duplicate_submissions_do_not_overwrite_or_block_the_queue(tmp_path):
    gate = asyncio.Event()
    saved = []

    async def write(value):
        await gate.wait()
        saved.append(value)

    recorder = outbox(tmp_path, write)
    await recorder.start()
    try:
        expected = record(3)
        first_path = await recorder.enqueue(expected)
        assert await recorder.enqueue(copy.deepcopy(expected)) == first_path
        conflicting = record(3)
        conflicting["reward"]["score"] = 0.0
        with pytest.raises(ValueError, match="different recording"):
            await recorder.enqueue(conflicting)
        gate.set()
        assert await recorder.flush(timeout=1)
        assert saved == [expected]
        committed = await recorder.enqueue(expected)
        assert committed.parent == tmp_path / "committed"
        assert await recorder.flush(timeout=1)
        assert saved == [expected]
    finally:
        gate.set()
        await recorder.close(timeout=1)


async def test_telemetry_failure_cannot_interrupt_database_acknowledgement(tmp_path, caplog):
    saved = []

    async def write(value):
        saved.append(value)

    def broken_telemetry(event, error):
        raise RuntimeError("metrics sink unavailable")

    recorder = DurableRolloutOutbox(tmp_path, write, workers=1, on_event=broken_telemetry)
    await recorder.start()
    try:
        expected = record(5)
        await recorder.enqueue(expected)
        assert await recorder.flush(timeout=1)
        assert saved == [expected]
        assert recorder.stats()["pending_writes"] == 0
        assert "Recording telemetry callback failed" in caplog.text
    finally:
        await recorder.close(timeout=1)


async def test_producer_cancellation_finishes_disk_capture_and_queues_record(tmp_path, monkeypatch):
    writing = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    atomic_json = recording_outbox._atomic_json
    saved = []

    def blocked_persistence(path, value, **kwargs):
        loop.call_soon_threadsafe(writing.set)
        assert release.wait(timeout=5)
        atomic_json(path, value, **kwargs)

    async def write(value):
        saved.append(value)

    recorder = outbox(tmp_path, write)
    await recorder.start()
    monkeypatch.setattr(recording_outbox, "_atomic_json", blocked_persistence)
    expected = record(6)
    capture = asyncio.create_task(recorder.enqueue(expected))
    try:
        await asyncio.wait_for(writing.wait(), timeout=1)
        capture.cancel()
        await asyncio.sleep(0)
        assert not capture.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await capture
        assert await recorder.flush(timeout=1)
        assert saved == [expected]
        assert not list((tmp_path / "pending").glob("*.json.gz"))
    finally:
        release.set()
        await asyncio.gather(capture, return_exceptions=True)
        await recorder.close(timeout=1)


def sample(slot, *, epoch=0, group_index=0):
    return Sample(epoch=epoch, index=group_index * 16 + slot, group_index=group_index, metadata={"_index": 7})


async def test_trials_keep_group_rounds_together_and_advance_on_retry(tmp_path):
    async def write(value):
        pass

    recorder = outbox(tmp_path, write)
    await recorder.start()
    try:
        samples = [sample(slot) for slot in range(16)]
        first = [await recorder.allocate_trial(value, 16) for value in reversed(samples)]
        assert first == list(reversed(range(16)))
        for value in samples:
            value.reset_for_retry()
        assert [await recorder.allocate_trial(value, 16) for value in samples] == list(range(16, 32))
        assert samples[0].epoch == 0
    finally:
        await recorder.close(timeout=1)


async def test_fresh_resumed_samples_cannot_reuse_persisted_trial_ids(tmp_path):
    async def write(value):
        pass

    first = outbox(tmp_path, write)
    await first.start()
    assert await first.allocate_trial(sample(0), 16) == 0
    await first.close(timeout=1)

    second = outbox(tmp_path, write)
    await second.start()
    try:
        assert await second.allocate_trial(sample(0), 16) == 16
        assert await second.allocate_trial(sample(1), 16) == 17
    finally:
        await second.close(timeout=1)


async def test_legacy_database_trials_seed_new_monotonic_rounds(tmp_path):
    async def write(value):
        pass

    recorder = outbox(tmp_path, write)
    await recorder.start()
    try:
        await recorder.seed_trials([("7", 31), ("other-query", 300)], 16)
        assert await recorder.allocate_trial(sample(0), 16) == 32
        assert await recorder.allocate_trial(sample(15), 16) == 47
        later = sample(0, epoch=1, group_index=2)
        assert await recorder.allocate_trial(later, 16) == 48
    finally:
        await recorder.close(timeout=1)
