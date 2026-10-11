import asyncio
import copy
import gzip
import json
import threading

import pytest

from miles.rollout import rollout_recording
from miles.rollout.rollout_recording import DurableRolloutOutbox
from miles.utils.types import Sample


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
    atomic_json = rollout_recording._atomic_json
    saved = []

    def blocked_persistence(path, value, **kwargs):
        loop.call_soon_threadsafe(writing.set)
        assert release.wait(timeout=5)
        atomic_json(path, value, **kwargs)

    async def write(value):
        saved.append(value)

    recorder = outbox(tmp_path, write)
    await recorder.start()
    monkeypatch.setattr(rollout_recording, "_atomic_json", blocked_persistence)
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
