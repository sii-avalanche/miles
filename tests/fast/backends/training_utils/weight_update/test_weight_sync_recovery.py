import asyncio
import threading
from argparse import Namespace
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from miles.backends.training_utils.weight_update import recovery
from miles.backends.training_utils.weight_update.protocols import broadcast
from miles.backends.training_utils.weight_update.updater import WeightUpdater


def _updater(monkeypatch, failure=None):
    events = []
    attempts = 0

    def iterate(*args, materialize, **kwargs):
        nonlocal attempts
        attempts += 1
        for bucket in range(3):
            events.append(("gather", attempts, bucket))
            if materialize:
                yield [bucket]

    def send(bucket):
        events.append(("send", attempts, bucket[0]))
        if bucket == [1] and (attempts == 1 or failure == "always"):
            if failure == "fatal":
                raise ValueError("receiver failure")
            raise broadcast.WeightSyncResponseError("lost reply")

    protocol = SimpleNamespace(
        is_sender=True,
        rollout_engines=[object()],
        send_bucket=send,
        recover_connection=lambda: events.append("reconnect"),
        after_base_weights=lambda: None,
        after_engines_resumed=lambda: None,
    )
    updater = SimpleNamespace(
        args=Namespace(colocate=False, update_weight_transfer_mode="broadcast"),
        is_lora=False,
        parallel_state=SimpleNamespace(pp=SimpleNamespace(size=1)),
        protocol=protocol,
        weight_version=7,
        weights_getter=lambda: {},
        _hf_weight_iterator=SimpleNamespace(iter_hf_weights=iterate, weight_update_selector="all"),
    )
    monkeypatch.setattr(recovery.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(recovery.dist, "all_reduce", lambda *a, **k: None)
    monkeypatch.setattr(recovery, "get_gloo_group", lambda: None)
    for name in ("pause_engines", "begin_weight_update", "end_weight_update", "resume_engines"):
        monkeypatch.setattr(recovery, name, lambda *a, _name=name, **k: events.append(_name))
    monkeypatch.setattr(recovery, "set_weight_version", lambda engines, version: events.append(("version", version)))
    monkeypatch.setattr(recovery, "_verify_versions", lambda engines, version: events.append("verified"))
    monkeypatch.setattr(recovery, "_abort_session", lambda engines: events.append("abort"))

    async def reset():
        events.append("reset_http")

    monkeypatch.setattr(recovery.GeneralHttpClientProvider, "reset", reset)
    return updater, events


def test_replays_from_first_bucket_and_publishes_once(monkeypatch):
    updater, events = _updater(monkeypatch)
    recovery.update_weights_with_retry(updater, retries=1)
    assert updater.weight_version == 8
    assert [e for e in events if isinstance(e, tuple) and e[0] == "send"] == [
        ("send", 1, 0),
        ("send", 1, 1),
        ("send", 2, 0),
        ("send", 2, 1),
        ("send", 2, 2),
    ]
    assert events.index(("gather", 1, 2)) < events.index("abort") < events.index("reconnect")
    assert events.index("reconnect") < events.index(("send", 2, 0))
    assert events.count("resume_engines") == 1
    assert events.count(("version", 8)) == 1
    assert events[-1] == "resume_engines"


@pytest.mark.parametrize("failure", ["always", "fatal", "reconnect"])
def test_failure_never_publishes_or_resumes(monkeypatch, failure):
    updater, events = _updater(monkeypatch, failure)
    if failure == "reconnect":
        updater.protocol.recover_connection = Mock(side_effect=RuntimeError("cannot reconnect"))
    with pytest.raises(RuntimeError):
        recovery.update_weights_with_retry(updater, retries=1)
    assert updater.weight_version == 7
    assert "resume_engines" not in events
    assert not any(isinstance(e, tuple) and e[0] == "version" for e in events)
    if failure == "fatal":
        assert "reconnect" not in events


def test_nonsender_drains_gathers_before_collective_failure(monkeypatch):
    updater, events = _updater(monkeypatch)
    updater.protocol.is_sender = False

    def remote_failure(status, **kwargs):
        assert events == [("gather", 1, i) for i in range(3)]
        status.fill_(1)

    monkeypatch.setattr(recovery.dist, "all_reduce", remote_failure)
    with pytest.raises(broadcast.WeightSyncResponseError):
        recovery._stream(updater)


@pytest.mark.parametrize(
    "errors,retryable", [([httpx.ReadError("lost")], True), ([httpx.ReadError("lost"), ValueError("bad")], False)]
)
def test_only_settled_read_errors_are_retryable(monkeypatch, errors, retryable):
    futures = []
    for error in errors:
        future = Future()
        future.set_exception(error)
        futures.append(future)
    last = Future()
    last.set_result({"success": True})
    futures.append(last)
    stream = Mock()
    monkeypatch.setattr(broadcast.torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(broadcast, "update_weights_from_distributed", lambda *a, **k: futures)
    protocol = SimpleNamespace(
        _engine_lock=nullcontext(),
        group_name="test",
        _model_update_groups=None,
        rollout_engines=[],
        _selector="all",
        args=Namespace(update_weight_sync_retries=1),
    )
    with pytest.raises(broadcast.WeightSyncResponseError if retryable else ValueError):
        broadcast.UpdateWeightFromDistributed.send_bucket(protocol, [])
    stream.synchronize.assert_called_once()


def test_version_verification_failure_does_not_resume(monkeypatch):
    updater, events = _updater(monkeypatch)
    monkeypatch.setattr(recovery, "_verify_versions", Mock(side_effect=RuntimeError("wrong version")))
    with pytest.raises(RuntimeError):
        recovery.update_weights_with_retry(updater, retries=1)
    assert "resume_engines" not in events
    assert updater.weight_version == 7


def test_abort_requires_positive_acknowledgement():
    class Engine:
        async def abort_weight_update(self):
            return {"success": False, "message": "session still active"}

    with pytest.raises(RuntimeError, match="Cannot confirm"):
        recovery._abort_session([Engine()])


def test_two_ranks_replay_together_without_nonsender_buckets(monkeypatch):
    local = threading.local()
    sync = threading.Barrier(2, timeout=5)
    gathers = threading.Barrier(2, timeout=5)
    statuses = [0, 0]
    updaters = []
    events = []

    def all_reduce(status, **kwargs):
        statuses[local.rank] = status.item()
        sync.wait()
        maximum = max(statuses)
        sync.wait()
        status.fill_(maximum)

    monkeypatch.setattr(recovery.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(recovery.dist, "get_rank", lambda: local.rank)
    monkeypatch.setattr(recovery, "get_gloo_group", lambda: None)
    for name in (
        "pause_engines",
        "begin_weight_update",
        "end_weight_update",
        "resume_engines",
        "set_weight_version",
        "_abort_session",
        "_verify_versions",
    ):
        monkeypatch.setattr(recovery, name, lambda *a, _name=name, **k: events.append(_name))

    async def reset():
        pass

    monkeypatch.setattr(recovery.GeneralHttpClientProvider, "reset", reset)

    def run(rank):
        local.rank = rank
        attempt = 0

        def iterate(*a, materialize, **kw):
            nonlocal attempt
            attempt += 1
            for bucket in range(3):
                gathers.wait()
                if materialize:
                    yield [bucket]

        def send(bucket):
            events.append(("send", attempt, bucket[0]))
            if attempt == 1 and bucket == [1]:
                raise broadcast.WeightSyncResponseError("lost")

        updater = SimpleNamespace(
            args=Namespace(colocate=False, update_weight_transfer_mode="broadcast"),
            is_lora=False,
            parallel_state=SimpleNamespace(pp=SimpleNamespace(size=1)),
            weight_version=4,
            weights_getter=lambda: {},
            _hf_weight_iterator=SimpleNamespace(iter_hf_weights=iterate, weight_update_selector="all"),
            protocol=SimpleNamespace(
                is_sender=rank == 0,
                rollout_engines=[],
                send_bucket=send,
                recover_connection=lambda: None,
                after_base_weights=lambda: None,
                after_engines_resumed=lambda: None,
            ),
        )
        recovery.update_weights_with_retry(updater, retries=1)
        updaters.append(updater)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run, rank) for rank in range(2)]
        for future in futures:
            future.result(timeout=10)
    assert [updater.weight_version for updater in updaters] == [5, 5]
    assert events.count("resume_engines") == 1
    assert ("send", 2, 0) in events


def test_http_reset_closes_old_client_on_owning_loop(monkeypatch):
    async def check():
        loop = asyncio.get_running_loop()
        old = SimpleNamespace(aclose=AsyncMock())
        monkeypatch.setattr(recovery.GeneralHttpClientProvider, "_clients", {loop: old})
        await recovery.GeneralHttpClientProvider.reset()
        old.aclose.assert_awaited_once()
        assert loop not in recovery.GeneralHttpClientProvider._clients

    asyncio.run(check())


def test_gpu_failure_is_not_classified_as_lost_response(monkeypatch):
    reply = Mock()
    reply.result.side_effect = httpx.ReadError("lost")
    monkeypatch.setattr(broadcast, "update_weights_from_distributed", lambda *a, **k: [reply])
    stream = SimpleNamespace(synchronize=Mock(side_effect=RuntimeError("CUDA failed")))
    monkeypatch.setattr(broadcast.torch.cuda, "current_stream", lambda: stream)
    protocol = SimpleNamespace(
        _engine_lock=nullcontext(),
        group_name="test",
        _model_update_groups=None,
        rollout_engines=[],
        _selector="all",
        args=Namespace(update_weight_sync_retries=1),
    )
    with pytest.raises(RuntimeError, match="CUDA failed"):
        broadcast.UpdateWeightFromDistributed.send_bucket(protocol, [])
    reply.result.assert_not_called()


def test_updater_routes_enabled_launch_option(monkeypatch):
    updater = WeightUpdater.__new__(WeightUpdater)
    updater.args = Namespace(update_weight_sync_retries=2)
    retry = Mock()
    monkeypatch.setattr(recovery, "update_weights_with_retry", retry)
    updater.update_weights()
    retry.assert_called_once_with(updater, retries=2)
