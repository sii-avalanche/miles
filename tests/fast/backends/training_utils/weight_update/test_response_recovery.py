"""Recovery boundaries: lost replies may replay; unfinished CUDA work must not."""

from argparse import Namespace
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import torch

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.weight_update import updater as updater_module
from miles.backends.training_utils.weight_update.protocols import broadcast
from miles.backends.training_utils.weight_update.updater import WeightUpdater


def future(result=None, error=None):
    value = Future()
    if error is not None:
        value.set_exception(error)
    else:
        value.set_result(result)
    return value


@pytest.fixture
def gpu_event(monkeypatch):
    event = MagicMock()
    event.query.return_value = True
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    return event


@pytest.mark.parametrize("error_type", [httpx.ReadError, httpx.ReadTimeout, httpx.RemoteProtocolError])
def test_only_completed_transfers_turn_lost_responses_into_retry(error_type, gpu_event):
    with pytest.raises(broadcast.WeightSyncResponseError):
        broadcast._wait_for_bucket([future(error=error_type("lost")), future({"success": True})], timeout=1)
    gpu_event.record.assert_called_once()


def test_lost_response_with_unfinished_cuda_is_fatal(gpu_event):
    gpu_event.query.return_value = False
    with pytest.raises(RuntimeError, match="did not settle") as caught:
        broadcast._wait_for_bucket([future(error=httpx.ReadError("lost"))], timeout=0)
    assert not isinstance(caught.value, broadcast.WeightSyncResponseError)


def test_pending_http_request_is_not_retried_or_cancelled(gpu_event):
    pending = Future()
    with pytest.raises(RuntimeError, match="pending HTTP receivers=\\[1\\]"):
        broadcast._wait_for_bucket([future(error=httpx.ReadError("lost")), pending], timeout=0)
    assert not pending.done()


@pytest.mark.parametrize("result", [None, {}, {"success": False}])
def test_receiver_rejection_wins_over_lost_response(gpu_event, result):
    with pytest.raises(RuntimeError, match="Cannot confirm") as caught:
        broadcast._wait_for_bucket([future(error=httpx.ReadError("lost")), future(result)], timeout=1)
    assert not isinstance(caught.value, broadcast.WeightSyncResponseError)


def test_completed_transfer_accepts_all_receiver_acknowledgements(gpu_event):
    broadcast._wait_for_bucket([future({"success": True}), future({"success": True})], timeout=1)


@pytest.fixture
def recovery(monkeypatch):
    calls = []
    visited = []
    sent = []

    class Engine:
        async def pause_generation(self, **kwargs):
            calls.append("pause")

        async def begin_weight_update(self, **kwargs):
            calls.append("begin")
            return {"success": True}

        async def abort_weight_update(self):
            calls.append("abort")
            return {"success": True}

        async def end_weight_update(self, **kwargs):
            calls.append("end")
            return {"success": True}

        async def update_weight_version(self, weight_version):
            calls.append(("version", weight_version))

        async def get_weight_version(self):
            calls.append("verify")
            return "8"

        async def continue_generation(self):
            calls.append("resume")

    engine = Engine()

    def buckets(*args, **kwargs):
        for index in range(3):
            visited.append(index)
            yield [(str(index), torch.zeros(1))]

    def send(bucket):
        sent.append(int(bucket[0][0]))

    instance = WeightUpdater.__new__(WeightUpdater)
    instance.args = Namespace(
        colocate=False,
        update_weight_transfer_mode="broadcast",
        update_weight_sync_retries=1,
        pause_generation_mode="in_place",
    )
    instance.protocol = SimpleNamespace(
        is_sender=True,
        rollout_engines=[engine],
        send_bucket=send,
        after_base_weights=lambda: None,
        after_engines_resumed=lambda: calls.append("done"),
    )
    instance.parallel_state = SimpleNamespace(pp=SimpleNamespace(size=1))
    instance._hf_weight_iterator = SimpleNamespace(iter_hf_weights=buckets, weight_update_selector="all")
    instance.weights_getter = lambda: {}
    instance.is_lora = False
    instance.weight_version = 7
    monkeypatch.setattr(updater_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(updater_module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(updater_module.dist, "all_reduce", lambda *a, **k: SimpleNamespace(wait=lambda **k: True))
    monkeypatch.setattr(updater_module, "get_gloo_group", lambda: None)
    reset = AsyncMock(side_effect=lambda: calls.append("reset"))
    monkeypatch.setattr(updater_module.GeneralHttpClientProvider, "reset", reset)
    return instance, engine, calls, visited, sent


def test_replay_restarts_from_first_bucket_without_rebuilding_nccl(recovery):
    instance, engine, calls, visited, sent = recovery

    def send(bucket):
        sent.append(int(bucket[0][0]))
        if len(sent) == 2:
            raise broadcast.WeightSyncResponseError("lost")

    instance.protocol.send_bucket = send
    instance.update_weights()
    assert visited == [0, 1, 2, 0, 1, 2]
    assert sent == [0, 1, 0, 1, 2]
    assert calls == ["pause", "begin", "reset", "abort", "begin", "end", ("version", "8"), "verify", "resume", "done"]
    assert instance.weight_version == 8


def test_unsafe_transfer_does_not_drain_gathers_or_publish_version(recovery):
    instance, engine, calls, visited, sent = recovery
    instance.protocol.send_bucket = MagicMock(side_effect=RuntimeError("CUDA unfinished"))
    with pytest.raises(RuntimeError, match="CUDA unfinished"):
        instance.update_weights()
    assert visited == [0]
    assert calls == ["pause", "begin"]
    assert instance.weight_version == 7


def test_abort_failure_never_replays_or_resumes(recovery):
    instance, engine, calls, visited, sent = recovery
    instance.protocol.send_bucket = MagicMock(side_effect=broadcast.WeightSyncResponseError("lost"))
    engine.abort_weight_update = AsyncMock(return_value={"success": False})
    with pytest.raises(RuntimeError, match="recovery is unsafe"):
        instance.update_weights()
    assert calls == ["pause", "begin", "reset"]
    assert instance.weight_version == 7


def test_retry_budget_exhaustion_never_commits(recovery):
    instance, engine, calls, visited, sent = recovery
    instance.protocol.send_bucket = MagicMock(side_effect=broadcast.WeightSyncResponseError("lost"))
    with pytest.raises(broadcast.WeightSyncResponseError):
        instance.update_weights()
    assert calls == ["pause", "begin", "reset", "abort", "begin"]
    assert instance.weight_version == 7


@pytest.mark.asyncio
async def test_bucket_request_forwards_opt_in_timeout(monkeypatch):
    request = AsyncMock(return_value={"success": True})
    monkeypatch.setattr(SGLangApiClient, "_make_request", request)
    await SGLangApiClient("http://engine").update_weights_from_distributed(
        ["weight"],
        [torch.float32],
        [(1,)],
        "group",
        timeout=60.0,
    )
    assert request.call_args.kwargs == {"timeout": 60.0}
