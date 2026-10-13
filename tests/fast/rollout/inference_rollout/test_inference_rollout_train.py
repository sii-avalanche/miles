import asyncio
import logging
from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.rollout.inference_rollout import inference_rollout_train


class TestAbort:
    async def test_discarded_groups_cancel_before_slow_sglang_abort(self, monkeypatch):
        state = SimpleNamespace(args=Namespace(partial_rollout=False), aborted=False)
        cancelled = asyncio.Event()
        release_abort = asyncio.Event()

        async def judge_forever():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def get_urls(_args):
            return ["http://worker"]

        async def slow_post(*_args):
            await release_abort.wait()

        async def noop_hook(_args):
            pass

        pending = asyncio.create_task(judge_forever())
        await asyncio.sleep(0)
        monkeypatch.setattr(inference_rollout_train, "get_worker_urls", get_urls)
        monkeypatch.setattr(inference_rollout_train, "post", slow_post)
        monkeypatch.setattr(inference_rollout_train, "call_agent_abort_hook", noop_hook)
        abort = asyncio.create_task(inference_rollout_train.abort(state, {pending}, rollout_id=0))
        try:
            await asyncio.wait_for(cancelled.wait(), timeout=1)
            assert not abort.done()
            release_abort.set()
            assert await asyncio.wait_for(abort, timeout=1) == []
            assert pending.cancelled()
            assert pending.cancelling() == 1
        finally:
            release_abort.set()
            await asyncio.gather(abort, pending, return_exceptions=True)

    async def test_worker_discovery_failure_still_joins_cancelled_groups(self, monkeypatch):
        state = SimpleNamespace(args=Namespace(partial_rollout=False), aborted=False)
        finished = asyncio.Event()

        async def judge_forever():
            try:
                await asyncio.Event().wait()
            finally:
                finished.set()

        async def failed_discovery(_args):
            raise ConnectionError("router unavailable")

        pending = asyncio.create_task(judge_forever())
        await asyncio.sleep(0)
        monkeypatch.setattr(inference_rollout_train, "get_worker_urls", failed_discovery)
        with pytest.raises(ConnectionError, match="router unavailable"):
            await inference_rollout_train.abort(state, {pending}, rollout_id=0)
        assert pending.cancelled()
        assert finished.is_set()

    async def test_one_unresponsive_worker_does_not_stop_abort_cleanup(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One failed worker abort still runs the agent hook and returns pending partial groups."""
        args = Namespace(
            partial_rollout=True,
            sglang_router_ip="router",
            sglang_router_port=30000,
            use_miles_router=True,
        )
        state = SimpleNamespace(args=args, aborted=False)
        sample = SimpleNamespace(response="partial", metadata={})
        hook_calls: list[Namespace] = []
        posted_urls: list[str] = []
        hook_finished = asyncio.Event()

        async def fake_get(url: str) -> dict[str, list[str]]:
            return {"urls": ["http://healthy", "http://unresponsive"]}

        async def fake_post(url: str, payload: dict[str, bool]) -> None:
            posted_urls.append(url)
            if "unresponsive" in url:
                raise ConnectionError("worker cannot answer")

        async def fake_agent_abort_hook(hook_args: Namespace) -> None:
            hook_calls.append(hook_args)
            hook_finished.set()

        async def finish_group() -> list[SimpleNamespace]:
            await hook_finished.wait()
            return [sample]

        pending = asyncio.create_task(finish_group())
        await asyncio.sleep(0)
        monkeypatch.setattr(inference_rollout_train, "get", fake_get)
        monkeypatch.setattr(inference_rollout_train, "post", fake_post)
        monkeypatch.setattr(inference_rollout_train, "call_agent_abort_hook", fake_agent_abort_hook)

        with caplog.at_level(logging.WARNING, logger=inference_rollout_train.__name__):
            aborted_groups = await inference_rollout_train.abort(state, {pending}, rollout_id=17)

        assert posted_urls == ["http://healthy/abort_request", "http://unresponsive/abort_request"]
        assert hook_calls == [args]
        assert aborted_groups == [[sample]]
        assert sample.metadata["start_rollout_id"] == 17
        assert "Failed to abort worker at http://unresponsive: worker cannot answer" in caplog.text
