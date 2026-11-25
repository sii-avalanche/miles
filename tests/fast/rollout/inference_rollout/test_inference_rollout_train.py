import asyncio
import logging
from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.rollout.errors import RolloutSampleError
from miles.rollout.inference_rollout import inference_rollout_train


class TestAbort:
    @pytest.mark.parametrize("partial", [True, False])
    async def test_one_unresponsive_worker_does_not_stop_abort_cleanup(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, partial: bool
    ) -> None:
        """One failed worker abort still runs the agent hook and returns pending partial groups."""
        args = Namespace(
            partial_rollout=partial,
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
        pendings = {pending}
        if not partial:
            exhausted = asyncio.get_running_loop().create_future()
            exhausted.set_exception(RolloutSampleError("unused terminal trial"))
            pendings.add(exhausted)
        await asyncio.sleep(0)
        monkeypatch.setattr(inference_rollout_train, "get", fake_get)
        monkeypatch.setattr(inference_rollout_train, "post", fake_post)
        monkeypatch.setattr(inference_rollout_train, "call_agent_abort_hook", fake_agent_abort_hook)

        with caplog.at_level(logging.WARNING, logger=inference_rollout_train.__name__):
            aborted_groups = await inference_rollout_train.abort(state, pendings, rollout_id=17)

        assert posted_urls == ["http://healthy/abort_request", "http://unresponsive/abort_request"]
        assert hook_calls == [args]
        assert aborted_groups == ([[sample]] if partial else [])
        assert sample.metadata == ({"start_rollout_id": 17} if partial else {})
        assert "Failed to abort worker at http://unresponsive: worker cannot answer" in caplog.text
