import asyncio
import logging
from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.rollout.filter_hub.base_types import FilterOutput
from miles.rollout.inference_rollout import batch_pool, inference_rollout_train
from miles.utils.perf_monitor import RuntimeMonitor
from miles.utils.types import Sample


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


@pytest.mark.parametrize("partial_rollout", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
async def test_batch_pool_tracks_only_selected_groups_and_abort_request(monkeypatch, partial_rollout, enabled):
    args = Namespace(
        rollout_global_dataset=True,
        rollout_batch_size=2,
        over_sampling_batch_size=4,
        n_samples_per_prompt=2,
        partial_rollout=partial_rollout,
        rollout_submission_granularity=None,
        dynamic_sampling_filter_path=None,
        rollout_sample_filter_path=None,
        rollout_all_samples_process_path=None,
        reward_key=None,
        sglang_router_ip="router",
        sglang_router_port=30000,
        perf_monitor_interval=10 if enabled else 0,
    )
    monitor = RuntimeMonitor(args=args, role="rollout", sink=lambda _: None)
    monkeypatch.setattr(batch_pool, "get_monitor", lambda _: monitor)
    monkeypatch.setattr(inference_rollout_train, "get_monitor", lambda _: monitor)
    state = SimpleNamespace(args=args, aborted=False, sampling_params={}, reset=lambda: None)
    hook_called = asyncio.Event()
    groups = [
        [Sample(index=2 * i + j, response="answer", reward=float(j), status=Sample.Status.COMPLETED) for j in range(2)]
        for i in range(4)
    ]
    source_calls = []

    def source(size):
        source_calls.append(size)
        return groups

    async def generate(_, group, **kwargs):
        if group is groups[3]:
            await hook_called.wait()
        return group

    async def configure(_):
        pass

    async def recompute(*args, **kwargs):
        pass

    async def worker_urls(_):
        return ["http://worker"]

    async def post(url, payload):
        assert url == "http://worker/abort_request" and payload == {"abort_all": True}
        if enabled:
            requests = [e for e in monitor._events if e["name"] == "batch_abort"]
            assert len(requests) == 1, "the event must precede the abort HTTP request"
            assert requests[0]["details"] == {
                "groups": 1,
                "samples_upper_bound": 2,
                "action": "retain_partial" if partial_rollout else "discard",
            }
            assert monitor.snapshot()["runtime/rollout/batch/collected_groups"] == 2

    async def abort_hook(_):
        hook_called.set()

    monkeypatch.setattr(inference_rollout_train, "generate_and_rm_group", generate)
    monkeypatch.setattr(inference_rollout_train.dumper_utils, "configure_sglang", configure)
    monkeypatch.setattr(inference_rollout_train, "recompute_samples_rollout_logprobs_via_prefill", recompute)
    monkeypatch.setattr(inference_rollout_train, "get_worker_urls", worker_urls)
    monkeypatch.setattr(inference_rollout_train, "post", post)
    monkeypatch.setattr(inference_rollout_train, "call_agent_abort_hook", abort_hook)
    monkeypatch.setattr(
        inference_rollout_train,
        "apply_preput_filters",
        lambda args, fn, group: FilterOutput(keep=group is not groups[0], reason="zero_std_0.0"),
    )

    output, aborted = await inference_rollout_train.generate_rollout_async(state, 17, source)
    assert output.samples == groups[1:3]
    assert source_calls == [4]
    assert aborted == ([groups[3]] if partial_rollout else [])
    assert output.metrics["rollout/dynamic_filter_drop_zero_std_0.0"] == 1
    events = [e for e in monitor._events if e["name"] == "batch_pool"]
    if enabled:
        assert [e["details"]["collected_groups"] for e in events] == [0, 1, 2, 0]
        assert [e["details"]["status"] for e in events] == ["collecting", "collecting", "ready", "released"]
        assert all(e["rollout_id"] == 17 and e["details"]["target_groups"] == 2 for e in events)
        values = monitor.snapshot()
        assert values["runtime/rollout/batch/collected_groups"] == 0
        assert values["runtime/rollout/batch/active"] == 0
        assert values["runtime/rollout/batch/abort_requested_groups_total"] == 1
        assert values["runtime/rollout/batch/groups_completed_total"] == 3
        assert values["runtime/rollout/batch/groups_selected_total"] == 2
        assert values["runtime/rollout/batch/groups_rejected_total"] == 1
        rejected = [e for e in monitor._events if e["name"] == "batch_reject"]
        assert len(rejected) == 1
        assert rejected[0]["rollout_id"] == 17
        assert rejected[0]["details"] == {"groups": 1, "reason": "zero_std_0.0", "action": "drop"}
    else:
        assert not monitor._events


@pytest.mark.parametrize("error_type,status", [(ValueError, "failed"), (asyncio.CancelledError, "cancelled")])
async def test_batch_pool_resets_after_failure_or_cancellation(monkeypatch, error_type, status):
    args = Namespace(rollout_batch_size=2, perf_monitor_interval=10)
    monitor = RuntimeMonitor(args=args, role="rollout", sink=lambda _: None)
    monkeypatch.setattr(batch_pool, "get_monitor", lambda _: monitor)

    async def fail(state, rollout_id, source, pool):
        pool.accept_group()
        raise error_type()

    monkeypatch.setattr(inference_rollout_train, "_generate_rollout_async", fail)
    with pytest.raises(error_type):
        await inference_rollout_train.generate_rollout_async(SimpleNamespace(args=args), 3, lambda _: [])
    assert monitor.snapshot()["runtime/rollout/batch/collected_groups"] == 0
    assert monitor.snapshot()["runtime/rollout/batch/active"] == 0
    assert monitor._events[-1]["details"]["status"] == status
