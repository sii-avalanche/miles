"""Exercise real group cancellation through LMaaJ and the native HTTP transport."""

import asyncio
import json
import threading
from argparse import Namespace
from types import SimpleNamespace

import pytest
from ava_core import TokioPost
from ava_core.core import ChatMessage, ChatTrace, HttpEndpoint, OpenAIClient, TokenTrace
from ava_core.core.retry import HTTPRetry
from ava_core.rewards.core import Reward
from ava_core.rewards.lmaaj import LMaaJ

from miles.rollout import avacore_rollout, rollout_recording
from miles.rollout.inference_rollout import inference_rollout_common as common
from miles.rollout.inference_rollout import inference_rollout_train as train
from miles.utils.perf_monitor import RuntimeMonitor
from miles.utils.types import Sample


class JudgeServer:
    def __init__(self, *, fail_first=False):
        self.fail_first = fail_first
        self.received = asyncio.Event()
        self.disconnected = asyncio.Event()
        self.calls = 0
        self.handlers = set()

    async def __aenter__(self):
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        self.url = f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"
        return self

    async def __aexit__(self, *_args):
        self.server.close()
        await self.server.wait_closed()
        for task in self.handlers:
            task.cancel()
        await asyncio.gather(*self.handlers, return_exceptions=True)

    async def handle(self, reader, writer):
        task = asyncio.current_task()
        self.handlers.add(task)
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            length = int(
                next(
                    line.split(b":", 1)[1]
                    for line in headers.split(b"\r\n")
                    if line.lower().startswith(b"content-length:")
                )
            )
            await reader.readexactly(length)
            self.calls += 1
            self.received.set()
            if self.calls == 1 and not self.fail_first:
                try:
                    assert await reader.read(1) == b""
                except ConnectionResetError:
                    pass
                self.disconnected.set()
            else:
                status = "503 Service Unavailable" if self.calls == 1 else "200 OK"
                body = json.dumps(
                    {"choices": [{"message": {"role": "assistant", "content": "A"}, "finish_reason": "stop"}]}
                ).encode()
                writer.write(
                    f"HTTP/1.1 {status}\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body
                )
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            self.handlers.discard(task)


def make_state(adapter):
    args = Namespace(
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        group_rm=True,
        sglang_router_policy="round_robin",
        use_rollout_routing_replay=False,
        n_samples_per_prompt=2,
    )
    adapter.args = args
    adapter._monitor = RuntimeMonitor(args=Namespace(perf_monitor_interval=10), role="rollout", sink=lambda _: None)
    return SimpleNamespace(
        args=args,
        aborted=False,
        generate_fn_semaphore=asyncio.Semaphore(2),
        generate_function=avacore_rollout.generate,
        avacore=adapter,
    )


@pytest.fixture
def abort_without_workers(monkeypatch):
    async def get_urls(_args):
        return []

    async def noop_hook(_args):
        pass

    monkeypatch.setattr(train, "get_worker_urls", get_urls)
    monkeypatch.setattr(train, "call_agent_abort_hook", noop_hook)
    # Native reqwest reads the proxy environment at construction time.
    for key in ("http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv(key.upper(), raising=False)


async def make_adapter(client):
    async def generate(_row, **_kwargs):
        return ChatTrace.from_messages([ChatMessage(role="assistant", content="candidate")])

    adapter = object.__new__(avacore_rollout.AvaCoreRollout)
    adapter.generate_fn = generate
    adapter.reward_fn = LMaaJ(model=client, prompt=lambda _trace, _row: "judge fixture", system_prompt=None)
    adapter.recording = None
    return adapter


async def test_abort_closes_active_judge_cancels_queue_and_releases_permit(abort_without_workers):
    async with JudgeServer() as server:
        transport = TokioPost(server.url, concurrency=1, worker_threads=2, timeout=600)
        client = OpenAIClient(
            HttpEndpoint.from_url(server.url), "fixture", post=transport.post, retry=HTTPRetry(max_retries=3)
        )
        adapter = await make_adapter(client)
        state = make_state(adapter)
        samples = [Sample(index=i, group_index=0, prompt="question") for i in range(2)]
        completed = []
        group = asyncio.create_task(
            common.generate_and_rm_group(state, samples, {}, sample_done_callback=lambda: completed.append(True))
        )
        try:
            await asyncio.wait_for(server.received.wait(), timeout=2)
            assert adapter._monitor.snapshot()["runtime/rollout/sample/reward/waiting"] == 2
            assert await asyncio.wait_for(train.abort(state, {group}, rollout_id=0), timeout=2) == []
            await asyncio.wait_for(server.disconnected.wait(), timeout=2)
            assert server.calls == 1  # the queued judge was never issued
            assert group.cancelled()
            assert len(completed) == 2
            assert all(s.status == Sample.Status.ABORTED for s in samples)
            assert adapter._monitor.snapshot()["runtime/rollout/sample/reward/waiting"] == 0
            reply = await asyncio.wait_for(client.chat([ChatMessage(role="user", content="next batch")]), timeout=2)
            assert reply.text_content == "A"  # cancelled requests returned their native semaphore permits
            assert server.calls == 2
        finally:
            group.cancel()
            await asyncio.gather(group, return_exceptions=True)
            transport.close()


async def test_abort_stops_judge_retry_backoff(abort_without_workers):
    async with JudgeServer(fail_first=True) as server:
        transport = TokioPost(server.url, concurrency=1, worker_threads=2, timeout=600)
        retrying = asyncio.Event()
        retry = HTTPRetry(max_retries=3)

        def backoff(_attempts):
            retrying.set()
            return 60

        retry.backoff = backoff
        client = OpenAIClient(HttpEndpoint.from_url(server.url), "fixture", post=transport.post, retry=retry)
        state = make_state(await make_adapter(client))
        group = asyncio.create_task(common.generate_and_rm_group(state, [Sample(index=0)], {}))
        try:
            await asyncio.wait_for(retrying.wait(), timeout=2)
            assert await asyncio.wait_for(train.abort(state, {group}, rollout_id=0), timeout=2) == []
            assert group.cancelled()
            assert server.calls == 1
        finally:
            group.cancel()
            await asyncio.gather(group, return_exceptions=True)
            transport.close()


async def test_abort_finishes_completed_reward_snapshot(abort_without_workers, monkeypatch, tmp_path):
    adapter = object.__new__(avacore_rollout.AvaCoreRollout)
    state = make_state(adapter)
    trace = TokenTrace(
        segments=(),
        parser=None,
        _messages=(ChatMessage(role="assistant", content="candidate", metadata={"weight_version": 1}),),
    )

    async def generate(_row, **_kwargs):
        return trace

    async def reward(_trace, _row):
        return Reward(score=1)

    async def prepared(_params):
        pass

    saved = []

    async def write(value):
        saved.append(value)

    adapter.generate_fn, adapter.reward_fn = generate, reward
    adapter.recording = {"model": "test", "run": "test", "resume": False, "spool_dir": tmp_path}
    adapter._write_record = write
    adapter._prepare_recording = prepared
    adapter._init_recording()
    await adapter._outbox.start()
    monkeypatch.setattr(
        avacore_rollout,
        "STORED",
        SimpleNamespace(
            unstructure=lambda value: (
                {"messages": []}
                if value is trace
                else {"score": value.score} if isinstance(value, Reward) else dict(value)
            )
        ),
    )
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    atomic_json = rollout_recording._atomic_json

    def blocked_capture(path, value, **kwargs):
        if path.parent == adapter._outbox.pending:
            loop.call_soon_threadsafe(started.set)
            assert release.wait(timeout=5)
        atomic_json(path, value, **kwargs)

    monkeypatch.setattr(rollout_recording, "_atomic_json", blocked_capture)
    sample = Sample(index=0, epoch=0, group_index=0, metadata={"_index": 7})
    group = asyncio.create_task(common.generate_and_rm_group(state, [sample], {}))
    abort = None
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        abort = asyncio.create_task(train.abort(state, {group}, rollout_id=0))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not abort.done()  # cancellation joins the shielded disk capture
        release.set()
        assert await asyncio.wait_for(abort, timeout=2) == []
        assert await adapter._outbox.flush(timeout=2)
        assert len(saved) == 1
        assert saved[0]["reward"]["score"] == 1
        assert saved[0]["trial_id"] == 0
        assert len(list(adapter._outbox.committed.glob("*.json.gz"))) == 1
        assert not list(adapter._outbox.pending.glob("*.json.gz"))
    finally:
        release.set()
        await asyncio.gather(group, *([abort] if abort else []), return_exceptions=True)
        await adapter._outbox.close(timeout=2)
