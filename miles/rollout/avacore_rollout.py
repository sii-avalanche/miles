import asyncio
import copy
import json
import logging
import os
from argparse import Namespace
from collections.abc import Mapping
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import numpy as np
import pybase64
from ava_core.config import interpolated, launch_converter
from ava_core.core import Schema, Trace
from ava_core.generate.core import GenerateFunction
from ava_core.generate.core import Sample as Row
from ava_core.rewards import RewardFunction
from ava_core.rewards.core import Reward
from ava_core.runner import RECOVERABLE_ERRORS
from ava_core.store.core import STORED, Run
from ava_core.store.postgres import PostgresBackend

from miles.rollout.avacore_trace import conversation_tips, graph_nodes, snapshot_traces, write_record
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.rollout_recording import DurableRolloutOutbox, _atomic_json
from miles.rollout.screening_resume import inherit_records
from miles.utils.perf_monitor import get_monitor
from miles.utils.types import Sample, WeightVersionsPerCall

__all__ = ["AvaCoreRollout", "generate"]

logger = logging.getLogger(__name__)


class AvaCoreRollout:
    def __init__(self, args: Namespace, config: Mapping[str, Any]) -> None:
        os.environ["SGLANG_ROUTER_URL"] = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        os.environ["HF_CHECKPOINT"] = args.hf_checkpoint
        self.args = args
        self.config = config
        conv = launch_converter()
        self._remote_generate = None
        generate_config = interpolated(self.config["generate"])
        if "remote_rollout" in config:
            # AvaTrain owns this optional service adapter. Router args stay GPU-local:
            # the dashboard and inference controller must keep their real SGLang endpoints.
            from avatrain.remote_rollout import RemoteGenerate, RemoteSettings

            run_id = os.environ.get("RUN_DIR")
            if not run_id:
                raise ValueError("Remote rollout requires RUN_DIR to share the run's sandbox concurrency gate")
            self._remote_generate = RemoteGenerate(
                generate_config, RemoteSettings(**interpolated(config["remote_rollout"])), run_id=run_id
            )
            self.generate_fn = self._remote_generate
        else:
            self.generate_fn = conv.structure(generate_config, GenerateFunction)
        self.reward_fn = conv.structure(interpolated(self.config["reward"]), RewardFunction)
        self.recording = interpolated(self.config["record"]) if "record" in self.config else None
        self.run: asyncio.Task[Run] | None = None
        self.stack = AsyncExitStack()
        self.failed_writes = 0
        self._monitor = get_monitor(args)
        self._record_pool = None
        self._outbox: DurableRolloutOutbox | None = None
        self._recording_ready: asyncio.Task | None = None
        if self.recording is not None:
            self._init_recording()
        self._monitor.register("record", self._record_perf_snapshot)

    def _init_recording(self) -> None:
        recording = self.recording
        self._pool_min = int(recording.get("pool_min_size", 8))
        self._pool_max = int(recording.get("pool_max_size", 64))
        workers = int(recording.get("write_workers", 16))
        if not 1 <= self._pool_min <= self._pool_max or not 1 <= workers <= self._pool_max:
            raise ValueError("Recording requires 1 <= pool_min_size <= pool_max_size and workers <= pool_max_size")
        self._pool_timeout = float(recording.get("pool_timeout", 120))
        self._shutdown_timeout = float(recording.get("shutdown_timeout", 300))
        if self._pool_timeout <= 0 or self._shutdown_timeout <= 0:
            raise ValueError("Recording pool and shutdown timeouts must be positive")
        default = Path(os.environ.get("RUN_DIR", ".")) / "rollout_records"
        self._outbox = DurableRolloutOutbox(
            Path(recording.get("spool_dir", default)), self._write_record, workers=workers, on_event=self._record_event
        )
        self._legacy_seeded = (self._outbox.path / "legacy_seeded.json").exists()

    async def _start_recording(self, sampling_params: dict[str, Any]) -> None:
        manifest = {
            "model": self.recording["model"],
            "run": self.recording["run"],
            "collection": Path(self.args.prompt_data).stem,
            "sampling_params": sampling_params,
        }
        path = self._outbox.path / "run.json"
        if path.exists():
            existing = json.loads(path.read_text())
            if any(existing[key] != manifest[key] for key in ("model", "run", "collection")):
                raise ValueError("Recording directory belongs to a different model, run or collection")
        else:
            await asyncio.to_thread(_atomic_json, path, manifest)
        self._sampling_params = sampling_params.copy()
        # Legacy resumed runs have no local allocation counters. Seed before generation
        # rather than assigning IDs that could overwrite their existing database rows.
        if self.recording.get("resume", True) and not self._legacy_seeded:
            delay = 1
            while True:
                try:
                    run = await self._get_run()
                    await self._outbox.seed_trials(list(run.statuses), self.args.n_samples_per_prompt)
                    await asyncio.to_thread(_atomic_json, self._outbox.path / "legacy_seeded.json", {"complete": True})
                    break
                except Exception as error:
                    self._record_event("failed", error)
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30)
        await self._outbox.start()

    async def _prepare_recording(self, sampling_params: dict[str, Any]) -> None:
        if self._recording_ready is None:
            self._recording_ready = asyncio.create_task(self._start_recording(sampling_params))
        await asyncio.shield(self._recording_ready)

    async def inherit_completed(self, groups, manifests, sampling_params) -> int:
        """Copy confirmed old responses to this run without regenerating or rejudging."""
        if self.recording is None:
            raise ValueError("Inherited rollouts require recording")
        if any(manifest["run"] == self.recording["run"] for manifest in manifests.values()):
            raise ValueError("Screening inheritance requires a new run")
        await self._prepare_recording(sampling_params)
        count = await inherit_records(self._outbox, groups, manifests)
        if not await self._outbox.flush(self._shutdown_timeout):
            raise RuntimeError("Inherited rollout writes are still pending; replay the new run's rollout_records")
        return count

    async def generate(
        self, sample: Sample, sampling_params: dict[str, Any], *, evaluation: bool = False
    ) -> Sample | list[Sample]:
        trial_id = None
        if self.recording is not None and not evaluation:
            await self._prepare_recording(sampling_params)
            trial_id = await self._outbox.allocate_trial(sample, self.args.n_samples_per_prompt)
        row = Row({**sample.metadata, "prompt": sample.prompt, "label": sample.label}, key=lambda _: sample.index)
        replay = {"return_routed_experts": True} if self.args.use_rollout_routing_replay else {}
        try:
            with self._monitor.wait("sample/generate"):
                traces = await self.generate_fn(row, sampling_params=sampling_params, **replay)
            if any(
                message.metadata["finish_reason"]["type"] == "abort"
                for node in graph_nodes(traces)
                for message in (node.message,)
                if "finish_reason" in message.metadata
            ):
                sample.status = Sample.Status.ABORTED
                return sample
            with self._monitor.wait("sample/reward"):
                reward = await self.reward_fn(traces[0], row)
        except RECOVERABLE_ERRORS as error:
            self._monitor.increment("sample/recoverable_errors_total")
            logger.warning("AvaCore rollout of sample %s aborted: %r", sample.index, error)
            sample.status = Sample.Status.ABORTED
            return sample

        assert traces and any(node.segments for node in graph_nodes(traces)), (
            "AvaCore rollouts must drive the policy through a token-level client"
        )
        if self.recording is not None and not evaluation:
            # Only disk durability is awaited here. Database writes run independently
            # with bounded concurrency; a retry cannot mutate an accepted snapshot.
            persist = asyncio.create_task(self.record(row, sample, traces, reward, sampling_params, trial_id=trial_id))
            try:
                await asyncio.shield(persist)
            except asyncio.CancelledError:
                await persist
                raise
        samples, owned = [], set()
        for tip in conversation_tips(traces):
            generated = {node for node in tip.prefixes if node.generated and node.segments}
            if generated - owned:
                samples.append(filled(self.args, copy.deepcopy(sample), tip, reward.score, owned=owned))
                owned.update(generated)
        return samples[0] if len(samples) == 1 else samples

    def _record_perf_snapshot(self) -> dict[str, float]:
        metrics = self._outbox.stats() if self._outbox is not None else {"pending_writes": 0}
        if self._record_pool is not None:
            stats = self._record_pool.get_stats()  # get_stats never resets the real pool's counters
            metrics.update({f"pool/{key}": value for key, value in stats.items()})
            for key in ("connections_errors", "connections_lost", "requests_errors"):
                # Mark cumulative fields explicitly so the monitor can derive interval rates.
                metrics[f"pool/{key}_total"] = stats.get(key, 0)
        return metrics

    async def open(self, sampling_params: dict[str, Any]) -> Run:
        assert self.recording is not None
        store = await self.stack.enter_async_context(
            PostgresBackend(
                self.recording["postgres"],
                min_size=self._pool_min,
                max_size=self._pool_max,
            )
        )
        store.pool.timeout = self._pool_timeout
        self._record_pool = store.pool
        run = await self.stack.enter_async_context(
            store.rl_run(
                model=self.recording["model"],
                run=self.recording["run"],
                collection=Path(self.args.prompt_data).stem,
                sampling_params=sampling_params,
                config=self.config,
                schema=Schema(),
                resume=True,
            )
        )
        await run.update(status="running")
        return run

    async def _get_run(self) -> Run:
        if self.run is None or (self.run.done() and not self.run.cancelled() and self.run.exception() is not None):
            self.run = asyncio.create_task(self._open_recording_run())
        return await asyncio.shield(self.run)

    async def _open_recording_run(self) -> Run:
        try:
            return await self.open(self._sampling_params)
        except BaseException:
            await self.stack.aclose()
            self.stack = AsyncExitStack()
            self._record_pool = None
            raise

    async def _write_record(self, record: dict[str, Any]) -> None:
        with self._monitor.wait("record/open_wait"):
            run = await self._get_run()
        with self._monitor.wait("record/create_rollout"):
            await write_record(run, record)

    def _record_event(self, event: str, error: Exception | None) -> None:
        self._monitor.increment(f"record/{event}_total")
        if event != "failed":
            return
        self.failed_writes += 1
        self._monitor.increment(
            "record/pool_timeout_total" if type(error).__name__ == "PoolTimeout" else "record/other_errors_total"
        )
        if self.failed_writes in (1, 10, 100) or self.failed_writes % 1000 == 0:
            logger.warning(
                "Postgres write failed %d times; durable records retained for retry: %r", self.failed_writes, error
            )

    async def record(
        self,
        row: Row,
        sample: Sample,
        traces: tuple[Trace, ...],
        reward: Reward,
        sampling_params: dict[str, Any],
        *,
        trial_id: int | None = None,
    ) -> None:
        with self._monitor.wait("record/end_to_end"):
            await self._prepare_recording(sampling_params)
            if trial_id is None:
                trial_id = await self._outbox.allocate_trial(sample, self.args.n_samples_per_prompt)
            # Raw-HF inference can report a name such as "default". Keep the
            # source version; staleness calculations separately select numeric versions.
            metadata = {
                "weight_version": traces[0].last_assistant().metadata.get("weight_version"),
                "epoch": sample.epoch,
                "group_index": sample.group_index,
                "sample_index": sample.index,
                "generation_attempt": getattr(
                    sample, "generation_attempt", trial_id // self.args.n_samples_per_prompt
                ),
                "generation_round": trial_id // self.args.n_samples_per_prompt,
            }

            def snapshot():
                return {
                    "query_id": str(sample.metadata["_index"]),
                    "trial_id": trial_id,
                    "instance": STORED.unstructure(row),
                    "trace_graph": snapshot_traces(traces),
                    "reward": STORED.unstructure(reward),
                    "status": "completed",
                    "metadata": metadata,
                }

            with self._monitor.wait("record/persist"):
                await self._outbox.enqueue(await asyncio.to_thread(snapshot))

    async def close(self) -> None:
        if self._remote_generate is not None:
            await self._remote_generate.aclose()
        if self._outbox is None:
            return
        if self._recording_ready is not None and not self._recording_ready.done():
            self._recording_ready.cancel()
            await asyncio.gather(self._recording_ready, return_exceptions=True)
        drained = await self._outbox.close(self._shutdown_timeout)
        if self.run is not None and not self.run.done():
            self.run.cancel()
            await asyncio.gather(self.run, return_exceptions=True)
        await self.stack.aclose()
        if not drained:
            raise RuntimeError(
                f"Rollout recordings are still pending in {self._outbox.pending}; replay them before declaring recording complete"
            )

    async def dispose(self) -> None:
        loop = (
            self._recording_ready.get_loop()
            if self._recording_ready is not None
            else getattr(self._remote_generate, "loop", None) or asyncio.get_running_loop()
        )
        if loop is asyncio.get_running_loop():
            await self.close()
        else:
            await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(self.close(), loop))


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    state = input.state
    if (rollout := getattr(state, "avacore", None)) is None:
        rollout = state.avacore = AvaCoreRollout(input.args, input.args.avacore_config)
    return GenerateFnOutput(
        samples=await rollout.generate(input.sample, input.sampling_params, evaluation=input.evaluation)
    )


def filled(args: Namespace, sample: Sample, trace: Trace, reward: float, *, owned: set[Trace] | None = None) -> Sample:
    # Shared generated prefixes train only in the first listed branch, as in the result store.
    owned = owned or set()
    segments = trace.path_segments
    segments = segments[: max(i for i, segment in enumerate(segments) if segment.is_generated) + 1]
    length = sum(len(segment.tokens) for segment in segments)
    prompt_length = len(segments[0].tokens)
    sample.tokens = trace.tokens[:length]
    sample.response = "".join(segment.template for segment in segments[1:])
    sample.response_length = length - prompt_length
    mask = [
        int(segment.is_generated and node not in owned)
        for node in trace.prefixes
        for segment in node.segments
        for _ in segment.tokens
    ]
    sample.loss_mask = mask[prompt_length:length]
    sample.rollout_log_probs = [log_prob or 0.0 for log_prob in trace.log_probs[prompt_length:length]]
    spans, offset = [], 0
    for segment in segments:
        if segment.is_generated:
            spans.append((offset, offset + len(segment.tokens)))
        offset += len(segment.tokens)
    finished = [node.message for node in trace.prefixes if node.generated and node.segments]
    for message, (start, end) in zip(finished, spans, strict=True):
        # The trace keeps log probs per token, not per call: anchor the call's weight versions to its own segment.
        versioning = ("weight_version", "weight_versions")
        sample.update_from_meta_info(args, {k: v for k, v in message.metadata.items() if k not in versioning})
        sample.weight_versions[-1] = WeightVersionsPerCall.from_meta_info(
            message.metadata | {"output_token_logprobs": [None] * (end - start)}, output_end=end
        )
    if args.use_rollout_routing_replay and sample.status != Sample.Status.ABORTED:
        chunks = [segment.metadata["routed_experts"] for segment in segments if "routed_experts" in segment.metadata]
        assert chunks, (
            "Internal error: routing replay enabled but no routed experts returned. Perhaps some traces are not generated by the configured generate function."
        )
        routed = np.frombuffer(b"".join(pybase64.b64decode(chunk) for chunk in chunks), dtype=np.int32)
        row = args.num_layers * args.moe_router_topk
        if routed.size == length * row:
            # sglang also forwarded the final token, whose routing feeds no training position
            routed = routed[: (length - 1) * row]
        sample.rollout_routed_experts = routed.reshape(length - 1, args.num_layers, args.moe_router_topk)
    sample.reward = reward
    return sample
