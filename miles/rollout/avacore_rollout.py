import asyncio
import copy
import logging
import os
import time
from argparse import Namespace
from collections.abc import Mapping
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import numpy as np
import pybase64
from ava_core.config import interpolated, launch_converter
from ava_core.core import Schema, TokenTrace, Trace
from ava_core.generate.core import GenerateFunction
from ava_core.generate.core import Sample as Row
from ava_core.rewards import RewardFunction
from ava_core.rewards.core import Reward
from ava_core.runner import RECOVERABLE_ERRORS
from ava_core.store.core import Run
from ava_core.store.postgres import PostgresBackend

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
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
        self.generate_fn = conv.structure(interpolated(self.config["generate"]), GenerateFunction)
        self.reward_fn = conv.structure(interpolated(self.config["reward"]), RewardFunction)
        self.recording = interpolated(self.config["record"]) if "record" in self.config else None
        self.run: asyncio.Task[Run] | None = None
        self.stack = AsyncExitStack()
        self.writes: set[asyncio.Task[None]] = set()
        self.failed_writes = 0
        self._monitor = get_monitor(args)
        self._record_pool = None
        self._writes_started: dict[asyncio.Task, float] = {}
        self._monitor.register("record", self._record_perf_snapshot)

    async def generate(self, sample: Sample, sampling_params: dict[str, Any]) -> Sample | list[Sample]:
        row = Row({**sample.metadata, "prompt": sample.prompt, "label": sample.label}, key=lambda _: sample.index)
        replay = {"return_routed_experts": True} if self.args.use_rollout_routing_replay else {}
        try:
            with self._monitor.wait("sample/generate"):
                trace = await self.generate_fn(row, sampling_params=sampling_params, **replay)
            if any(
                message.metadata["finish_reason"]["type"] == "abort"
                for node in flattened(trace)
                for message in node.messages
                if "finish_reason" in message.metadata
            ):
                sample.status = Sample.Status.ABORTED
                return sample
            with self._monitor.wait("sample/reward"):
                reward = await self.reward_fn(trace, row)
        except RECOVERABLE_ERRORS as error:
            self._monitor.increment("sample/recoverable_errors_total")
            logger.warning("AvaCore rollout of sample %s aborted: %r", sample.index, error)
            sample.status = Sample.Status.ABORTED
            return sample

        assert isinstance(trace, TokenTrace), "AvaCore rollouts must drive the policy through a token-level client"
        if self.recording is not None:
            write = asyncio.create_task(self.record(row, sample, trace, reward, sampling_params))
            self.writes.add(write)
            if self._monitor.enabled:
                self._writes_started[write] = time.monotonic()
            self._monitor.increment("record/scheduled_total")
            write.add_done_callback(self._write_done)
        samples = [
            filled(self.args, copy.deepcopy(sample), node, reward.score)
            for node in flattened(trace)
            if isinstance(node, TokenTrace) and any(segment.is_generated for segment in node.segments)
        ]
        return samples[0] if len(samples) == 1 else samples

    def _write_done(self, write: asyncio.Task) -> None:
        self.writes.discard(write)
        self._writes_started.pop(write, None)

    def _record_perf_snapshot(self) -> dict[str, float]:
        started = self._writes_started.copy().values()
        metrics = {
            "pending_writes": len(self.writes),
            "oldest_pending_seconds": max((time.monotonic() - t for t in started), default=0.0),
        }
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
            PostgresBackend(self.recording["postgres"], min_size=1, max_size=4)
        )
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

    async def record(
        self,
        row: Row,
        sample: Sample,
        trace: TokenTrace,
        reward: Reward,
        sampling_params: dict[str, Any],
    ) -> None:
        started = time.monotonic()
        try:
            assert sample.epoch is not None and sample.index is not None
            n = self.args.n_samples_per_prompt
            query_id = str(sample.metadata["_index"])
            trial_id = sample.epoch * n + sample.index % n
            if self.run is None:
                self.run = asyncio.create_task(self.open(sampling_params))
            with self._monitor.wait("record/open_wait"):
                run = await self.run
            with self._monitor.wait("record/create_rollout"):
                await run.create_rollout(
                    query_id=query_id,
                    trial_id=trial_id,
                    instance=row,
                    trace=trace,
                    reward=reward,
                    status="completed",
                    metadata={"weight_version": int(trace.last_assistant().metadata["weight_version"])},
                )
            self._monitor.increment("record/success_total")
        except Exception as error:
            self.failed_writes += 1
            self._monitor.increment("record/failed_total")
            if type(error).__name__ == "PoolTimeout":
                self._monitor.increment("record/pool_timeout_total")
            else:
                self._monitor.increment("record/other_errors_total")
            if self.failed_writes in (1, 10, 100) or self.failed_writes % 1000 == 0:
                logger.warning(
                    "Recording rollouts to Postgres failed %d times; last error: %r", self.failed_writes, error
                )
        finally:
            self._monitor.observe("record/end_to_end", time.monotonic() - started)


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    state = input.state
    if (rollout := getattr(state, "avacore", None)) is None:
        rollout = state.avacore = AvaCoreRollout(input.args, input.args.avacore_config)
    return GenerateFnOutput(samples=await rollout.generate(input.sample, input.sampling_params))


def flattened(trace: Trace) -> list[Trace]:
    return [trace] + [node for subtrace in trace.subtraces for node in flattened(subtrace)]


def filled(args: Namespace, sample: Sample, trace: TokenTrace, reward: float) -> Sample:
    # Context after the last generated turn trains nothing, and no request routed it for replay.
    segments = trace.segments[: max(i for i, segment in enumerate(trace.segments) if segment.is_generated) + 1]
    length = sum(len(segment.tokens) for segment in segments)
    prompt_length = len(segments[0].tokens)
    sample.tokens = trace.tokens[:length]
    sample.response = "".join(segment.template for segment in segments[1:])
    sample.response_length = length - prompt_length
    sample.loss_mask = trace.loss_mask[prompt_length:length]
    sample.rollout_log_probs = [log_prob or 0.0 for log_prob in trace.log_probs[prompt_length:length]]
    spans, offset = [], 0
    for segment in segments:
        if segment.is_generated:
            spans.append((offset, offset + len(segment.tokens)))
        offset += len(segment.tokens)
    finished = [message for message in trace.messages if "finish_reason" in message.metadata]
    for message, (start, end) in zip(finished, spans, strict=True):
        # The trace keeps log probs per token, not per call: anchor the call's weight versions to its own segment.
        versioning = ("weight_version", "weight_versions")
        sample.update_from_meta_info(args, {k: v for k, v in message.metadata.items() if k not in versioning})
        sample.weight_versions[-1] = WeightVersionsPerCall.from_meta_info(
            message.metadata | {"output_token_logprobs": [None] * (end - start)}, output_end=end
        )
    if args.use_rollout_routing_replay and sample.status != Sample.Status.ABORTED:
        assert (
            trace.routed_experts
        ), "Internal error: routing replay enabled but no routed experts returned. Perhaps some traces are not generated by the configured generate function."
        routed = np.frombuffer(b"".join(pybase64.b64decode(chunk) for chunk in trace.routed_experts), dtype=np.int32)
        row = args.num_layers * args.moe_router_topk
        if routed.size == length * row:
            # sglang also forwarded the final token, whose routing feeds no training position
            routed = routed[: (length - 1) * row]
        sample.rollout_routed_experts = routed.reshape(length - 1, args.num_layers, args.moe_router_topk)
    sample.reward = reward
    return sample
