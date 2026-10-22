import asyncio
import copy
import logging
import os
from argparse import Namespace
from collections.abc import Iterator, Mapping
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import numpy as np
import pybase64
from ava_core.config import interpolated, launch_converter
from ava_core.core import Schema, Trace
from ava_core.core.trace import iter_trace_nodes
from ava_core.generate.core import GenerateFunction
from ava_core.generate.core import Sample as Row
from ava_core.rewards import RewardFunction
from ava_core.rewards.core import Reward
from ava_core.runner import RECOVERABLE_ERRORS
from ava_core.store.core import Run
from ava_core.store.postgres import PostgresBackend

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
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

    async def generate(self, sample: Sample, sampling_params: dict[str, Any]) -> Sample | list[Sample]:
        # Compact siblings of one episode share rollout_id (validate_compact_rollout_ids); every
        # return below is a list, not a bare Sample, so the batch stays uniformly nested for the flatten.
        sample.rollout_id = sample.index
        row = Row({**sample.metadata, "prompt": sample.prompt, "label": sample.label}, key=lambda _: sample.index)
        replay = {"return_routed_experts": True} if self.args.use_rollout_routing_replay else {}
        try:
            traces = await self.generate_fn(row, sampling_params=sampling_params, **replay)
            if any(
                message.metadata["finish_reason"]["type"] == "abort"
                for node in iter_trace_nodes(traces)
                for message in (node.message,)
                if "finish_reason" in message.metadata
            ):
                sample.status = Sample.Status.ABORTED
                return [sample]
            reward = await self.reward_fn(traces[0], row)
        except RECOVERABLE_ERRORS as error:
            logger.warning("AvaCore rollout of sample %s aborted: %r", sample.index, error)
            sample.status = Sample.Status.ABORTED
            return [sample]

        assert traces and any(node.segments for node in iter_trace_nodes(traces)), (
            "AvaCore rollouts must drive the policy through a token-level client"
        )
        if self.recording is not None:
            write = asyncio.create_task(self.record(row, sample, traces, reward, sampling_params))
            self.writes.add(write)
            write.add_done_callback(self.writes.discard)
        samples, owned = [], set()
        for tip in conversation_tips(traces):
            generated = {node for node in tip.prefixes if node.generated and node.segments}
            if generated - owned:
                samples.append(filled(self.args, copy.deepcopy(sample), tip, reward.score, owned=owned))
                owned.update(generated)
        return samples

    async def open(self, sampling_params: dict[str, Any]) -> Run:
        assert self.recording is not None
        store = await self.stack.enter_async_context(
            PostgresBackend(self.recording["postgres"], min_size=1, max_size=4)
        )
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
        traces: tuple[Trace, ...],
        reward: Reward,
        sampling_params: dict[str, Any],
    ) -> None:
        try:
            assert sample.epoch is not None and sample.index is not None
            n = self.args.n_samples_per_prompt
            query_id = str(sample.metadata["_index"])
            trial_id = sample.epoch * n + sample.index % n
            if self.run is None:
                self.run = asyncio.create_task(self.open(sampling_params))
            run = await self.run
            await run.create_rollout(
                query_id=query_id,
                trial_id=trial_id,
                instance=row,
                traces=traces,
                reward=reward,
                status="completed",
                metadata={"weight_version": int(traces[0].last_assistant().metadata["weight_version"])},
            )
        except Exception as error:
            self.failed_writes += 1
            if self.failed_writes in (1, 10, 100) or self.failed_writes % 1000 == 0:
                logger.warning(
                    "Recording rollouts to Postgres failed %d times; last error: %r", self.failed_writes, error
                )


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    state = input.state
    if (rollout := getattr(state, "avacore", None)) is None:
        rollout = state.avacore = AvaCoreRollout(input.args, input.args.avacore_config)
    return GenerateFnOutput(samples=await rollout.generate(input.sample, input.sampling_params))


def conversation_tips(traces: tuple[Trace, ...]) -> Iterator[Trace]:
    tips = {}
    for node in iter_trace_nodes(traces):
        tips.pop(node.prev, None)
        tips[node] = None
    # Match the store's ownership order: listed traces first, source-only tips next.
    for trace in dict.fromkeys(traces):
        tips.pop(trace, None)
        yield trace
    yield from tips


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
