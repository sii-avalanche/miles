import asyncio
import copy
import logging
import os
from argparse import Namespace
from collections.abc import Iterator, Mapping
from contextlib import AsyncExitStack
from dataclasses import replace
from math import isfinite
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

from miles.rollout.generate_utils.rollout_topk_logprobs import (
    append_rollout_topk_logprobs,
    configure_rollout_topk_logprobs_request,
    pad_rollout_topk_logprobs,
    validate_rollout_topk_logprobs_sample,
)
from miles.rollout.generate_utils.sampling_mask import append_forced_sampling_tokens, append_sampling_metadata
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.types import Sample, WeightVersionsPerCall

__all__ = ["AvaCoreRollout", "generate"]

logger = logging.getLogger(__name__)

_CANDIDATE_SLOT = "_group_oversampling_candidate_slot"
_CANDIDATE_WIDTH = "_group_oversampling_candidate_width"


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

    async def generate(
        self, sample: Sample, sampling_params: dict[str, Any], *, evaluation: bool = False
    ) -> list[Sample]:
        metadata = {
            key: value for key, value in sample.metadata.items() if key not in {_CANDIDATE_SLOT, _CANDIDATE_WIDTH}
        }
        row_key = sample.metadata.get("_index", sample.index)
        row = Row({**metadata, "prompt": sample.prompt, "label": sample.label}, key=lambda _: row_key)
        replay = {"return_routed_experts": True} if self.args.use_rollout_routing_replay else {}
        if getattr(self.args, "rollout_top_logprobs_num", 0) and not evaluation:
            request = {"sampling_params": dict(sampling_params)}
            configure_rollout_topk_logprobs_request(self.args, request)
            sampling_params = request["sampling_params"]
            replay["sampler_logprobs"] = dict(
                count=self.args.rollout_top_logprobs_num,
                temperature=self.args.rollout_temperature,
                top_p=self.args.rollout_top_p,
                top_k=self.args.rollout_top_k,
                mode=self.args.rollout_sampling_logprobs_mode,
            )
        try:
            traces = await self.generate_fn(row, sampling_params=sampling_params, **replay)
            if not isinstance(traces, tuple) or not traces or not all(isinstance(t, Trace) for t in traces):
                raise TypeError("AvaCore generate must return a nonempty tuple of Trace objects")
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

        assert any(node.segments for node in iter_trace_nodes(traces)), (
            "AvaCore rollouts must drive the policy through a token-level client"
        )
        owned: set[Trace] = set()
        samples = []
        for trace in conversation_tips(traces):
            generated = {node for node in trace.prefixes if node.generated}
            if any(not node.segments for node in generated):
                raise ValueError("Training requires token-level segments for every generated turn")
            if generated - owned:
                samples.append(
                    await asyncio.to_thread(
                        filled,
                        self.args,
                        copy.deepcopy(sample),
                        trace,
                        reward.score,
                        owned=set(owned),
                        evaluation=evaluation,
                    )
                )
            owned.update(generated)
        if not samples:
            raise ValueError("AvaCore rollout contains no generated training tokens")
        if self.recording is not None:
            write = asyncio.create_task(self.record(row, sample, traces, reward, sampling_params))
            self.writes.add(write)
            write.add_done_callback(self.writes.discard)
        if len(samples) > 1:
            rollout_id = sample.rollout_id if sample.rollout_id is not None else sample.index
            if rollout_id is None:
                raise ValueError("A compact AvaCore rollout needs an input index or rollout_id")
            for output in samples:
                output.rollout_id = rollout_id
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
            metadata = {
                "weight_version": next(
                    (
                        node.message.metadata["weight_version"]
                        for node in reversed(traces[0].prefixes)
                        if node.generated and "weight_version" in node.message.metadata
                    ),
                    None,
                ),
                "weight_versions": [
                    v
                    for trace in traces
                    for node in trace.prefixes
                    if node.generated and (v := node.message.metadata.get("weight_version")) is not None
                ],
            }
            if (slot := sample.metadata.get(_CANDIDATE_SLOT)) is not None:
                width = int(sample.metadata[_CANDIDATE_WIDTH])
                trial_id = sample.epoch * width + int(slot)
                metadata |= {_CANDIDATE_SLOT: int(slot), _CANDIDATE_WIDTH: width}
            else:
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
                metadata=metadata,
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
    return GenerateFnOutput(
        samples=await rollout.generate(input.sample, input.sampling_params, evaluation=input.evaluation)
    )


def conversation_tips(traces: tuple[Trace, ...]) -> Iterator[Trace]:
    tips = {}
    for node in iter_trace_nodes(traces):
        tips.pop(node.prev, None)
        tips[node] = None
    for trace in dict.fromkeys(traces):
        tips.pop(trace, None)
        yield trace
    yield from tips


def filled(
    args: Namespace,
    sample: Sample,
    trace: Trace,
    reward: float,
    *,
    owned: set[Trace] | None = None,
    evaluation: bool = False,
) -> Sample:
    owned = owned or set()
    path = trace.path_segments
    # Tool observations after the last generated token train nothing.
    segments = path[: max(i for i, segment in enumerate(path) if segment.is_generated) + 1]
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
    for segment in segments:
        if segment.is_generated and (
            len(segment.log_probs) != len(segment.tokens)
            or any(value is None or not isfinite(value) for value in segment.log_probs)
        ):
            raise ValueError("Generated tokens require finite, aligned rollout logprobs")
    sample.rollout_log_probs = [
        log_prob if log_prob is not None else 0.0 for log_prob in trace.log_probs[prompt_length:length]
    ]
    spans, offset = [], 0
    for segment in segments:
        if segment.is_generated:
            spans.append((offset, offset + len(segment.tokens)))
        offset += len(segment.tokens)
    finished = [node.message for node in trace.prefixes if node.generated and "finish_reason" in node.message.metadata]
    if not evaluation and getattr(args, "rollout_top_logprobs_num", 0):
        sample = _with_sampling_metadata(args, sample, segments, finished)
        sample.validate()
        if getattr(args, "ci_test", False):
            validate_rollout_topk_logprobs_sample(sample, args.rollout_top_logprobs_num)
    for message, (start, end) in zip(finished, spans, strict=True):
        # The trace keeps log probs per token, not per call: anchor the call's weight versions to its own segment.
        versioning = ("weight_version", "weight_versions")
        sample.update_from_meta_info(args, {k: v for k, v in message.metadata.items() if k not in versioning})
        sample.weight_versions[-1] = WeightVersionsPerCall.from_meta_info(
            message.metadata | {"output_token_logprobs": [None] * (end - start)}, output_end=end
        )
    if args.use_rollout_routing_replay and sample.status != Sample.Status.ABORTED:
        chunks = [s.metadata["routed_experts"] for s in segments if "routed_experts" in s.metadata]
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


def _with_sampling_metadata(args: Namespace, sample: Sample, segments, messages) -> Sample:
    """Return a sample with native sampling metadata, leaving the input sample untouched."""
    k, mode = args.rollout_top_logprobs_num, args.rollout_sampling_logprobs_mode
    support = mode == "support"
    contract = dict(
        count=args.rollout_top_logprobs_num,
        temperature=args.rollout_temperature,
        top_p=args.rollout_top_p,
        top_k=args.rollout_top_k,
        mode=args.rollout_sampling_logprobs_mode,
    )
    messages, log_probs = iter(messages), []
    result = replace(
        sample,
        response_length=0,
        rollout_log_probs=log_probs,
        rollout_topk_token_ids=None,
        rollout_topk_log_probs=None,
        rollout_sampling_mask=None if support else sample.rollout_sampling_mask,
    )
    append_rollout_topk_logprobs(result, {}, k)
    for segment in segments[1:]:
        if segment.is_generated:
            meta = next(messages).metadata
            if meta.get("sampler_logprobs") != contract:
                raise ValueError("AvaCore sampler logprobs contract missing or mismatched; update the policy client")
            log_probs.extend(
                append_sampling_metadata(result, segment.tokens, meta, sampling_logprobs_mode=mode)
                if support
                else segment.log_probs
            )
            append_rollout_topk_logprobs(
                result, meta | {"output_token_logprobs": segment.log_probs}, k, sampling_logprobs_mode=mode
            )
        else:
            if support:
                append_forced_sampling_tokens(result, segment.tokens)
            pad_rollout_topk_logprobs(result, len(segment.tokens))
            log_probs.extend([0.0] * len(segment.tokens))
        result.response_length += len(segment.tokens)
    return result
