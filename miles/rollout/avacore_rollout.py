import asyncio
import copy
import logging
import os
from argparse import Namespace
from collections.abc import Iterator, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
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
from ava_core.runner import RECOVERABLE_ERRORS, Retry
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


@dataclass(frozen=True)
class _GroupOversampling:
    """Retry failed slots without increasing the group's live rollout width."""

    max_extra_attempts_per_slot: int = 0
    initial_backoff_seconds: float = 2.0
    max_backoff_seconds: float = 20.0

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "_GroupOversampling":
        raw = config.get("group_oversampling", {})
        if not isinstance(raw, Mapping):
            raise TypeError("group_oversampling must be a table")
        unknown = set(raw) - {
            "max_extra_attempts_per_slot",
            "initial_backoff_seconds",
            "max_backoff_seconds",
        }
        if unknown:
            raise ValueError(f"unknown group_oversampling options: {sorted(unknown)}")
        policy = cls(**raw)
        if isinstance(policy.max_extra_attempts_per_slot, bool) or policy.max_extra_attempts_per_slot < 0:
            raise ValueError("group_oversampling.max_extra_attempts_per_slot must be a non-negative integer")
        if not isinstance(policy.max_extra_attempts_per_slot, int):
            raise TypeError("group_oversampling.max_extra_attempts_per_slot must be an integer")
        if policy.initial_backoff_seconds < 0 or policy.max_backoff_seconds < policy.initial_backoff_seconds:
            raise ValueError("group_oversampling backoff must satisfy 0 <= initial <= max")
        return policy

    def backoff(self, failed_attempt: int) -> float:
        return min(self.max_backoff_seconds, self.initial_backoff_seconds * 2 ** (failed_attempt - 1))


def _retryable(error: BaseException) -> bool:
    """Honor the remote service's classification, with local AvaCore as fallback."""

    classified = getattr(error, "retryable", None)
    return bool(classified) if classified is not None else Retry.should_retry(error)


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
        # GenerateState constructs this adapter while the rollout worker is
        # initialized, so malformed run-wide config fails before requests.
        self.group_oversampling = _GroupOversampling.from_config(config)

    async def generate(
        self, sample: Sample, sampling_params: dict[str, Any], *, evaluation: bool = False
    ) -> Sample | list[Sample]:
        metadata = {
            key: value for key, value in sample.metadata.items() if key not in {_CANDIDATE_SLOT, _CANDIDATE_WIDTH}
        }
        row = Row({**metadata, "prompt": sample.prompt, "label": sample.label}, key=lambda _: sample.index)
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
        policy = self.group_oversampling
        max_attempts = policy.max_extra_attempts_per_slot + 1
        for attempt in range(1, max_attempts + 1):
            try:
                traces = await self.generate_fn(row, sampling_params=sampling_params, **replay)
                if not isinstance(traces, tuple) or not traces or not all(isinstance(trace, Trace) for trace in traces):
                    raise TypeError("AvaCore generate must return a nonempty tuple of Trace objects")
                aborted = any(
                    message.metadata["finish_reason"]["type"] == "abort"
                    for node in iter_trace_nodes(traces)
                    for message in (node.message,)
                    if "finish_reason" in message.metadata
                )
                if aborted:
                    logger.warning("AvaCore rollout slot aborted by finish reason: sample=%s", sample.index)
                    sample.status = Sample.Status.ABORTED
                    return sample
                reward = await self.reward_fn(traces[0], row)
                break
            except RECOVERABLE_ERRORS as error:
                if attempt == max_attempts or not _retryable(error):
                    logger.warning(
                        "AvaCore rollout slot aborted: sample=%s attempts=%d/%d retryable=%s error=%r",
                        sample.index,
                        attempt,
                        max_attempts,
                        _retryable(error),
                        error,
                    )
                    sample.status = Sample.Status.ABORTED
                    return sample
                await self._retry_slot(sample, attempt, max_attempts, type(error).__name__)

        assert traces and any(
            node.segments for node in iter_trace_nodes(traces)
        ), "AvaCore rollouts must drive the policy through a token-level client"
        if self.recording is not None:
            write = asyncio.create_task(self.record(row, sample, traces, reward, sampling_params))
            self.writes.add(write)
            write.add_done_callback(self.writes.discard)
        samples, owned = [], set()
        for tip in conversation_tips(traces):
            generated = {node for node in tip.prefixes if node.generated and node.segments}
            if generated - owned:
                samples.append(
                    await asyncio.to_thread(
                        filled,
                        self.args,
                        copy.deepcopy(sample),
                        tip,
                        reward.score,
                        owned=set(owned),
                        evaluation=evaluation,
                    )
                )
                owned.update(generated)
        if not samples:
            raise ValueError("AvaCore rollout contains no generated training tokens")
        if len(samples) > 1:
            rollout_id = sample.rollout_id if sample.rollout_id is not None else sample.index
            if rollout_id is None:
                raise ValueError("A multi-sample AvaCore rollout needs an input sample index or rollout_id")
            for output in samples:
                output.rollout_id = rollout_id
        return samples[0] if len(samples) == 1 else samples

    async def _retry_slot(self, sample: Sample, failed_attempt: int, max_attempts: int, reason: str) -> None:
        policy = self.group_oversampling
        delay = policy.backoff(failed_attempt)
        logger.warning(
            "AvaCore rollout slot retry: sample=%s next_attempt=%d/%d delay=%.1fs reason=%s",
            sample.index,
            failed_attempt + 1,
            max_attempts,
            delay,
            reason,
        )
        sample.reset_for_retry()
        sample.status = Sample.Status.PENDING
        if delay:
            await asyncio.sleep(delay)

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
                "weight_version": _weight_version(traces[0]),
                "weight_versions": [
                    version
                    for trace in traces
                    for node in trace.prefixes
                    if node.generated and (version := node.message.metadata.get("weight_version")) is not None
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


def _weight_version(trace: Trace) -> int:
    """Return the version from the last token-level policy turn, even when a tool result ends the trace."""
    for node in reversed(trace.prefixes):
        if node.generated and node.segments and "weight_version" in node.message.metadata:
            version = node.message.metadata["weight_version"]
            return 0 if version == "default" else int(version)
    raise ValueError("AvaCore rollout trace has no generated weight_version")


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    state = input.state
    if error := getattr(state, "avacore_init_error", None):
        raise RuntimeError(error)
    if (rollout := getattr(state, "avacore", None)) is None:
        try:
            rollout = state.avacore = AvaCoreRollout(input.args, input.args.avacore_config)
        except Exception as error:
            # Keep only text: a traceback would retain partially constructed clients.
            state.avacore_init_error = f"AvaCore rollout initialization failed: {error}"
            raise
    return GenerateFnOutput(
        samples=await rollout.generate(input.sample, input.sampling_params, evaluation=input.evaluation)
    )


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


def filled(
    args: Namespace,
    sample: Sample,
    trace: Trace,
    reward: float,
    *,
    owned: set[Trace] | None = None,
    evaluation: bool = False,
) -> Sample:
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
    finished = [node.message for node in trace.prefixes if node.generated and node.segments]
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
        chunks = [segment.metadata["routed_experts"] for segment in segments if "routed_experts" in segment.metadata]
        assert (
            chunks
        ), "Internal error: routing replay enabled but no routed experts returned. Perhaps some traces are not generated by the configured generate function."
        routed = np.frombuffer(b"".join(pybase64.b64decode(chunk) for chunk in chunks), dtype=np.int32)
        row = args.num_layers * args.moe_router_topk
        if routed.size == length * row:
            # sglang also forwarded the final token, whose routing feeds no training position
            routed = routed[: (length - 1) * row]
        sample.rollout_routed_experts = routed.reshape(length - 1, args.num_layers, args.moe_router_topk)
    sample.reward = reward
    return sample


def _with_sampling_metadata(args: Namespace, sample: Sample, segments, messages) -> Sample:
    k, mode = args.rollout_top_logprobs_num, args.rollout_sampling_logprobs_mode
    support = mode == "support"
    contract = dict(
        count=k,
        temperature=args.rollout_temperature,
        top_p=args.rollout_top_p,
        top_k=args.rollout_top_k,
        mode=mode,
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
