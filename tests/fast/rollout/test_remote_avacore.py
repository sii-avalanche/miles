from argparse import Namespace
from dataclasses import replace
from unittest.mock import AsyncMock

import numpy as np
import pybase64
import pytest
from ava_core.core import ChatMessage, Trace
from ava_core.core.training_trace import _segment
from ava_core.rewards.core import Reward

from miles.rollout.avacore_rollout import AvaCoreRollout, filled
from miles.utils.types import Sample


def segment(tokens, generated, text, **metadata):
    return _segment(
        dict(
            tokens=tokens,
            log_probs=[-0.25 if generated else None] * len(tokens),
            template=text,
            is_generated=generated,
            metadata=metadata,
        )
    )


def turn(prev, prompt, output, version=1):
    return Trace(
        prev=prev,
        message=ChatMessage(
            role="assistant",
            content="answer",
            metadata={
                "finish_reason": {"type": "stop"},
                "weight_version": version,
            },
        ),
        generated=True,
        segments=(segment(prompt, False, "prompt"), segment(output, True, "answer")),
    )


def test_training_masks_shared_turn_and_keeps_tool_prompt_and_version_spans():
    root = Trace(message=ChatMessage(role="user", content="question"))
    shared = turn(root, [1, 2], [3, 4])
    tool = Trace(prev=shared, message=ChatMessage(role="tool", content="result"))
    tip = turn(tool, [5, 6], [7, 8], 2)
    args = Namespace(use_rollout_routing_replay=False, sglang_speculative_algorithm=None)
    sample = filled(args, Sample(), tip, 1, owned={shared})
    assert sample.tokens == [1, 2, 3, 4, 5, 6, 7, 8]
    assert sample.response_length == 6
    assert sample.loss_mask == [0, 0, 0, 0, 1, 1]
    assert sample.rollout_log_probs == [-0.25, -0.25, 0, 0, -0.25, -0.25]
    assert len(sample.weight_versions) == 2


def test_routing_replay_reads_generated_segment_metadata():
    root = Trace(message=ChatMessage(role="user", content="question"))
    tip = turn(root, [1, 2], [3, 4])
    routing = np.arange(8, dtype=np.int32).reshape(4, 1, 2)
    tip = replace(
        tip,
        segments=(
            tip.segments[0],
            replace(
                tip.segments[1],
                metadata={
                    "routed_experts": pybase64.b64encode(routing.tobytes()).decode(),
                },
            ),
        ),
    )
    args = Namespace(
        use_rollout_routing_replay=True, sglang_speculative_algorithm=None, num_layers=1, moe_router_topk=2
    )
    sample = filled(args, Sample(), tip, 1)
    np.testing.assert_array_equal(sample.rollout_routed_experts, routing[:3])


@pytest.mark.asyncio
async def test_remote_dispose_closes_client():
    adapter = object.__new__(AvaCoreRollout)
    adapter._remote_generate = type("Client", (), {"loop": None, "aclose": AsyncMock()})()
    await adapter.dispose()
    adapter._remote_generate.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_scores_primary_trace_and_masks_shared_prefix():
    shared = turn(Trace(message=ChatMessage(role="user", content="question")), [1, 2], [3])
    left, right = turn(shared, [4], [5]), turn(shared, [6], [7])
    adapter = object.__new__(AvaCoreRollout)
    adapter.args = Namespace(use_rollout_routing_replay=False, sglang_speculative_algorithm=None)
    adapter.recording = None
    adapter.generate_fn = AsyncMock(return_value=(left, right))
    adapter.reward_fn = AsyncMock(return_value=Reward(score=1))
    samples = await adapter.generate(Sample(index=7, prompt="question"), {})
    assert len(samples) == 2
    assert adapter.reward_fn.call_args.args[0] is left
    assert samples[0].loss_mask == [1, 0, 1]
    assert samples[1].loss_mask == [0, 0, 1]
