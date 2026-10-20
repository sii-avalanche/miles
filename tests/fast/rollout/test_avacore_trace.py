import json
from argparse import Namespace
from dataclasses import replace
from unittest.mock import AsyncMock

import numpy as np
import pybase64
import pytest
from ava_core.core import ChatMessage, Trace
from ava_core.store.postgres import message_rows

from miles.rollout.avacore_trace import (
    _legacy_trace,
    _segment,
    conversation_tips,
    restore_traces,
    snapshot_traces,
    write_record,
)
from miles.rollout.avacore_rollout import filled
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


def test_graph_roundtrip_preserves_shared_identity_sources_tokens_and_routes():
    root = Trace(message=ChatMessage(role="user", content="question"))
    shared = turn(root, [1, 2], [3, 4])
    left = turn(shared, [5], [6])
    source = turn(Trace(message=ChatMessage(role="user", content="subagent")), [7], [8])
    right = replace(turn(shared, [9], [10], 2), sources=(source,), metadata={"extra": 1})
    right = replace(
        right, segments=(*right.segments[:1], replace(right.segments[1], metadata={"routed_experts": "AA=="}))
    )
    snapshot = json.loads(json.dumps(snapshot_traces((left, right))))
    restored = restore_traces(snapshot)
    assert restored[0].prev is restored[1].prev
    assert restored[1].sources[0].tokens == [7, 8]
    assert restored[1].segments[-1].metadata["routed_experts"] == "AA=="
    assert restored[1].metadata == {"extra": 1}
    assert list(message_rows(restored).values()) == list(message_rows((left, right)).values())
    assert restored[1].loss_mask == right.loss_mask
    assert len(conversation_tips(restored)) == 3


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


class CharTokenizer:
    def decode(self, tokens, **kwargs):
        return "".join(chr(t) for t in tokens)


def test_legacy_multiturn_preserves_all_tokens_and_rejects_wrong_tokenizer():
    data = dict(
        messages=[
            dict(role="user", content="q"),
            dict(role="assistant", content="a", metadata={"finish_reason": {"type": "stop"}}),
            dict(role="tool", content="tool"),
            dict(role="assistant", content="b", metadata={"finish_reason": {"type": "stop"}}),
        ],
        tools=[],
        tokens=list(b"qaTb"),
        log_probs=[None, -0.1, None, -0.2],
        loss_mask=[0, 1, 0, 1],
        template="qaTb",
        metadata={"old": True},
    )
    trace = _legacy_trace(data, CharTokenizer())
    assert trace.tokens == data["tokens"]
    assert trace.log_probs == data["log_probs"]
    assert trace.loss_mask == data["loss_mask"]
    assert trace.template == data["template"]
    assert trace.metadata == data["metadata"]
    with pytest.raises(ValueError, match="template"):
        _legacy_trace(data | {"template": "different"}, CharTokenizer())


async def test_writer_uses_public_create_rollout_and_restores_reward():
    tip = turn(Trace(message=ChatMessage(role="user", content="question")), [1], [2])
    run = AsyncMock()
    await write_record(
        run,
        dict(
            query_id="7",
            trial_id=2,
            instance={"prompt": "q"},
            trace_graph=snapshot_traces((tip,)),
            reward={"score": 1},
            status="completed",
            metadata={"weight_version": 1},
        ),
    )
    kwargs = run.create_rollout.call_args.kwargs
    assert kwargs["traces"][0].tokens == [1, 2]
    assert kwargs["reward"].score == 1
    assert kwargs["instance"].key == "7"
    assert kwargs["trial_id"] == 2


async def test_nonnumeric_weight_version_is_preserved_without_breaking_database_index():
    tip = turn(Trace(message=ChatMessage(role="user", content="q")), [1], [2])
    run = AsyncMock()
    await write_record(
        run,
        dict(
            query_id="7",
            trial_id=0,
            instance={},
            trace_graph=snapshot_traces((tip,)),
            reward={"score": 1},
            status="completed",
            metadata={"weight_version": "default"},
        ),
    )
    assert run.create_rollout.call_args.kwargs["metadata"] == {
        "weight_version": None,
        "source_weight_version": "default",
    }


async def test_generate_scores_primary_tip_and_emits_each_owned_branch():
    from ava_core.rewards.core import Reward
    from miles.rollout.avacore_rollout import AvaCoreRollout
    from miles.utils.perf_monitor import RuntimeMonitor

    shared = turn(Trace(message=ChatMessage(role="user", content="question")), [1, 2], [3])
    left, right = turn(shared, [4], [5]), turn(shared, [6], [7])
    adapter = object.__new__(AvaCoreRollout)
    adapter.args = Namespace(use_rollout_routing_replay=False, sglang_speculative_algorithm=None)
    adapter.recording = None
    adapter.generate_fn = AsyncMock(return_value=(left, right))
    adapter.reward_fn = AsyncMock(return_value=Reward(score=1))
    adapter._monitor = RuntimeMonitor(args=Namespace(perf_monitor_interval=10), role="rollout", sink=lambda _: None)
    samples = await adapter.generate(Sample(index=0, prompt="question"), {})
    assert len(samples) == 2
    assert adapter.reward_fn.call_args.args[0] is left
    assert samples[0].loss_mask == [1, 0, 1]
    assert samples[1].loss_mask == [0, 0, 1]
