from argparse import Namespace
from contextlib import nullcontext
from unittest.mock import AsyncMock, Mock

import pytest
from ava_core.core import ChatMessage, Trace
from ava_core.rewards.core import Reward

from miles.backends.training_utils.data.sampling_mask import get_rollout_sampling_masks
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.rollout.avacore_rollout import AvaCoreRollout, filled
from miles.rollout.avacore_trace import _segment
from miles.utils.types import Sample


@pytest.fixture
def args():
    return Namespace(
        use_sampling_support_replay=True,
        rollout_temperature=1.0,
        use_rollout_routing_replay=False,
        sglang_speculative_algorithm=None,
        reward_key=None,
        advantage_estimator="grpo",
        rewards_normalization=False,
        rollout_top_logprobs_num=0,
        use_dynamic_global_batch_size=False,
    )


def turn(prev=None, *, prompt=(1, 2), output=(3, 4), with_support=True):
    metadata = {"finish_reason": {"type": "stop"}, "weight_version": 0}
    if with_support:
        metadata.update(
            output_token_sampling_mask=[[token, token + 10] for token in output],
            output_token_sampling_logprobs=[-0.125] * len(output),
        )
    return Trace(
        prev=prev,
        message=ChatMessage(role="assistant", content="answer", metadata=metadata),
        generated=True,
        segments=tuple(
            _segment(
                dict(
                    tokens=list(tokens),
                    log_probs=[-0.75 if generated else None] * len(tokens),
                    template="answer" if generated else "prompt",
                    is_generated=generated,
                )
            )
            for tokens, generated in ((prompt, False), (output, True))
        ),
    )


def convert(args, samples):
    return convert_samples_to_train_data(args, samples, {}, None, None)


@pytest.mark.parametrize("evaluation,replay_enabled", [(False, True), (True, True), (False, False)])
async def test_generate_requests_support_only_for_replayed_training(args, evaluation, replay_enabled):
    args.use_sampling_support_replay = replay_enabled
    capture = replay_enabled and not evaluation
    rollout = object.__new__(AvaCoreRollout)
    rollout.args = args
    rollout.recording = None
    rollout._monitor = Mock()
    rollout._monitor.wait.side_effect = lambda *_: nullcontext()
    rollout.generate_fn = AsyncMock(return_value=(turn(with_support=capture),))
    rollout.reward_fn = AsyncMock(return_value=Reward(score=1))
    params = {"top_p": 0.95, "top_k": 512, "temperature": 1.0}

    sample = await rollout.generate(Sample(index=7, metadata={}), params, evaluation=evaluation)

    expected = {"sampling_params": params}
    if capture:
        expected.update(return_sampling_mask=True, sampling_logprobs_mode="selected")
    assert rollout.generate_fn.await_args.kwargs == expected
    if capture:
        batch = convert(args, [sample])
        mask = get_rollout_sampling_masks(batch)[0]
        ids, offsets = mask._as_tensors()
        assert ids.tolist() == [3, 13, 4, 14]
        assert offsets.tolist() == [0, 2, 4]
        assert batch["rollout_log_probs"] == [[-0.125, -0.125]]
    else:
        assert sample.rollout_sampling_mask is None
        assert sample.rollout_log_probs == [-0.75, -0.75]


def test_multiturn_masks_align_with_environment_tokens_and_shared_prefix(args):
    shared = turn()
    tip = turn(shared, prompt=(5, 6), output=(7, 8))
    sample = filled(args, Sample(index=1), tip, 1, owned={shared})
    batch = convert(args, [sample])
    ids, offsets = get_rollout_sampling_masks(batch)[0]._as_tensors()

    assert sample.tokens == [1, 2, 3, 4, 5, 6, 7, 8]
    assert sample.response_length == 6
    assert sample.loss_mask == [0, 0, 0, 0, 1, 1]
    assert ids.tolist() == [3, 13, 4, 14, 5, 6, 7, 17, 8, 18]
    assert offsets.tolist() == [0, 2, 4, 5, 6, 8, 10]
    assert sample.rollout_log_probs == [-0.125, -0.125, 0.0, 0.0, -0.125, -0.125]


def test_replayed_training_rejects_missing_engine_metadata(args):
    with pytest.raises(ValueError, match="missing output_token_sampling_mask"):
        filled(args, Sample(), turn(with_support=False), 1)


@pytest.mark.parametrize("replay_enabled", [True, False])
def test_conversion_rejects_entire_missing_batch_only_when_replay_enabled(args, replay_enabled):
    args.use_sampling_support_replay = False
    sample = filled(args, Sample(index=7), turn(with_support=False), 1)
    args.use_sampling_support_replay = replay_enabled
    if replay_enabled:
        with pytest.raises(ValueError, match="sampling-mask data must be present.*sample_index=7"):
            convert(args, [sample])
    else:
        assert "rollout_sampling_mask_ids" not in convert(args, [sample])


@pytest.mark.parametrize("mode", ["selected", "support"])
async def test_score_centering_preserves_curriculum_and_shared_prefixes(args, mode):
    args.rollout_top_logprobs_num = 2
    args.rollout_sampling_logprobs_mode = mode
    args.rollout_top_p = 1.0
    args.rollout_top_k = 2 if mode == "support" else -1
    args.curriculum_enabled = True
    shared = turn()
    tip = turn(shared, prompt=(5, 6), output=(7, 8))
    contract = dict(count=2, temperature=1.0, top_p=1.0, top_k=args.rollout_top_k, mode=mode)
    for node in (shared, tip):
        metadata = node.message.metadata
        metadata["sampler_logprobs"] = contract
        tokens = node.segments[-1].tokens
        metadata["output_top_logprobs"] = [[[-0.75, token], [-1.0, token + 10]] for token in tokens]
        if mode == "support":
            metadata["output_token_sampling_logprobs"] = [[-0.75, -1.0] for _ in tokens]
    rollout = object.__new__(AvaCoreRollout)
    rollout.args = args
    rollout.recording = None
    rollout._monitor = Mock()
    rollout._monitor.wait.side_effect = lambda *_: nullcontext()
    rollout.generate_fn = AsyncMock(return_value=(shared, tip))
    rollout.reward_fn = AsyncMock(return_value=Reward(score=1, metadata={"grader_invalid": True}))
    params = {"temperature": 1.0, "top_p": 1.0, "top_k": args.rollout_top_k}
    original = params.copy()
    samples = await rollout.generate(Sample(index=7, metadata={}), params)
    assert params == original
    assert rollout.generate_fn.await_args.kwargs["sampler_logprobs"] == contract
    assert len(samples) == 2
    assert all(sample.metadata["grader_invalid"] for sample in samples)
    assert samples[0].rollout_id == samples[1].rollout_id == 7
    assert samples[1].loss_mask == [0, 0, 0, 0, 1, 1]
    assert samples[1].rollout_topk_token_ids.tolist() == [
        [3, 13], [4, 14], [-1, -1], [-1, -1], [7, 17], [8, 18]
    ]
    assert samples[1].rollout_log_probs == [-0.75, -0.75, 0.0, 0.0, -0.75, -0.75]
