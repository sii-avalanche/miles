import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from ava_core.core import ChatMessage, Trace
from ava_core.core.segment import ChatSegment
from ava_core.core.trace import decode_traces, encode_traces
from ava_core.rewards.credit_resolution import build_credit_context
from ava_core.rewards.feedback import LocalCredit, RewardFeedback, StepTarget, TokenSpan

from miles.backends.training_utils.data.context_parallel import assemble_log_prob_from_cp, slice_log_prob_with_cp
from miles.backends.training_utils.loss import credit_shaping
from miles.backends.training_utils.loss.hub.math_utils import compute_policy_loss
from miles.backends.training_utils.loss.objective import compute_advantages_and_returns
from miles.backends.training_utils.parallel import GroupInfo, set_parallel_state
from miles.ray.rollout.train_data_conversion import _package_shards, convert_samples_to_train_data
from miles.rollout.credit_assignment import fill_credit_assignment
from miles.utils.credit_assignment import (
    CreditConfig,
    ReweightPolicy,
    checked_coefficients,
    span_scores,
    token_fingerprint,
    validate_credit_args,
)
from miles.utils.types import Sample
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args, make_parallel_state
from tests.fast.ray.rollout.conftest import make_args as make_conversion_args


def payload(tokens, length, spans=None):
    return {
        "version": 1,
        "token_hash": token_fingerprint(tokens),
        "response_length": length,
        "reward": 0.75,
        "annotations": [],
        "spans": spans or [],
        "ignored": [],
    }


def span(start=0, end=1, score=1, confidence=1):
    return dict(start=start, end=end, score=score, confidence=confidence)


def batch(advantage=2, length=6, prompt=10):
    tokens = torch.arange(prompt + length)
    return {
        "tokens": [tokens],
        "total_lengths": [len(tokens)],
        "response_lengths": [length],
        "loss_masks": [torch.tensor(([1, 0, 1, 1, 1, 1])[:length])],
        "rewards": [advantage],
        "sample_indices": [7],
        "log_probs": [torch.zeros(length)],
        "credit_assignment": [payload(tokens, length, [span()])],
    }


@pytest.fixture(autouse=True)
def parallel_state():
    state = make_parallel_state()
    yield state
    set_parallel_state(state)


@pytest.mark.parametrize("advantage", [2.0, -2.0, 0.0])
@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0])
def test_bounded_nonnegative_mass_and_desirability(advantage, strength):
    scores = torch.tensor([1.0, -1.0, 0.5, 0.0, 1.0])
    mask = torch.tensor([1, 1, 1, 1, 0])
    weights = checked_coefficients(
        ReweightPolicy(strength), CreditConfig(mode="reweight", strength=strength), advantage, scores, mask
    )
    assert weights.min() >= 1 - strength and weights.max() <= 1 + strength
    assert weights[mask.bool()].sum() == pytest.approx(4.0)
    assert weights[-1] == 1
    final = advantage * weights
    assert final[0] >= final[1]  # More desirable tokens have more positive (or less negative) advantage.
    assert final[mask.bool()].abs().sum() == pytest.approx(abs(advantage) * 4)
    if advantage == 0:
        assert torch.equal(weights, torch.ones_like(weights))


@pytest.mark.parametrize("mode", ["none", "reweight"])
@pytest.mark.parametrize("annotations", [None, []])
def test_no_annotations_bitwise_baseline(mode, annotations, tmp_path):
    data = batch()
    if annotations is None:
        del data["credit_assignment"]
    else:
        data["credit_assignment"][0]["spans"] = annotations
    original = [torch.full((6,), 2.0)]
    shaped = credit_shaping.shape_token_advantages(
        make_args(credit_mode=mode, credit_audit_dir=str(tmp_path)), data, original
    )
    assert shaped[0] is original[0]


def test_overlap_dedup_confidence_and_invalid_coordinates():
    tokens, mask = [10, 11, 12, 13], torch.tensor([1, 0, 1, 1])
    spans = [
        span(0, 3, 1),
        span(0, 3, 1),
        span(2, 4, -1, 0.5),
        span(3, 4, -1, 0.1),
        span(-1, 2),
        span(0, 99),
        span(0, 1, float("nan")),
        span(0, 1, 1, 2),
    ]
    scores, errors = span_scores(payload(tokens, 4, spans), tokens, mask, min_confidence=0.5)
    torch.testing.assert_close(scores, torch.tensor([1.0, 0.0, 0.25, -0.5], dtype=torch.float64))
    assert len(errors) == 4
    stale, errors = span_scores(payload(tokens, 4, spans), tokens[::-1], mask, min_confidence=0.5)
    assert not stale.any() and "changed" in errors[0]


@pytest.mark.parametrize("value", [float("nan"), -1, 1.1])
def test_config_rejects_invalid_bounds(value):
    with pytest.raises(ValueError):
        CreditConfig(strength=value)
    with pytest.raises(ValueError):
        CreditConfig(min_confidence=value)


def test_signed_and_incompatible_objectives_are_not_silently_enabled():
    with pytest.raises(ValueError, match="deferred"):
        CreditConfig(mode="signed")
    for changes in ({"advantage_estimator": "ppo"}, {"use_opd": True}, {"rewards_normalization": False}):
        with pytest.raises(ValueError):
            validate_credit_args(make_args(credit_mode="reweight", **{"rewards_normalization": True, **changes}))


@pytest.mark.parametrize("values", [[1.2, 1.2], [-1.0, 3.0], [float("nan"), 1.0]])
def test_plugins_cannot_bypass_mass_and_bounds(values):
    plugin = SimpleNamespace(shape=lambda *args: torch.tensor(values))
    with pytest.raises(ValueError):
        checked_coefficients(plugin, CreditConfig(mode="reweight"), 1.0, torch.tensor([1.0, -1.0]), torch.ones(2))


@pytest.mark.parametrize("advantage", [2.0, -2.0, 0.0])
def test_training_hook_mask_audit_and_ppo_clip_unchanged(advantage, tmp_path):
    data = batch(advantage)
    args = make_args(credit_mode="reweight", credit_audit_dir=str(tmp_path), kl_coef=0)
    compute_advantages_and_returns(args, data)
    final = data["advantages"][0]
    valid = data["loss_masks"][0].bool()
    assert final[~valid].tolist() == [advantage]
    assert final[valid].sum() == pytest.approx(advantage * valid.sum().item())
    # The existing PPO primitive still clips ratios, never the shaped advantage.
    ratios = torch.tensor([1.8, 0.3, 0.9, 1.1, 2.0, 0.8])
    actual, _, _ = compute_policy_loss(-ratios.log(), final, 0.2, 0.2)
    expected = torch.maximum(-ratios * final, -ratios.clamp(0.8, 1.2) * final)
    torch.testing.assert_close(actual, expected)
    rows = [json.loads(line) for path in tmp_path.glob("*.jsonl") for line in path.read_text().splitlines()]
    assert rows[0]["feedback"]["reward"] == 0.75
    assert rows[0]["original_group_advantage"] == advantage
    assert rows[0]["final_token_advantages"] == final.tolist()


def test_shaping_after_whitening_preserves_normalized_baseline_mass(monkeypatch, tmp_path):
    # Run the real normalization; a single-rank all_reduce is the identity.
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    data = batch(2.0)
    other = batch(-0.5)
    for key in data:
        data[key] += other[key]
    args = make_args(kl_coef=0, normalize_advantages=True, credit_audit_dir=str(tmp_path))
    baseline = copy.deepcopy(data)
    compute_advantages_and_returns(args, baseline)
    args.credit_mode = "reweight"
    compute_advantages_and_returns(args, data)
    for before, after, mask in zip(baseline["advantages"], data["advantages"], data["loss_masks"], strict=True):
        torch.testing.assert_close(before[mask.bool()].sum(), after[mask.bool()].sum())
        assert not torch.equal(before, after)


def test_zero_group_is_unchanged_even_if_batch_whitening_shifts_it(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    data, other = batch(0.0), batch(2.0)
    for key in data:
        data[key] += other[key]
    args = make_args(kl_coef=0, normalize_advantages=True, credit_audit_dir=str(tmp_path))
    baseline = copy.deepcopy(data)
    compute_advantages_and_returns(args, baseline)
    args.credit_mode = "reweight"
    compute_advantages_and_returns(args, data)
    assert torch.equal(baseline["advantages"][0], data["advantages"][0])


@pytest.mark.parametrize("qkv_format", ["thd", "bshd"])
@pytest.mark.parametrize("length", [1, 6])
def test_cp_global_mass_and_empty_response_ranks(parallel_state, monkeypatch, tmp_path, qkv_format, length):
    data = batch(-2.0, length=length, prompt=21)
    data["max_seq_lens"] = [32] if qkv_format == "bshd" else None
    original = torch.full((length,), -2.0)
    args = make_args(credit_mode="reweight", qkv_format=qkv_format, credit_audit_dir=str(tmp_path))
    expected = credit_shaping.shape_token_advantages(args, copy.deepcopy(data), [original])[0]
    count = data["loss_masks"][0].sum().item()
    monkeypatch.setattr(
        torch.distributed, "all_reduce", lambda tensor, **kwargs: tensor.copy_(torch.tensor([[-2.0 * count, count]]))
    )
    chunks = {}
    for rank in range(4):
        set_parallel_state(replace(parallel_state, cp=GroupInfo(rank=rank, size=4, group=None)))
        local = slice_log_prob_with_cp(original, 21 + length, length, qkv_format, 32 if qkv_format == "bshd" else None)
        chunks[rank] = credit_shaping.shape_token_advantages(args, copy.deepcopy(data), [local])[0]
    actual = assemble_log_prob_from_cp(
        chunks, 21 + length, length, 4, qkv_format, 32 if qkv_format == "bshd" else None
    )
    torch.testing.assert_close(actual, expected)
    assert any(len(value) == 0 for value in chunks.values())


def test_sample_roundtrip_retry_and_dp_payload_alignment():
    samples = [
        Sample(
            index=i,
            group_index=0,
            tokens=[8, i, 3],
            response_length=2,
            loss_mask=[1, 1],
            reward=float(i),
            status=Sample.Status.COMPLETED,
            credit_assignment=payload([8, i, 3], 2, [span()]),
        )
        for i in range(2)
    ]
    restored = Sample.from_dict(json.loads(json.dumps(samples[0].to_dict())))
    assert restored.credit_assignment == samples[0].credit_assignment
    restored.reset_for_retry()
    assert restored.credit_assignment is None
    legacy = samples[0].to_dict()
    del legacy["credit_assignment"]
    assert Sample.from_dict(legacy).credit_assignment is None
    args = make_conversion_args(
        rewards_normalization=True,
        grpo_std_normalization=False,
        reward_key=None,
        n_samples_per_prompt=2,
        rollout_batch_size=1,
        overlong_reward_mode="none",
    )
    data = convert_samples_to_train_data(args, samples, {}, None, None)
    assert data["rewards"] == [-0.5, 0.5]
    shards = _package_shards(args, data, [[1], [0]])
    for shard, index in zip(shards, [1, 0], strict=True):
        assert shard["sample_indices"] == [index]
        assert shard["credit_assignment"][0] == samples[index].credit_assignment


def test_trace_to_sample_to_token_advantages_without_retokenizing(tmp_path):
    class Tokenizer:
        def decode(self, tokens, **kwargs):
            return "".join(map(chr, tokens))

        def encode(self, *args, **kwargs):
            raise AssertionError("Retokenization is forbidden")

    def node(previous, text, generated, role):
        return Trace(
            prev=previous,
            generated=generated,
            message=ChatMessage(role=role, content=text),
            segments=(ChatSegment(tokens=tuple(map(ord, text)), tokenizer=Tokenizer(), is_generated=generated),),
        )

    prompt = node(None, "Q?", False, "user")
    first = node(prompt, "abcd", True, "assistant")
    tool = node(first, "TOOL", False, "tool")
    tip = node(tool, "efgh", True, "assistant")
    (restored,) = decode_traces(json.loads(json.dumps(encode_traces((tip,)))))
    context = build_credit_context((restored,))
    a, b = context.segments
    context = build_credit_context((restored,), steps={"reason": [TokenSpan(a, 1, 3), TokenSpan(b, 0, 1)]})
    reward = RewardFeedback(0.75, (LocalCredit(StepTarget("reason"), -1),)).to_reward()
    sample = Sample(tokens=restored.tokens, response_length=12, loss_mask=restored.loss_mask[2:])
    fill_credit_assignment(sample, restored, reward, context)
    # Canonical offsets include the intervening tool result, but never target it.
    assert [(s["start"], s["end"]) for s in sample.credit_assignment["spans"]] == [(1, 3), (8, 9)]
    data = {
        "tokens": [torch.tensor(sample.tokens)],
        "total_lengths": [14],
        "response_lengths": [12],
        "loss_masks": [torch.tensor(sample.loss_mask)],
        "rewards": [2.0],
        "credit_assignment": [json.loads(json.dumps(sample.credit_assignment))],
    }
    before = torch.full((12,), 2.0)
    (after,) = credit_shaping.shape_token_advantages(
        make_args(credit_mode="reweight", credit_audit_dir=str(tmp_path)), data, [before]
    )
    assert torch.equal(after[4:8], before[4:8])
    assert after[1] < after[0]
    assert after[data["loss_masks"][0].bool()].sum() == pytest.approx(16.0)
