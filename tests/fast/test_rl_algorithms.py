"""Algorithm recipes and numerical references, using the production reducers/converter."""

import argparse
import sys
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.data import context_parallel as cp
from miles.backends.training_utils.loss.hub.math_utils import compute_policy_loss
from miles.ray.rollout.train_data_conversion import (
    _normalize_rewards_by_rollout,
    _shape_rollout_rewards,
    convert_samples_to_train_data,
    split_train_data_by_dp_raw,
)
from miles.rollout.filter_hub.base_types import MetricGatherer
from miles.rollout.filter_hub.common_filters import apply_preput_filters, apply_reward_nonzero_std_filter
from miles.utils.arguments import get_miles_extra_args_provider, validate_rl_algorithm_args
from miles.utils.types import Sample


def recipe(monkeypatch, name=None, *extra):
    argv = ([] if name is None else ["--rl-algorithm", name]) + list(extra)
    monkeypatch.setattr(sys, "argv", ["test", *argv])
    parser = get_miles_extra_args_provider()(argparse.ArgumentParser())
    for action in parser._actions:
        action.required = False
    args = parser.parse_args(argv)
    args.rollout_max_response_len = 250000
    args.rollout_batch_size = 2
    args.n_samples_per_prompt = 2
    args.global_batch_size = 4
    args.num_steps_per_rollout = 1
    validate_rl_algorithm_args(args)
    return args


@pytest.mark.parametrize(
    "name,std,clip,mode",
    [
        ("grpo_original", True, 0.2, "response_mean"),
        ("dapo", True, 0.28, "token_mean"),
        ("dr_grpo", False, 0.2, "fixed_token_sum"),
    ],
)
def test_presets(monkeypatch, name, std, clip, mode):
    args = recipe(monkeypatch, name)
    assert args.grpo_std_normalization is std
    assert args.eps_clip_high == clip
    assert args.policy_loss_reduction == mode
    if name == "dapo":
        assert args.overlong_buffer_length == 50000
        assert args.dynamic_sampling_max_groups == 20
    if name == "dr_grpo":
        assert args.policy_loss_fixed_normalizer == 250000


def test_legacy_defaults_and_explicit_overrides(monkeypatch):
    legacy = recipe(monkeypatch)
    assert legacy.policy_loss_reduction is None
    assert legacy.overlong_reward_mode is None
    args = recipe(
        monkeypatch,
        "dapo",
        "--eps-clip-high",
        "0.3",
        "--disable-grpo-std-normalization",
        "--overlong-reward-mode",
        "none",
        "--policy-loss-reduction",
        "response_mean",
    )
    assert args.eps_clip_high == 0.3 and not args.grpo_std_normalization
    assert not args.calculate_per_token_loss and args.overlong_reward_mode == "none"


def samples(rewards=(1.0, 0.0, 1.0, 1.0)):
    return [
        Sample(
            index=i,
            rollout_id=i,
            group_index=i // 2,
            reward=r,
            tokens=list(range(i + 4)),
            response_length=i + 2,
            status=Sample.Status.COMPLETED,
        )
        for i, r in enumerate(rewards)
    ]


@pytest.mark.parametrize("std", [False, True])
def test_group_advantage_keeps_existing_sample_std(monkeypatch, std):
    args = recipe(monkeypatch, "grpo_original" if std else "dr_grpo")
    data = samples()
    order = [0, 2, 1, 3]  # interleaved prompt IDs must not change centering
    result = torch.tensor(
        _normalize_rewards_by_rollout(args, [data[i] for i in order], [data[i].reward for i in order], None)
    )
    scale = (torch.tensor([1.0, 0.0]).std() + 1e-6) if std else 1.0
    torch.testing.assert_close(result, torch.tensor([0.5 / scale, 0, -0.5 / scale, 0]))
    assert result.mean() == 0


@pytest.mark.parametrize("high,expected,gradient", [(0.2, -1.2, 0.0), (0.28, -1.25, -1.25)])
def test_clip_higher_and_negative_advantage(high, expected, gradient):
    log_ratio = torch.tensor([1.25, 0.7, 1.4]).log().requires_grad_()
    loss, clip, _ = compute_policy_loss(-log_ratio, torch.tensor([1.0, -1.0, -1.0]), 0.2, high)
    torch.testing.assert_close(loss, torch.tensor([expected, 0.8, 1.4]))
    loss.sum().backward()
    torch.testing.assert_close(log_ratio.grad, torch.tensor([gradient, 0.0, 1.4]))
    assert clip[1] == 1 and clip[2] == 0


@pytest.mark.parametrize("mode", ["response_mean", "token_mean", "group_token_mean", "fixed_token_sum"])
@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("micro_size", [1, 2, 4])
def test_reducers_cp_and_accumulation_match_full_batch(monkeypatch, mode, cp_size, micro_size):
    """Exercise real zigzag CP slicing and uneven microbatches, including a fully masked response."""
    lengths = [2, 3, 4, 5]
    masks = [torch.tensor(x, dtype=torch.float64) for x in ([1, 1], [1, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1, 1])]
    values = torch.arange(1.0, 15.0, dtype=torch.float64, requires_grad=True)
    chunks = values.split(lengths)
    numerator = [(x * mask).sum() for x, mask in zip(chunks, masks, strict=True)]
    if mode == "response_mean":
        reference = sum(x / m.sum().clamp_min(1) for x, m in zip(numerator, masks, strict=True)) / 4
    elif mode == "token_mean":
        reference = sum(numerator) / sum(m.sum() for m in masks)
    elif mode == "group_token_mean":
        reference = ((numerator[0] + numerator[1]) / 4 + (numerator[2] + numerator[3]) / 5) / 2
    else:
        reference = sum(numerator) / (4 * 8)
    expected_grad = torch.autograd.grad(reference, values, retain_graph=True)[0]
    total = values.sum() * 0
    # DP partitions can put prompt-group members on different ranks. Reducer denominators remain global.
    for partition in ([0, 2], [1, 3]):
        for rank in range(cp_size):
            monkeypatch.setattr(
                cp,
                "get_parallel_state",
                lambda rank=rank: SimpleNamespace(cp=SimpleNamespace(size=cp_size, rank=rank)),
            )
            for start in range(0, len(partition), micro_size):
                idx = partition[start : start + micro_size]
                local = [
                    chunks[i]
                    if cp_size == 1
                    else cp._slice_loss_mask_for_local_cp(lengths[i] + 2, lengths[i], chunks[i], "thd", None)
                    for i in idx
                ]
                params = ([lengths[i] + 2 for i in idx], [lengths[i] for i in idx], [masks[i] for i in idx])
                if mode == "fixed_token_sum":
                    reducer = cp.get_drgrpo_pgloss_reducer(*params, fixed_normalizer=8)
                else:
                    denominators = (
                        [values.new_tensor([2.0, 2.0, 2.5, 2.5][i]) for i in idx]
                        if mode == "group_token_mean"
                        else None
                    )
                    reducer = cp.get_sum_of_sample_mean(*params, mode == "token_mean", denominators=denominators)
                total = total + reducer(torch.cat(local))
    actual = total / (sum(m.sum() for m in masks) if mode == "token_mean" else 4)
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(torch.autograd.grad(actual, values)[0], expected_grad)


def test_group_denominators_survive_dp_transport(monkeypatch):
    args = recipe(monkeypatch, "dapo", "--policy-loss-reduction", "group_token_mean", "--overlong-reward-mode", "none")
    args.balance_data = False
    converted = convert_samples_to_train_data(args, samples(), {}, None, None)
    assert converted["policy_loss_denominators"] == [2.5, 2.5, 4.5, 4.5]
    shards = split_train_data_by_dp_raw(args, converted, dp_size=2)
    assert shards[0]["policy_loss_denominators"] == [2.5, 4.5]
    assert shards[1]["policy_loss_denominators"] == [2.5, 4.5]


def test_raw_filter_precedes_length_penalty_and_keeps_group(monkeypatch):
    args = recipe(monkeypatch, "dapo")
    group = samples((1.0, 1.0))
    group[0].response_length, group[1].response_length = 200000, 250000
    assert not apply_preput_filters(args, apply_reward_nonzero_std_filter, group).keep
    raw, penalty, shaped = _shape_rollout_rewards(args, group)
    assert raw == [1.0, 1.0] and penalty == [0.0, -1.0] and shaped == [1.0, 0.0]
    assert [s.reward for s in group] == [1.0, 1.0]
    group[1].reward = 0.0
    assert apply_preput_filters(args, apply_reward_nonzero_std_filter, group).keep
    assert len(group) == 2 and [s.index for s in group] == [0, 1]


@pytest.mark.parametrize(
    "length,penalty", [(199999, 0.0), (200000, 0.0), (225000, -0.5), (250000, -1.0), (260000, -1.0)]
)
def test_soft_penalty_boundaries(monkeypatch, length, penalty):
    args = recipe(monkeypatch, "dapo")
    sample = samples((1.0,))[0]
    sample.response_length = length
    assert _shape_rollout_rewards(args, [sample]) == ([1.0], [penalty], [1.0 + penalty])


def test_truncation_masks_sibling_fragments_only_when_requested(monkeypatch):
    args = recipe(monkeypatch, "dapo", "--overlong-reward-mode", "mask_truncated")
    data = samples()
    data[1].rollout_id = data[0].rollout_id
    data[1].reward = data[0].reward
    data[1].status = Sample.Status.TRUNCATED
    result = convert_samples_to_train_data(args, data, {}, None, None)
    assert all(not any(mask) for mask in result["loss_masks"][:2])
    assert all(all(mask) for mask in result["loss_masks"][2:])
    assert result["raw_reward"] == result["shaped_reward"]


def test_dynamic_budget_and_acceptance(monkeypatch):
    args = recipe(monkeypatch, "dapo", "--dynamic-sampling-max-groups", "3")
    gatherer = MetricGatherer()
    gatherer.on_filter_result(args, keep=False)
    gatherer.on_filter_result(args, keep=True)
    assert gatherer.collect()["rollout/group_acceptance_rate"] == 0.5
    with pytest.raises(RuntimeError, match="accepted 1/2"):
        gatherer.on_filter_result(args, keep=False)
    gatherer = MetricGatherer()
    for _ in range(5):
        gatherer.on_filter_result(args, keep=False)
        gatherer.on_filter_result(args, keep=True)
        gatherer.on_filter_result(args, keep=True)  # a filled batch resets the budget
