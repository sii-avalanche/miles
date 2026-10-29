"""Preset wiring and the two additions to native Miles: fixed C and soft length penalties."""

import argparse
import sys
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.data import context_parallel as cp
from miles.ray.rollout.train_data_conversion import _penalize_overlong_rewards, _post_process_rewards
from miles.rollout.filter_hub.common_filters import apply_reward_nonzero_std_filter
from miles.utils.arguments import get_miles_extra_args_provider, validate_rl_algorithm_args
from miles.utils.types import Sample


def recipe(monkeypatch, name=None, *extra, **overrides):
    argv = ([] if name is None else ["--rl-algorithm", name]) + list(extra)
    monkeypatch.setattr(sys, "argv", ["test", *argv])
    parser = get_miles_extra_args_provider()(argparse.ArgumentParser())
    for action in parser._actions:
        action.required = False
    args = parser.parse_args(argv)
    args.rollout_max_response_len = 10
    vars(args).update(overrides)
    validate_rl_algorithm_args(args)
    return args


@pytest.mark.parametrize(
    "name,std,high,token",
    [
        ("grpo_original", True, 0.2, False),
        ("dapo", True, 0.28, True),
        ("dr_grpo", False, 0.2, False),
    ],
)
def test_presets_reuse_native_options(monkeypatch, name, std, high, token):
    args = recipe(monkeypatch, name)
    assert args.grpo_std_normalization is std
    assert args.eps_clip == 0.2 and args.eps_clip_high == high
    assert args.calculate_per_token_loss is token
    assert args.rewards_normalization and args.advantage_estimator == "grpo"
    assert bool(args.dynamic_sampling_filter_path) is token
    assert (args.overlong_reward_mode == "soft_penalty") is token
    if token:
        assert args.overlong_buffer_length == 2


def test_opt_in_and_explicit_overrides(monkeypatch):
    args = recipe(monkeypatch)
    assert args.rl_algorithm is None and args.eps_clip_high is None
    assert not args.calculate_per_token_loss and args.overlong_reward_mode == "none"
    args = recipe(
        monkeypatch,
        "dapo",
        "--eps-clip-high=0.3",
        "--disable-grpo-std-normalization",
        "--dynamic-sampling-filter-path",
        "",
        "--overlong-reward-mode",
        "none",
    )
    assert args.eps_clip_high == 0.3 and not args.grpo_std_normalization
    assert not args.dynamic_sampling_filter_path and args.overlong_reward_mode == "none"
    args = recipe(monkeypatch, "dr_grpo", "--calculate-per-token-loss", rollout_max_response_len=None)
    assert args.calculate_per_token_loss  # Explicit token reduction bypasses the fixed-C preset.
    args = recipe(
        monkeypatch,
        "dr_grpo",
        "--custom-pg-loss-reducer-function-path",
        "custom.reducer",
        rollout_max_response_len=None,
    )
    assert args.custom_pg_loss_reducer_function_path == "custom.reducer"


@pytest.mark.parametrize(
    "overrides",
    [
        {"train_backend": "fsdp"},
        {"loss_type": "score_centering"},
        {"multi_lora": True},
        {"rollout_max_response_len": None},
        {"overlong_buffer_length": 0},
        {"overlong_buffer_length": 11},
        {"overlong_penalty_strength": -1},
        {"overlong_penalty_strength": float("nan")},
        {"custom_reward_post_process_path": "custom.reward"},
        {"custom_convert_samples_to_train_data_path": "custom.data"},
    ],
)
def test_invalid_presets_fail_early(monkeypatch, overrides):
    with pytest.raises(ValueError):
        recipe(monkeypatch, "dapo", **overrides)


@pytest.mark.parametrize("length,expected", [(7, 1.0), (8, 1.0), (9, 0.5), (10, 0.0), (11, 0.0)])
def test_soft_penalty_boundaries(monkeypatch, length, expected):
    args = recipe(monkeypatch, "dapo")
    sample = Sample(index=1, response_length=length, reward=1.0)
    assert _penalize_overlong_rewards(args, [sample], [1.0]) == [expected]
    assert sample.reward == 1.0 and sample.loss_mask is None


def test_penalty_preserves_raw_filtering_and_normalizes_whole_rollouts(monkeypatch):
    args = recipe(monkeypatch, "dapo", "--disable-grpo-std-normalization")
    samples = [
        Sample(index=0, group_index=0, rollout_id=10, reward=1.0, response_length=20, loss_mask=[1] * 4 + [0] * 16),
        Sample(index=1, group_index=0, rollout_id=10, reward=1.0, response_length=9, loss_mask=[0] * 4 + [1] * 5),
        Sample(index=2, group_index=0, rollout_id=11, reward=1.0, response_length=7),
    ]
    assert not apply_reward_nonzero_std_filter(args, samples).keep
    raw, normalized = _post_process_rewards(args, samples, None)
    assert raw == [1.0, 1.0, 1.0]
    assert normalized == pytest.approx([-0.25, -0.25, 0.25])
    assert [sample.reward for sample in samples] == raw
    args.rewards_normalization = False
    assert _post_process_rewards(args, samples, None)[1] == [0.5, 0.5, 1.0]
    args.overlong_reward_mode = "none"
    assert _post_process_rewards(args, samples, None) == (raw, raw)


@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("divisor", [10, 20])
def test_native_drgrpo_reducer_uses_explicit_divisor_with_cp(monkeypatch, cp_size, divisor):
    monkeypatch.setenv("DIVISOR", "ignored")
    lengths = [2, 3]
    masks = [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 0.0, 1.0])]
    values = torch.arange(1.0, 6.0, requires_grad=True)
    total = values.sum() * 0
    for rank in range(cp_size):
        monkeypatch.setattr(
            cp, "get_parallel_state", lambda rank=rank: SimpleNamespace(cp=SimpleNamespace(size=cp_size, rank=rank))
        )
        local = [
            part if cp_size == 1 else cp._slice_loss_mask_for_local_cp(length + 2, length, part, "thd", None)
            for length, part in zip(lengths, values.split(lengths), strict=True)
        ]
        reducer = cp.get_drgrpo_pgloss_reducer([4, 5], lengths, masks, divisor=divisor)
        total = total + reducer(torch.cat(local)) / 2
    expected = (values * torch.cat(masks)).sum() / (2 * divisor)
    torch.testing.assert_close(total, expected)
    torch.testing.assert_close(
        torch.autograd.grad(total, values, retain_graph=True)[0], torch.autograd.grad(expected, values)[0]
    )
