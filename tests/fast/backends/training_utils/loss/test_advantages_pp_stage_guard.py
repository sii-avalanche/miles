import pytest
import torch

from miles.backends.training_utils.loss.objective import compute_advantages_and_returns

from .loss_test_utils import make_args, make_parallel_state


def _ppo_args(use_rollout_logprobs: bool):
    return make_args(
        advantage_estimator="ppo",
        use_rollout_logprobs=use_rollout_logprobs,
        kl_coef=0.0,
        lambd=1.0,
    )


def _rollout_data(log_probs_key: str | None, with_values: bool) -> dict:
    response_length = 4
    data = {
        "ref_log_probs": [torch.zeros(response_length)],
        "rewards": [1.0],
        "response_lengths": [response_length],
        "loss_masks": [torch.ones(response_length)],
        "total_lengths": [response_length + 2],
    }
    if log_probs_key is not None:
        data[log_probs_key] = [torch.zeros(response_length)]
    if with_values:
        data["values"] = [torch.zeros(response_length)]
    return data


def test_intermediate_pp_stage_returns_early_with_rollout_log_probs() -> None:
    """Under --use-rollout-logprobs an intermediate stage has log-probs but no
    values; deciding the stage from the tensors runs the PPO estimator there
    and dereferences `values`."""
    make_parallel_state(is_pp_last_stage=False)
    rollout_data = _rollout_data("rollout_log_probs", with_values=False)

    compute_advantages_and_returns(_ppo_args(use_rollout_logprobs=True), rollout_data)

    assert "advantages" not in rollout_data
    assert "returns" not in rollout_data


def test_intermediate_pp_stage_returns_early_without_any_log_probs() -> None:
    make_parallel_state(is_pp_last_stage=False)
    rollout_data = _rollout_data(None, with_values=False)

    compute_advantages_and_returns(_ppo_args(use_rollout_logprobs=False), rollout_data)

    assert "advantages" not in rollout_data


def test_last_pp_stage_computes_advantages_with_rollout_log_probs() -> None:
    make_parallel_state(is_pp_last_stage=True)
    rollout_data = _rollout_data("rollout_log_probs", with_values=True)

    compute_advantages_and_returns(_ppo_args(use_rollout_logprobs=True), rollout_data)

    assert len(rollout_data["advantages"]) == 1
    assert len(rollout_data["returns"]) == 1
    assert rollout_data["advantages"][0].shape == (4,)


@pytest.mark.parametrize("grpo_advantage", [0.8, -0.8, 0.0])
def test_grpo_turn_weights_scale_outcome_advantage(grpo_advantage):
    make_parallel_state()
    data = _rollout_data("rollout_log_probs", with_values=False)
    data["rewards"] = [grpo_advantage]
    data["turn_advantage_weights"] = [[0.65, 0.65, 0.5, 0.3]]
    args = make_args(advantage_estimator="grpo", use_rollout_logprobs=True, kl_coef=0.0, normalize_advantages=False)
    compute_advantages_and_returns(args, data)
    expected = torch.tensor([0.52, 0.52, 0.40, 0.24]) * (grpo_advantage / 0.8)
    torch.testing.assert_close(data["advantages"][0], expected)
    torch.testing.assert_close(data["returns"][0], torch.full_like(expected, grpo_advantage))
