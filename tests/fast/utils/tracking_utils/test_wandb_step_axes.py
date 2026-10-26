from types import SimpleNamespace

import pytest

from miles.utils.tracking_utils import wandb_utils


@pytest.mark.parametrize(
    "metric",
    [
        "rollout/episode_response_length/min",
        "rollout/episode_response_length/mean",
        "rollout/episode_response_length/median",
        "rollout/episode_response_length/max",
        "rollout/episode_total_response_length/mean",
        "rollout/length_penalty/mean",
        "rollout/episode_raw_reward",
        "rollout/num_training_samples",
        "rollout/group_acceptance_rate",
        "rollout/group_reward_variance_min",
    ],
)
def test_primary_declares_worker_rollout_axes_before_any_values_are_logged(monkeypatch, metric):
    """Shared-mode workers cannot publish run-level metric definitions."""
    definitions = {}
    monkeypatch.setattr(wandb_utils.wandb, "init", lambda **kwargs: None)
    monkeypatch.setattr(wandb_utils.wandb, "run", SimpleNamespace(id="test-run"))
    monkeypatch.setattr(
        wandb_utils.wandb, "define_metric", lambda name, **kwargs: definitions.update({name: kwargs})
    )
    monkeypatch.setattr(wandb_utils.wandb, "log", lambda *args, **kwargs: pytest.fail("Primary must not log worker values"))
    args = SimpleNamespace(
        env_report=None,
        rank=0,
        use_wandb=True,
        wandb_dir=None,
        wandb_group="test",
        wandb_key=None,
        wandb_mode="online",
        wandb_project="test",
        wandb_random_suffix=False,
        wandb_run_id=None,
        wandb_team=None,
    )
    monkeypatch.setenv("WANDB_MODE", "online")

    wandb_utils.init_wandb_primary(args)

    assert definitions[metric] == {"step_metric": "rollout/step", "step_sync": False}
    assert "rollout/step" in definitions
