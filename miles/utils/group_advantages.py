"""The scalar, per-episode GRPO advantages shared by training and SEC feedback."""

import torch


def normalize_group_rewards(
    rewards: torch.Tensor, *, standardize: bool, quantile: float | None = None
) -> torch.Tensor:
    baseline = rewards.mean() if quantile is None else rewards.quantile(quantile)
    advantages = rewards - baseline
    if standardize and len(rewards) > 1:
        std = rewards.std()
        if std > 0:
            advantages = advantages / (std + 1e-6)
    return advantages
