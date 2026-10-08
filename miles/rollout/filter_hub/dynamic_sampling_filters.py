from typing_extensions import deprecated

from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.rollout.filter_hub.common_filters import apply_aborted_filter, apply_reward_nonzero_std_filter
from miles.utils.types import Sample

__all__ = ["check_reward_nonzero_std", "check_no_aborted"]


@deprecated("Use miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter", category=None)
def check_reward_nonzero_std(args, samples: list[Sample | list[Sample]], **kwargs):
    return apply_reward_nonzero_std_filter(args, samples, **kwargs)


@deprecated("Use miles.rollout.filter_hub.common_filters.apply_aborted_filter", category=None)
def check_no_aborted(args, samples: list[Sample | list[Sample]], **kwargs):
    return apply_aborted_filter(args, samples, **kwargs)


def _episode_reward(args, sample):
    """An episode that yields several samples (--generate-multi-samples) carries one reward on each."""
    return (sample[0] if isinstance(sample, list) else sample).get_reward_value(args)


def check_passrate(args, samples: list[Sample], **kwargs):
    """Keep groups only when passrate falls between the configured thresholds."""
    rewards = [_episode_reward(args, sample) for sample in samples]
    passrate = sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0
    threshold_low = args.passrate_threshold_low
    threshold_high = args.passrate_threshold_high + 1e-5
    keep = threshold_low < passrate < threshold_high
    print(
        f"[dynamic filter] passrate: {passrate:.3f}, threshold_low: {threshold_low:.3f}, threshold_high: {threshold_high:.3f}, keep: {keep}"
    )
    return DynamicFilterOutput(keep=keep, reason=None if keep else f"passrate_{passrate:.3f}")
