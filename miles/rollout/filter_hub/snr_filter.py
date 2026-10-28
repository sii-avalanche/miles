import itertools
import logging
import statistics

__all__ = ["snr_aware_filter", "select_high_variance_nucleus", "group_reward_variance", "variance_metrics"]

logger = logging.getLogger(__name__)


def _iter_samples(group):
    if not isinstance(group, list):
        yield group
        return
    for s in group:
        if isinstance(s, list):
            yield from _iter_samples(s)
        else:
            yield s


def group_reward_variance(args, group: list) -> float:
    """Sample variance (1/(G-1)) of the rewards within one prompt group."""
    samples = list(_iter_samples(group))
    if len(samples) < 2:
        return 0.0
    rewards = [s.get_reward_value(args) for s in samples]
    return statistics.variance(rewards)


def select_high_variance_nucleus(variances: list[float], keep_ratio: float) -> list[int]:
    """Indices of the smallest set of highest-variance groups whose cumulative
    variance reaches ``keep_ratio`` of the total (top-p / nucleus selection).

    Groups are ranked by descending reward variance; the prefix is kept up to and
    including the one that first reaches ``keep_ratio * sum(variances)``. The
    low-signal tail, including every zero-variance group, is dropped. When no
    group carries variance the ranking is undefined, so all are kept rather than
    emptying the batch.
    """
    order = sorted(range(len(variances)), key=lambda i: variances[i], reverse=True)
    total = sum(variances)
    if total <= 0.0:
        return order
    threshold = keep_ratio * total
    cumulative = itertools.accumulate(variances[i] for i in order)
    k = next((n for n, c in enumerate(cumulative, start=1) if c >= threshold), len(order))
    return order[:k]


def variance_metrics(variances: list[float]) -> dict[str, float]:
    """Reward-variance stats across the rollout's prompt groups."""
    return {
        "rollout/group_reward_variance_mean": statistics.mean(variances) if variances else 0.0,
        "rollout/group_reward_variance_max": max(variances, default=0.0),
        "rollout/group_reward_variance_min": min(variances, default=0.0),
    }


def snr_aware_filter(args, data: list, variances: list[float]) -> tuple[list, dict[str, float]]:
    """Drop low-reward-variance prompt groups (RAGEN-2 SNR-Aware Filtering).

    ``data`` is a list of prompt groups (each a list of samples); ``variances`` is the
    matching per-group reward variance. Returns the kept groups in their original
    order, plus kept-count metrics.
    """
    kept_indices = set(select_high_variance_nucleus(variances, args.snr_filter_keep_ratio))
    kept = [group for i, group in enumerate(data) if i in kept_indices]
    metrics = {
        "rollout/snr_kept_ratio": len(kept) / len(data) if data else 0.0,
    }
    logger.info(
        f"SNR-aware filter (keep_ratio={args.snr_filter_keep_ratio}): "
        f"kept {len(kept)}/{len(data)} groups by reward variance"
    )
    return kept, metrics
