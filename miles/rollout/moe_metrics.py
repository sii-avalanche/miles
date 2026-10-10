"""Expert assignment counts from the rollout batch's existing routing replay.

This measures the served policy, including prompt positions, BEFORE training.
It is not a measurement of the trainer router or of expert compute time.
"""

from collections.abc import Iterable

import numpy as np

from miles.utils.types import Sample


def expert_load_metrics(samples: Iterable[Sample], *, layer: int, num_experts: int) -> dict[str, float]:
    counts = np.zeros(num_experts, dtype=np.int64)
    samples_seen = samples_counted = 0
    for sample in samples:
        samples_seen += 1
        routed = sample.rollout_routed_experts
        if routed is None:
            continue
        if routed.ndim != 3 or routed.shape[1] < layer:
            continue
        samples_counted += 1
        # Only one layer; bound temporary int64 allocations even for 250k-token traces.
        for start in range(0, routed.shape[0], 8192):
            ids = np.asarray(routed[start : start + 8192, layer - 1, :]).reshape(-1)
            if ids.size and (ids.min() < 0 or ids.max() >= num_experts):
                raise ValueError("routing replay contains out-of-range expert ids")
            counts += np.bincount(ids.astype(np.int64, copy=False), minlength=num_experts)
    prefix = f"moe/rollout_layer_{layer}/"
    result = {
        f"{prefix}sample_coverage": samples_counted / samples_seen if samples_seen else 0,
        f"{prefix}assignments": int(counts.sum()),
    }
    mean = float(counts.mean())
    if mean > 0:
        result.update(
            {
                f"{prefix}cv": float(counts.std()) / mean,
                f"{prefix}max_over_mean": int(counts.max()) / mean,
                f"{prefix}cold_experts_fraction": float(np.mean(counts < 0.1 * mean)),
            }
        )
    return result
