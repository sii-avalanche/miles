"""Explicit global-token / slowest-rank throughput, independent of FLOP models."""

import math

import torch.distributed as dist


def summarize_rank_updates(records: list[dict]) -> dict[str, float | int]:
    if not records:
        raise ValueError("training metrics require at least one rank")
    seconds = [record["seconds"] for record in records]
    if any(isinstance(value, bool) or not math.isfinite(value) or value <= 0 for value in seconds):
        raise ValueError("training rank times must be finite and positive")
    count = len(records)
    result = {
        "perf/training_world_size": count,
        "perf/actor_train_max_rank_time": max(seconds),
        "perf/actor_train_min_rank_time": min(seconds),
        "perf/actor_train_mean_rank_time": sum(seconds) / count,
    }
    counts = records[0]["token_counts"]
    if any(record["token_counts"] != counts for record in records):
        raise ValueError("batch-global training token counts differ between ranks")
    if counts is None:
        # Do not promote legacy seq_lens (possibly containing dummy samples)
        # to a verified real-token numerator.
        return result
    for key in ("model_input_tokens", "supervised_tokens", "dummy_tokens"):
        value = counts[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid training token count: {key}={value!r}")
        result[f"perf/actor_train_{key}"] = value
    if counts["supervised_tokens"] > counts["model_input_tokens"]:
        raise ValueError("supervised tokens exceed model input tokens")
    result["perf/actor_train_tgs_per_gpu"] = counts["model_input_tokens"] / (count * max(seconds))
    return result


def collect_update_metrics(seconds: float, token_counts: dict | None, *, group) -> dict:
    """All trainer ranks join once, outside their measured actor interval."""
    records = [None] * dist.get_world_size(group)
    dist.all_gather_object(records, {"seconds": seconds, "token_counts": token_counts}, group=group)
    return summarize_rank_updates(records)
