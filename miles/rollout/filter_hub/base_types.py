import argparse
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

from miles.utils.types import Sample


@dataclass
class FilterOutput:
    keep: bool
    reason: str | None = None


DynamicFilterOutput = FilterOutput


def iter_samples(group: list[Sample | list[Sample]]) -> Iterator[Sample]:
    for sample in group:
        if isinstance(sample, list):
            yield from sample
        else:
            yield sample


def call_dynamic_filter(fn, args, samples: list[Sample | list[Sample]], **kwargs):
    if fn is None:
        return FilterOutput(keep=True)

    output = fn(args, samples, **kwargs)

    # compatibility for legacy version
    if not isinstance(output, FilterOutput):
        output = FilterOutput(keep=output)

    return output


class MetricGatherer:
    def __init__(self):
        self._dynamic_filter_drop_reason_count = defaultdict(lambda: 0)
        self._unfiltered_reward_sum = 0.0
        self._unfiltered_reward_count = 0
        self._groups_seen = self._groups_kept = 0
        self._budget_seen = self._budget_kept = 0

    def on_group_before_dynamic_filter(self, args: argparse.Namespace, group: list) -> None:
        for sample in _iter_group_samples(group):
            if sample.reward is None:
                continue
            if not args.reward_key and isinstance(sample.reward, dict):
                continue
            if (value := sample.get_reward_value(args)) is None:
                continue
            self._unfiltered_reward_sum += float(value)
            self._unfiltered_reward_count += 1

    def on_dynamic_filter_drop(self, reason: str | None):
        if not reason:
            return
        self._dynamic_filter_drop_reason_count[reason] += 1

    def on_filter_result(self, args, *, keep: bool):
        self._groups_seen += 1
        self._groups_kept += int(keep)
        self._budget_seen += 1
        self._budget_kept += int(keep)
        if self._budget_kept >= args.rollout_batch_size:
            self._budget_seen = self._budget_kept = 0
        limit = getattr(args, "dynamic_sampling_max_groups", 0) or 0
        if limit and self._budget_seen >= limit:
            raise RuntimeError(
                f"Dynamic sampling exhausted {limit} groups: accepted "
                f"{self._budget_kept}/{args.rollout_batch_size}; check raw rewards and filter settings"
            )

    def collect(self):
        metrics = {
            f"rollout/dynamic_filter_drop_{reason}": count
            for reason, count in self._dynamic_filter_drop_reason_count.items()
        }
        if self._groups_seen:
            metrics["rollout/group_acceptance_rate"] = self._groups_kept / self._groups_seen
        if self._unfiltered_reward_count:
            metrics["rollout/raw_reward_unfiltered"] = self._unfiltered_reward_sum / self._unfiltered_reward_count
        return metrics


def _iter_group_samples(group: list) -> Iterator[Sample]:
    for item in group:
        if isinstance(item, list):
            yield from item
        else:
            yield item
