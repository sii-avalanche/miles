"""Paces prompt-group submission for the rollout drivers (--rollout-submission-granularity).

The driver owns the pending group tasks and passes their count in as ``pending_groups``;
the scheduler only accounts samples in flight.
"""

import asyncio
import logging
from argparse import Namespace

from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def candidate_group_size(n_samples_per_prompt: int, group_oversampling_extra: int = 0) -> int:
    return n_samples_per_prompt + group_oversampling_extra


class GroupLevelSubmission:
    """A submission slot frees only when the whole group task returns."""

    sample_done_callback = None

    def available_groups(self, *, pending_groups: int, group_budget: int) -> int:
        return max(0, group_budget - pending_groups)

    def has_capacity(self, *, pending_groups: int, group_budget: int) -> bool:
        return self.available_groups(pending_groups=pending_groups, group_budget=group_budget) > 0

    def on_submit(self, groups: list[list[Sample]]) -> None:
        pass

    async def wait_for_progress(self, pendings: set[asyncio.Task]) -> tuple[set[asyncio.Task], set[asyncio.Task]]:
        return await asyncio.wait(pendings, return_when=asyncio.FIRST_COMPLETED)


class SampleBackfillSubmission:
    """Each finished sample frees its own slot; a replacement group fits once ``group_size`` samples complete."""

    def __init__(self, group_size: int):
        self.group_size = group_size
        self.samples_in_flight = 0
        self._sample_done = asyncio.Event()

    def sample_done_callback(self) -> None:
        self.samples_in_flight -= 1
        self._sample_done.set()

    def available_groups(self, *, pending_groups: int, group_budget: int) -> int:
        if pending_groups == 0 and self.samples_in_flight:
            # a group returned without spawning its sample tasks, or a stub skipped the callback
            logger.warning(f"samples_in_flight={self.samples_in_flight} with no pending groups; resetting to 0")
            self.samples_in_flight = 0
        available = max(0, (group_budget * self.group_size - self.samples_in_flight) // self.group_size)
        if available == 0:
            self._sample_done.clear()
        return available

    def has_capacity(self, *, pending_groups: int, group_budget: int) -> bool:
        return self.available_groups(pending_groups=pending_groups, group_budget=group_budget) > 0

    def on_submit(self, groups: list[list[Sample]]) -> None:
        self.samples_in_flight += len(groups) * self.group_size

    async def wait_for_progress(self, pendings: set[asyncio.Task]) -> tuple[set[asyncio.Task], set[asyncio.Task]]:
        waiter = asyncio.create_task(self._sample_done.wait())
        try:
            done, pending = await asyncio.wait(pendings | {waiter}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiter.cancel()
        return done - {waiter}, pending - {waiter}


def make_submission_scheduler(args: Namespace, *, default: str) -> GroupLevelSubmission | SampleBackfillSubmission:
    extra = getattr(args, "group_oversampling_extra", 0)
    granularity = args.rollout_submission_granularity or ("sample" if extra else default)
    assert not extra or granularity == "sample", (
        "--group-oversampling-extra requires --rollout-submission-granularity sample"
    )
    if granularity == "group":
        return GroupLevelSubmission()
    assert granularity == "sample", f"unknown submission granularity: {granularity}"
    return SampleBackfillSubmission(candidate_group_size(args.n_samples_per_prompt, extra))
