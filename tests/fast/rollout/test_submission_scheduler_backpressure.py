"""Single-sample backfill respects completed groups still waiting for FIFO put."""

from miles.rollout.submission_scheduler import SampleBackfillSubmission


def test_single_sample_completion_keeps_group_slot_until_put_is_accepted():
    scheduler = SampleBackfillSubmission(1)
    scheduler.on_submit([[object()], [object()]])
    scheduler.sample_done_callback()
    assert not scheduler.has_capacity(pending_groups=2, group_budget=2)
    # The worker removes one completed group only after its put succeeds.
    assert scheduler.has_capacity(pending_groups=1, group_budget=2)


def test_multi_sample_groups_keep_existing_sample_level_backfill():
    scheduler = SampleBackfillSubmission(2)
    scheduler.on_submit([[object(), object()], [object(), object()]])
    scheduler.sample_done_callback()
    assert not scheduler.has_capacity(pending_groups=2, group_budget=2)
    scheduler.sample_done_callback()
    assert scheduler.has_capacity(pending_groups=2, group_budget=2)
