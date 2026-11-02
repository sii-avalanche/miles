import asyncio
from types import SimpleNamespace

import pytest

from miles.rollout.base_types import GenerateFnOutput
from miles.rollout.inference_rollout import inference_rollout_common as common
from miles.utils.types import Sample


def make_state(generate):
    args = SimpleNamespace(
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        group_rm=False,
        group_oversampling_extra=1,
        sglang_router_policy="round_robin",
        sglang_enable_deterministic_inference=False,
    )
    return SimpleNamespace(
        args=args,
        generate_fn_semaphore=asyncio.Semaphore(8),
        aborted=False,
        generate_function=generate,
    )


@pytest.mark.parametrize("unusable", [Sample.Status.ABORTED, Sample.Status.PENDING])
async def test_exhausted_candidates_never_return_an_unusable_training_group(unusable):
    async def generate(input):
        input.sample.status = Sample.Status.COMPLETED if input.sample.index == 1 else unusable
        return GenerateFnOutput(samples=input.sample)

    group = [
        Sample(group_index=0, index=1, reward=1),
        Sample(group_index=0, index=2, reward=1),
    ]

    with pytest.raises(RuntimeError, match="only 1/2 usable"):
        await common.generate_and_rm_group(make_state(generate), group, sampling_params={})


def test_identity_restoration_is_stable_when_candidates_alias_originals():
    original = [Sample(group_index=0, index=index) for index in range(3)]
    selected = [original[2], original[0], original[1]]

    common._restore_group_identities(selected, original)

    assert [sample.index for sample in selected] == [0, 1, 2]


async def test_group_failure_observes_every_candidate_exception(caplog):
    ready = asyncio.Event()
    calls = 0

    async def generate(_input):
        nonlocal calls
        calls += 1
        if calls == 3:
            ready.set()
        await ready.wait()
        raise RuntimeError("candidate failed")

    group = [Sample(group_index=0, index=1), Sample(group_index=0, index=2)]

    with pytest.raises(common.GroupOversamplingExhaustedError, match="3 candidate errors"):
        await common.generate_and_rm_group(make_state(generate), group, sampling_params={})

    assert calls == 3
    assert sum("Oversampling candidate=" in record.message for record in caplog.records) == 3


async def test_candidate_exception_is_absorbed_when_enough_candidates_succeed():
    calls = 0

    async def generate(input):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("one sandbox failed")
        input.sample.status = Sample.Status.COMPLETED
        input.sample.reward = 1
        return GenerateFnOutput(samples=input.sample)

    state = make_state(generate)
    state.args.group_oversampling_extra = 2
    group = [Sample(group_index=0, index=index) for index in range(4)]

    result = await common.generate_and_rm_group(state, group, sampling_params={})

    assert len(result) == 4
    assert all(sample.status == Sample.Status.COMPLETED for sample in result)
    assert calls == 6


async def test_parent_cancellation_waits_for_surplus_cleanup_to_finish():
    child_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def child() -> None:
        child_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup_started.set()
            await release_cleanup.wait()
            cleanup_finished.set()
            raise

    child_task = asyncio.create_task(child())
    await child_started.wait()
    settlement = asyncio.create_task(common._cancel_and_settle({child_task}))
    await cleanup_started.wait()

    settlement.cancel()
    await asyncio.sleep(0)
    assert not settlement.done()
    assert not cleanup_finished.is_set()

    release_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await settlement
    assert cleanup_finished.is_set()


async def test_partial_rollout_abort_returns_original_group_when_extras_are_unusable():
    all_started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def generate(input):
        nonlocal calls
        calls += 1
        if calls == 3:
            all_started.set()
        await all_started.wait()
        await release.wait()
        input.sample.status = Sample.Status.ABORTED
        input.sample.response = "partial"
        return GenerateFnOutput(samples=input.sample)

    state = make_state(generate)
    state.args.partial_rollout = True
    group = [Sample(group_index=0, index=1), Sample(group_index=0, index=2)]

    result_task = asyncio.create_task(common.generate_and_rm_group(state, group, sampling_params={}))
    await all_started.wait()
    state.aborted = True
    release.set()
    result = await result_task

    assert calls == 3
    assert [sample.index for sample in result] == [1, 2]
    assert all(sample.status == Sample.Status.ABORTED for sample in result)
    assert all(sample.response == "partial" for sample in result)
