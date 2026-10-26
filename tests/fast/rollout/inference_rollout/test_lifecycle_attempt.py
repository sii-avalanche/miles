import asyncio
from types import SimpleNamespace

import pytest

from miles.rollout.base_types import GenerateFnOutput
from miles.rollout.inference_rollout import inference_rollout_common as common
from miles.utils.lifecycle import TrajectoryLifecycle
from miles.utils.types import Sample


class RecordingSink:
    def __init__(self):
        self.events = []

    def attempt_start(self, sample):
        self.events.append(("attempt_start", sample.index))

    def gen_start(self, sample):
        self.events.append(("gen_start", sample.index))

    def attempt_end(self, sample):
        self.events.append(("attempt_end", sample.index))


@pytest.fixture
def lifecycle_sink():
    sink = RecordingSink()
    TrajectoryLifecycle().sink = sink
    yield sink
    TrajectoryLifecycle().sink = None


def make_state(generate_function):
    args = SimpleNamespace(
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        group_rm=True,
        group_oversampling_extra=0,
        sglang_router_policy="round_robin",
    )
    return SimpleNamespace(
        args=args,
        generate_fn_semaphore=asyncio.Semaphore(2),
        aborted=False,
        generate_function=generate_function,
    )


async def test_attempt_ends_once_after_success(lifecycle_sink):
    async def successful_generate(input):
        return GenerateFnOutput(samples=input.sample)

    sample = Sample(index=1)
    result = await common.generate_and_rm(make_state(successful_generate), sample, sampling_params={})

    assert result is sample
    assert lifecycle_sink.events == [
        ("attempt_start", 1),
        ("gen_start", 1),
        ("attempt_end", 1),
    ]


async def test_attempt_ends_once_after_abort(lifecycle_sink):
    async def unexpected_generate(_input):
        pytest.fail("aborted attempts must not generate")

    sample = Sample(index=1)
    state = make_state(unexpected_generate)
    state.aborted = True
    result = await common.generate_and_rm(state, sample, sampling_params={})

    assert result is sample
    assert sample.status == Sample.Status.ABORTED
    assert lifecycle_sink.events == [
        ("attempt_start", 1),
        ("attempt_end", 1),
    ]


async def test_attempt_ends_when_generate_raises(lifecycle_sink):
    async def failing_generate(_input):
        raise RuntimeError("generate failed")

    sample = Sample(index=1)
    with pytest.raises(RuntimeError, match="generate failed"):
        await common.generate_and_rm(make_state(failing_generate), sample, sampling_params={})

    assert lifecycle_sink.events == [
        ("attempt_start", 1),
        ("gen_start", 1),
        ("attempt_end", 1),
    ]


async def test_cancelled_group_sibling_ends_attempt(lifecycle_sink):
    all_started = asyncio.Event()
    never_finishes = asyncio.Event()
    started = 0
    cancelled = []

    async def generate(input):
        nonlocal started
        started += 1
        if started == 2:
            all_started.set()
        try:
            await all_started.wait()
            if input.sample.index == 1:
                raise RuntimeError("sample failed")
            await never_finishes.wait()
        except asyncio.CancelledError:
            cancelled.append(input.sample.index)
            raise

    samples = [Sample(index=1), Sample(index=2)]
    with pytest.raises(RuntimeError, match="sample failed"):
        await common.generate_and_rm_group(make_state(generate), samples, sampling_params={})

    assert cancelled == [2]
    for sample in samples:
        assert [kind for kind, index in lifecycle_sink.events if index == sample.index] == [
            "attempt_start",
            "gen_start",
            "attempt_end",
        ]


async def test_group_oversampling_returns_after_target_usable_candidates(lifecycle_sink):
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0
    cancelled = []

    async def generate(input):
        nonlocal calls
        calls += 1
        if calls == 4:
            started.set()
        await started.wait()
        if input.sample.index == 1:
            input.sample.status = Sample.Status.ABORTED
        elif input.sample.index == -1_000_002:
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.append(input.sample.index)
                raise
        else:
            input.sample.status = Sample.Status.COMPLETED
        return GenerateFnOutput(samples=input.sample)

    state = make_state(generate)
    state.args.group_rm = False
    state.args.group_oversampling_extra = 2
    state.generate_fn_semaphore = asyncio.Semaphore(4)
    samples = [Sample(group_index=0, index=1, reward=1), Sample(group_index=0, index=2, reward=1)]

    result = await common.generate_and_rm_group(state, samples, sampling_params={})

    assert calls == 4
    assert cancelled == [-1_000_002]
    assert len(result) == 2
    assert all(sample.status == Sample.Status.COMPLETED for sample in result)
    assert [sample.index for sample in result] == [1, 2]
    candidate_ids = {1, 2, -1_000_001, -1_000_002}
    assert {index for _, index in lifecycle_sink.events} == candidate_ids
    for index in candidate_ids:
        assert [kind for kind, observed in lifecycle_sink.events if observed == index] == [
            "attempt_start",
            "gen_start",
            "attempt_end",
        ]
