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


async def test_exhausted_candidates_never_return_an_unusable_training_group():
    async def generate(input):
        input.sample.status = Sample.Status.COMPLETED if input.sample.index == 1 else Sample.Status.ABORTED
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


async def test_group_failure_observes_every_candidate_task(monkeypatch):
    ready = asyncio.Event()
    calls = 0

    async def generate(_input):
        nonlocal calls
        calls += 1
        if calls == 3:
            ready.set()
        await ready.wait()
        raise RuntimeError("candidate failed")

    observed = set()

    async def settle(tasks):
        observed.update(tasks)
        await asyncio.gather(*tasks, return_exceptions=True)

    monkeypatch.setattr(common, "_cancel_and_settle", settle)
    group = [Sample(group_index=0, index=1), Sample(group_index=0, index=2)]

    with pytest.raises(RuntimeError, match="candidate failed"):
        await common.generate_and_rm_group(make_state(generate), group, sampling_params={})

    assert len(observed) == 3
