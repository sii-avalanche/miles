import asyncio
from argparse import Namespace
from types import SimpleNamespace

import pytest

import miles.rollout.inference_rollout.inference_rollout_train as rollout
from miles.rollout.base_types import RolloutFnTrainInput, RolloutFnTrainOutput
from miles.rollout.continuous_sec import ContinuousSEC, METADATA_KEY
from miles.rollout.curriculum_config import SECConfig, validate_curriculum_args
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn
from miles.rollout.sec_observer import SECBatchObserver
from miles.utils.arguments import resolve_rollout_function_paths
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall


def make_curriculum():
    return ContinuousSEC(
        [{"question_id": "q", "initial_probability": 0.5}], config=SECConfig(), seed=42, standardize=True
    )


def make_group():
    return [Sample(index=i, tokens=[1, 2], metadata={METADATA_KEY: {"question_id": "q"}}) for i in range(16)]


@pytest.mark.parametrize("fully_async", [False, True])
@pytest.mark.parametrize("curriculum_enabled", [False, True])
def test_curriculum_does_not_change_rollout_mode(monkeypatch, fully_async, curriculum_enabled):
    monkeypatch.setenv("MILES_USE_LEGACY_ROLLOUT_V1", "0")
    args = Namespace(
        fully_async=fully_async,
        curriculum_enabled=curriculum_enabled,
        rollout_function_path=None,
        eval_function_path=None,
    )
    standard = "miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn"
    expected = "miles.rollout.fully_async_rollout.FullyAsyncRolloutFn" if fully_async else standard
    assert resolve_rollout_function_paths(args) == (expected, expected)


async def test_observer_preserves_generation_calls_and_updates_sec(monkeypatch):
    calls = []

    async def generate(state, group, **kwargs):
        calls.append((group, kwargs))
        for i, sample in enumerate(group):
            sample.status = Sample.Status.COMPLETED
            sample.reward = float(i % 2)
            sample.weight_versions = [
                WeightVersionsPerCall(spans=[WeightVersionSpan(version="4", abs_start=0, abs_end=1)])
            ]
        return group

    monkeypatch.setattr(rollout, "generate_and_rm_group", generate)
    params = {"temperature": 1.0, "top_p": 0.95, "top_k": 512, "max_new_tokens": 100000}
    baseline = SimpleNamespace(sampling_params=params.copy())
    await asyncio.gather(*rollout.submit_generate_tasks(baseline, [make_group()]))
    curriculum = make_curriculum()
    observer = SECBatchObserver(curriculum, 4, None)
    state = SimpleNamespace(sampling_params=params.copy(), curriculum_observer=observer)
    group = make_group()
    tasks = rollout.submit_generate_tasks(state, [group])
    await asyncio.gather(*tasks)
    observer.close()
    assert calls[0][1] == calls[1][1]
    assert calls[1][1]["sampling_params"] == params
    assert state.sampling_params == params and calls[1][0] is group
    assert len(tasks) == 1 and tasks[0].done()
    assert curriculum.version == 1
    assert not curriculum._inflight and not curriculum._reserved
    # Repeated callback/close cannot apply feedback twice or start new work.
    observer.close()
    await asyncio.sleep(0)
    assert len(calls) == 2 and curriculum.version == 1


async def test_cancelled_original_task_only_marks_feedback_invalid():
    curriculum = make_curriculum()
    observer = SECBatchObserver(curriculum, 4, None)
    gate = asyncio.Event()
    task = asyncio.create_task(gate.wait())
    observer.watch(task, make_group())
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    observer.close()
    assert curriculum.version == 0 and curriculum.metrics()["groups/invalid_total"] == 1
    assert not curriculum._reserved and not curriculum._inflight


def test_minimal_sec_rejects_unsupported_continuous_producer_mode():
    with pytest.raises(ValueError, match="fully_async=false"):
        validate_curriculum_args(Namespace(curriculum_enabled=True, fully_async=True))


async def test_invalid_judge_verdict_does_not_update_sec_or_change_sample_status():
    curriculum = make_curriculum()
    observer = SECBatchObserver(curriculum, 4, None)
    group = make_group()
    for sample in group:
        sample.status = Sample.Status.COMPLETED
        sample.reward = 0.0
        sample.metadata["grader_invalid"] = True

    async def complete():
        return group

    task = asyncio.create_task(complete())
    observer.watch(task, group)
    await task
    observer.close()
    assert curriculum.version == 0
    assert all(sample.status == Sample.Status.COMPLETED and sample.reward == 0 for sample in group)


@pytest.mark.parametrize("enabled", [False, True])
async def test_original_batch_entry_point_is_called_once_with_original_arguments(monkeypatch, enabled):
    calls = []
    output = RolloutFnTrainOutput(samples=[], metrics={"original": 1})
    source = SimpleNamespace(
        curriculum=make_curriculum() if enabled else None,
        get_samples=lambda count: [],
        add_samples=lambda samples: calls.append(samples),
    )
    state = SimpleNamespace(args=Namespace(reward_key=None), sampling_params={"top_p": 0.95})

    async def original(s, rollout_id, get_samples):
        assert s is state and rollout_id == 7 and get_samples is source.get_samples
        assert hasattr(s, "curriculum_observer") is enabled
        calls.append("original_rollout")
        return output, []

    monkeypatch.setattr(rollout, "generate_rollout_async", original)
    fn = object.__new__(InferenceRolloutFn)
    fn.data_source, fn.state = source, state
    result = await fn._call_train(RolloutFnTrainInput(rollout_id=7, weight_version=4))
    assert result is output and calls == ["original_rollout", []]
    assert state.sampling_params == {"top_p": 0.95} and not hasattr(state, "curriculum_observer")
