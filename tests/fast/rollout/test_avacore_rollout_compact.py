from __future__ import annotations

import asyncio
import copy
from argparse import Namespace
from types import SimpleNamespace
from unittest import mock

import httpx
import pytest
from ava_core.rewards.core import Reward

from miles.rollout import avacore_rollout
from miles.rollout.inference_rollout import inference_rollout_common as common
from miles.utils.types import Sample


async def test_multi_tip_generation_assigns_one_compact_rollout_id(monkeypatch):
    nodes = [
        mock.Mock(generated=True, segments=(object(),), message=SimpleNamespace(metadata={})),
        mock.Mock(generated=True, segments=(object(),), message=SimpleNamespace(metadata={})),
    ]
    tips = [SimpleNamespace(prefixes=(node,)) for node in nodes]

    async def generate(_row, **_kwargs):
        return tuple(tips)

    async def reward(_trace, _row):
        return Reward(score=1.0)

    adapter = object.__new__(avacore_rollout.AvaCoreRollout)
    adapter.args = Namespace(use_rollout_routing_replay=False)
    adapter.generate_fn = generate
    adapter.reward_fn = reward
    adapter.recording = None
    adapter.group_oversampling = avacore_rollout._GroupOversampling()

    monkeypatch.setattr(avacore_rollout, "iter_trace_nodes", lambda _traces: iter(nodes))
    monkeypatch.setattr(avacore_rollout, "conversation_tips", lambda _traces: iter(tips))
    monkeypatch.setattr(
        avacore_rollout,
        "filled",
        lambda _args, sample, _tip, _reward, *, owned: copy.deepcopy(sample),
    )

    output = await adapter.generate(Sample(index=17, prompt="question"), sampling_params={})

    assert isinstance(output, list)
    assert [sample.rollout_id for sample in output] == [17, 17]


def _single_tip_adapter(generate):
    node = mock.Mock(
        generated=True,
        segments=(object(),),
        message=SimpleNamespace(metadata={"finish_reason": {"type": "stop"}}),
    )
    tip = SimpleNamespace(prefixes=(node,))

    async def reward(_trace, _row):
        return Reward(score=1.0)

    adapter = object.__new__(avacore_rollout.AvaCoreRollout)
    adapter.args = Namespace(use_rollout_routing_replay=False)
    adapter.generate_fn = generate
    adapter.reward_fn = reward
    adapter.recording = None
    adapter.group_oversampling = avacore_rollout._GroupOversampling(
        max_extra_attempts_per_slot=1,
        initial_backoff_seconds=0,
        max_backoff_seconds=0,
    )
    return adapter, node, tip


async def test_retryable_failure_refills_only_the_failed_group_slot(monkeypatch):
    calls = 0

    async def generate(_row, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            error = httpx.HTTPError("remote infrastructure failure")
            error.retryable = True
            raise error
        return (tip,)

    adapter, node, tip = _single_tip_adapter(generate)
    monkeypatch.setattr(avacore_rollout, "iter_trace_nodes", lambda _traces: iter((node,)))
    monkeypatch.setattr(avacore_rollout, "conversation_tips", lambda _traces: iter((tip,)))

    def fill(_args, sample, _tip, reward, **_kwargs):
        sample.reward = reward
        return sample

    monkeypatch.setattr(avacore_rollout, "filled", fill)
    sample = Sample(index=17, prompt="question")

    output = await adapter.generate(sample, sampling_params={})

    assert output is not sample
    assert output.index == sample.index
    assert calls == 2
    assert output.reward == 1.0
    assert sample.reward is None


async def test_non_retryable_failure_does_not_oversample(monkeypatch):
    calls = 0

    async def generate(_row, **_kwargs):
        nonlocal calls
        calls += 1
        error = httpx.HTTPError("semantic remote failure")
        error.retryable = False
        raise error

    adapter, _node, _tip = _single_tip_adapter(generate)
    sample = Sample(index=17, prompt="question")

    output = await adapter.generate(sample, sampling_params={})

    assert output is sample
    assert calls == 1
    assert output.status == Sample.Status.ABORTED


async def test_abort_finish_reason_is_not_retried_as_infrastructure_failure(monkeypatch):
    calls = 0

    async def generate(_row, **_kwargs):
        nonlocal calls
        calls += 1
        return (tip,)

    adapter, node, tip = _single_tip_adapter(generate)
    node.message.metadata = {"finish_reason": {"type": "abort"}}
    monkeypatch.setattr(avacore_rollout, "iter_trace_nodes", lambda _traces: iter((node,)))
    sample = Sample(index=17, prompt="question")

    output = await adapter.generate(sample, sampling_params={})

    assert output is sample
    assert calls == 1
    assert output.status == Sample.Status.ABORTED


async def test_record_uses_candidate_slot_for_stable_trial_identity():
    class FakeRun:
        def __init__(self):
            self.calls = []

        async def create_rollout(self, **kwargs):
            self.calls.append(kwargs)

    run = FakeRun()
    adapter = object.__new__(avacore_rollout.AvaCoreRollout)
    adapter.args = Namespace(n_samples_per_prompt=8)
    adapter.run = asyncio.create_task(asyncio.sleep(0, result=run))
    adapter.failed_writes = 0
    node = SimpleNamespace(
        generated=True,
        segments=(object(),),
        message=SimpleNamespace(metadata={"weight_version": 3}),
    )
    trace = SimpleNamespace(prefixes=(node,))
    cases = [(0, 0), (0, 8), (0, 8), (1, 0), (1, 8)]

    for epoch, slot in cases:
        sample = Sample(
            epoch=epoch,
            index=slot,
            metadata={
                "_index": "prompt",
                "_group_oversampling_candidate_slot": slot,
                "_group_oversampling_candidate_width": 10,
            },
        )
        await adapter.record(SimpleNamespace(), sample, (trace,), Reward(score=1.0), {})

    assert [call["query_id"] for call in run.calls] == ["prompt"] * 5
    assert [call["trial_id"] for call in run.calls] == [0, 8, 8, 10, 18]
    assert [call["metadata"] for call in run.calls] == [{"weight_version": 3}] * 5


def test_generate_state_constructs_avacore_adapter_at_startup(monkeypatch):
    class InvalidAvaCoreRollout:
        def __init__(self, _args, _config):
            raise ValueError("invalid group oversampling config")

    monkeypatch.setattr(common, "load_tokenizer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(common, "load_processor", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(avacore_rollout, "AvaCoreRollout", InvalidAvaCoreRollout)
    args = SimpleNamespace(
        hf_checkpoint="checkpoint",
        chat_template_path=None,
        sglang_server_concurrency=1,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_max_response_len=16,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        rollout_repeat_min_count=0,
        rollout_repeat_min_length=0,
        rollout_repeat_max_length=0,
        custom_generate_function_path="miles.rollout.avacore_rollout.generate",
        avacore_config={"group_oversampling": {"unexpected": 1}},
    )

    with pytest.raises(ValueError, match="invalid group oversampling config"):
        common.GenerateState(args)


@pytest.mark.parametrize(
    ("raw", "exception"),
    [
        ({"max_extra_attempts_per_slot": -1}, ValueError),
        ({"max_extra_attempts_per_slot": True}, ValueError),
        ({"unexpected": 1}, ValueError),
        ({"initial_backoff_seconds": 2, "max_backoff_seconds": 1}, ValueError),
    ],
)
def test_group_oversampling_rejects_invalid_config(raw, exception):
    with pytest.raises(exception):
        avacore_rollout._GroupOversampling.from_config({"group_oversampling": raw})
