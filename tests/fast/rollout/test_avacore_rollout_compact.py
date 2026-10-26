from __future__ import annotations

import copy
from argparse import Namespace
from types import SimpleNamespace
from unittest import mock

import httpx
import pytest
from ava_core.rewards.core import Reward

from miles.rollout import avacore_rollout
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

    assert output is sample
    assert calls == 2
    assert output.reward == 1.0


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
