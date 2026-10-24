from __future__ import annotations

import copy
from argparse import Namespace
from types import SimpleNamespace
from unittest import mock

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
