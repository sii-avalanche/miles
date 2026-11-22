from types import SimpleNamespace

import pytest
import torch
from torch import nn

from miles.backends.dmodel_utils.step_runner import DModelStepRunner


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, **_kwargs):
        return torch.ones((1, 2, 3)) * self.weight


def test_forward_only_temporarily_uses_eval_and_restores_train(monkeypatch):
    model = _Model()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    runner = DModelStepRunner(model, optimizer, None, object(), max_grad_norm=1.0)
    states = []

    monkeypatch.setattr(
        runner,
        "_forward",
        lambda _batch: states.append(model.training) or SimpleNamespace(),
    )
    result = runner.forward_only_step([{}], lambda _scores, _batch: "ok")

    assert result == ["ok"]
    assert states == [False]
    assert model.training is True


def test_forward_backward_honors_native_microbatch_sync(monkeypatch):
    from dmodel.training import rl

    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    runner = DModelStepRunner(model, optimizer, None, object(), max_grad_norm=1.0, sync_every_microbatch=True)
    monkeypatch.setattr(runner, "_forward", lambda _batch: model.weight)
    calls = []
    original = rl.prepare_gradient_accumulation

    def record(model, index, count, *, sync_every_microbatch=False):
        calls.append((index, count, sync_every_microbatch))
        return original(model, index, count, sync_every_microbatch=sync_every_microbatch)

    monkeypatch.setattr(rl, "prepare_gradient_accumulation", record)
    runner.forward_backward_step([{}, {}], lambda score, _batch: (score.square(), {"loss": score.detach()}))
    assert calls == [(0, 2, True), (1, 2, True)]


@pytest.mark.parametrize("norm", [float("nan"), float("inf")])
def test_nonfinite_norm_never_steps_optimizer_or_scheduler(monkeypatch, norm):
    from dmodel.utils import grad_clip

    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    runner = DModelStepRunner(model, optimizer, scheduler, object(), max_grad_norm=1.0)
    monkeypatch.setattr(grad_clip, "clip_grad_norm_mixed_", lambda *_: torch.tensor(norm))
    calls = []
    monkeypatch.setattr(optimizer, "step", lambda: calls.append("optimizer"))
    monkeypatch.setattr(scheduler, "step", lambda: calls.append("scheduler"))
    with pytest.raises(FloatingPointError, match="non-finite"):
        runner.apply_step()
    assert calls == []


def test_remote_nonfinite_rank_stops_local_update(monkeypatch):
    from dmodel.utils import grad_clip

    from miles.backends.dmodel_utils import step_runner

    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    runner = DModelStepRunner(model, optimizer, None, object(), max_grad_norm=1.0)
    group = object()
    monkeypatch.setattr(grad_clip, "clip_grad_norm_mixed_", lambda *_: torch.tensor(0.5))
    monkeypatch.setattr(step_runner, "get_gloo_group", lambda: group)
    monkeypatch.setattr(step_runner.dist, "is_initialized", lambda: True)

    def reduce(tensor, *, op, group):
        tensor.zero_()

    monkeypatch.setattr(step_runner.dist, "all_reduce", reduce)
    calls = []
    monkeypatch.setattr(optimizer, "step", lambda: calls.append(True))
    with pytest.raises(FloatingPointError):
        runner.apply_step()
    assert calls == []
