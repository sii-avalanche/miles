import copy
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.dmodel_utils import checkpoint
from miles.backends.megatron_utils.megatron_config import resolve_args_checkpoint_load


def _args(tmp_path, **overrides):
    values = dict(
        train_backend="dmodel",
        megatron_to_hf_mode="core",
        load=None,
        ref_load=str(tmp_path / "reference"),
        hf_checkpoint=str(tmp_path / "hf"),
        ref_ckpt_step=7,
        ckpt_step=None,
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        start_rollout_id=9,
        num_rollout=10,
        save=str(tmp_path / "save"),
        no_save_optim=False,
        no_save_rng=False,
    )
    return Namespace(**(values | overrides))


def _actor(args):
    from dmodel.training.optimizer.combined import CombinedOptimizer

    model = torch.nn.Linear(3, 2)
    optimizer = CombinedOptimizer(
        torch.optim.Muon([model.weight], lr=0.02),
        torch.optim.AdamW([model.bias], lr=0.01),
    )
    return SimpleNamespace(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5),
        global_step=0,
        micro_step=0,
    )


def _step(actor):
    actor.optimizer.zero_grad(set_to_none=True)
    actor.model(torch.ones(2, 3)).square().mean().backward()
    actor.optimizer.step()
    actor.lr_scheduler.step()
    actor.global_step += 1
    actor.micro_step += 1


@pytest.mark.parametrize("mode", ["core", "bridge"])
@pytest.mark.parametrize("start", [None, 0, 9])
def test_fresh_dmodel_preserves_explicit_start_without_fallback(tmp_path, mode, start):
    args = _args(tmp_path, megatron_to_hf_mode=mode, start_rollout_id=start)
    expected = vars(args).copy() | {"requested_load": None, "start_rollout_id": 0 if start is None else start}

    resolve_args_checkpoint_load(args)

    assert vars(args) == expected


@pytest.mark.parametrize("mode", ["core", "bridge"])
@pytest.mark.parametrize("opt_out", [False, True])
def test_dmodel_load_preserves_path_and_flags_for_native_validation(tmp_path, mode, opt_out):
    args = _args(
        tmp_path,
        megatron_to_hf_mode=mode,
        load=str(tmp_path / "missing"),
        ckpt_step=3,
        no_load_optim=opt_out,
        no_load_rng=opt_out,
        finetune=opt_out,
    )
    expected = vars(args).copy() | {"requested_load": args.load, "start_rollout_id": None}

    resolve_args_checkpoint_load(args)

    assert vars(args) == expected
    with pytest.raises(FileNotFoundError):
        checkpoint.load(SimpleNamespace(args=args))
    assert args.start_rollout_id is None


@pytest.mark.parametrize("backend", ["megatron", "fsdp", "torchtitan"])
@pytest.mark.parametrize("mode", ["core", "bridge"])
def test_other_backends_keep_existing_fresh_fallback(tmp_path, backend, mode):
    args = _args(tmp_path, train_backend=backend, megatron_to_hf_mode=mode)
    expected = vars(args).copy() | {"requested_load": None, "load": args.ref_load, "start_rollout_id": 0}
    if mode == "core":
        expected.update(no_load_optim=True, no_load_rng=True, finetune=True, ckpt_step=7)

    resolve_args_checkpoint_load(args)

    assert vars(args) == expected


@pytest.mark.parametrize("backend", ["megatron", "fsdp", "torchtitan"])
@pytest.mark.parametrize("mode", ["core", "bridge"])
def test_other_backends_keep_existing_resume_arguments(tmp_path, backend, mode):
    load = tmp_path / "existing"
    load.mkdir()
    (load / checkpoint.CHECKPOINT_TRACKER_FILENAME).write_text("10")
    args = _args(tmp_path, train_backend=backend, megatron_to_hf_mode=mode, load=str(load))
    expected = vars(args).copy() | {"requested_load": args.load}

    resolve_args_checkpoint_load(args)

    assert vars(args) == expected


@pytest.mark.parametrize("start,expected_iteration", [(None, 1), (9, 10)])
def test_fresh_dmodel_native_checkpoint_uses_resolved_cursor(tmp_path, start, expected_iteration):
    args = _args(tmp_path, start_rollout_id=start)
    resolve_args_checkpoint_load(args)
    actor = _actor(args)
    checkpoint.finalize_load(actor, checkpoint.load(actor))
    _step(actor)

    checkpoint.save(actor, args.start_rollout_id)

    assert actor.global_step == 1
    assert (tmp_path / "save" / f"iter_{expected_iteration:07d}" / "meta.json").is_file()
    assert (tmp_path / "save" / checkpoint.CHECKPOINT_TRACKER_FILENAME).read_text() == str(expected_iteration)


@pytest.mark.parametrize("start", [None, 9])
def test_native_dmodel_resume_owns_cursor_after_successful_restore(tmp_path, start):
    saved = _actor(_args(tmp_path))
    _step(saved)
    checkpoint.save(saved, 4)
    args = _args(tmp_path, load=saved.args.save, start_rollout_id=start)
    resolve_args_checkpoint_load(args)
    # The creator keeps its own args while the actor restores a serialized copy.
    resumed = _actor(copy.deepcopy(args))

    payload = checkpoint.load(resumed)
    assert resumed.args.start_rollout_id is None
    assert resumed.global_step == 0
    checkpoint.finalize_load(resumed, payload)

    assert args.start_rollout_id is None  # The shared creator selects the restored cursor.
    assert resumed.args.start_rollout_id == 5
    assert resumed.global_step == resumed.micro_step == 1
    assert resumed.lr_scheduler.state_dict() == saved.lr_scheduler.state_dict()
    _step(saved)
    _step(resumed)
    for actual, expected in zip(resumed.model.parameters(), saved.model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
