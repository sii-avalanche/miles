import copy
import json
import random
import shutil
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.checkpoint.api import CheckpointException

from miles.backends.dmodel_utils import checkpoint
from miles.utils.distributed_utils import init_gloo_group


def _actor(root, *, combined=True, **overrides):
    model = torch.nn.Linear(3, 2)
    if combined:
        from dmodel.training.optimizer.combined import CombinedOptimizer

        optimizer = CombinedOptimizer(
            torch.optim.Muon([model.weight], lr=0.02),
            torch.optim.AdamW([model.bias], lr=0.01),
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    args = dict(
        save=str(root), load=None, ckpt_step=None, start_rollout_id=0,
        no_save_optim=False, no_load_optim=False, no_save_rng=False, no_load_rng=False,
    )
    args.update(overrides)
    return SimpleNamespace(
        args=SimpleNamespace(**args), model=model, optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5),
        global_step=0, micro_step=0,
    )


def _step(actor):
    actor.optimizer.zero_grad()
    actor.model(torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10).square().mean().backward()
    actor.optimizer.step()
    actor.lr_scheduler.step()
    actor.global_step += 1
    actor.micro_step += 1


def _assert_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_equal(left, right)
    else:
        assert actual == expected


def test_combined_optimizer_real_dcp_roundtrip_and_next_update(tmp_path):
    actor = _actor(tmp_path)
    _step(actor)
    optimizer_before = copy.deepcopy(actor.optimizer.state_dict())
    checkpoint.save(actor, 0)
    _assert_equal(actor.optimizer.state_dict(), optimizer_before)

    resumed = _actor(tmp_path, load=str(tmp_path))
    payload = checkpoint.load(resumed)
    assert resumed.global_step == resumed.micro_step == resumed.args.start_rollout_id == 0
    checkpoint.finalize_load(resumed, payload)
    assert resumed.global_step == resumed.micro_step == resumed.args.start_rollout_id == 1
    _assert_equal(resumed.model.state_dict(), actor.model.state_dict())
    _assert_equal(resumed.optimizer.state_dict(), actor.optimizer.state_dict())
    _assert_equal(resumed.lr_scheduler.state_dict(), actor.lr_scheduler.state_dict())
    for leaf in resumed.optimizer.inner_optimizers().values():
        assert all(value.device.type == "cpu" for state in leaf.state.values() for value in state.values())
        assert all(any(group is outer for outer in resumed.optimizer.param_groups) for group in leaf.param_groups)
    _step(actor)
    _step(resumed)
    _assert_equal(resumed.model.state_dict(), actor.model.state_dict())
    _assert_equal(resumed.optimizer.state_dict(), actor.optimizer.state_dict())


def test_fresh_optimizer_serialization_does_not_run_step(tmp_path, monkeypatch):
    actor = _actor(tmp_path)
    before = copy.deepcopy(actor.model.state_dict())

    def forbidden_step(*args, **kwargs):
        pytest.fail("checkpoint serialization must not execute optimizer.step")

    for leaf in actor.optimizer.inner_optimizers().values():
        monkeypatch.setattr(leaf, "step", forbidden_step)
    checkpoint.save(actor, 0)
    _assert_equal(actor.model.state_dict(), before)
    resumed = _actor(tmp_path, load=str(tmp_path))
    checkpoint.finalize_load(resumed, checkpoint.load(resumed))
    _step(resumed)
    assert resumed.global_step == 1


def test_no_load_optim_leaves_optimizer_and_scheduler_fresh(tmp_path):
    actor = _actor(tmp_path)
    _step(actor)
    checkpoint.save(actor, 0)
    shutil.rmtree(tmp_path / "iter_0000001" / "optimizer")
    shutil.rmtree(tmp_path / "iter_0000001" / "lr_scheduler")
    resumed = _actor(tmp_path, load=str(tmp_path), no_load_optim=True)
    resumed.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(resumed.optimizer, gamma=0.8)
    optimizer_before = copy.deepcopy(resumed.optimizer.state_dict())
    scheduler_before = copy.deepcopy(resumed.lr_scheduler.state_dict())
    checkpoint.finalize_load(resumed, checkpoint.load(resumed))
    _assert_equal(resumed.optimizer.state_dict(), optimizer_before)
    _assert_equal(resumed.lr_scheduler.state_dict(), scheduler_before)


def test_no_save_optim_and_rng_require_explicit_load_opt_out(tmp_path):
    actor = _actor(tmp_path, no_save_optim=True, no_save_rng=True)
    checkpoint.save(actor, 0)
    directory = tmp_path / "iter_0000001"
    assert not (directory / "optimizer").exists()
    assert not (directory / "lr_scheduler").exists()
    assert not list(directory.glob("rng*"))
    resumed = _actor(tmp_path, load=str(tmp_path))
    with pytest.raises(ValueError, match="required optimizer"):
        checkpoint.load(resumed)
    resumed.args.no_load_optim = True
    with pytest.raises(ValueError, match="per-rank RNG"):
        checkpoint.load(resumed)
    resumed.args.no_load_rng = True
    checkpoint.finalize_load(resumed, checkpoint.load(resumed))
    assert resumed.args.start_rollout_id == 1


@pytest.mark.parametrize("component", ["model", "optimizer", "lr_scheduler", "rng_rank_00000.pt", "meta.json"])
def test_missing_required_state_aborts_without_cursor_finalize(tmp_path, component):
    actor = _actor(tmp_path)
    _step(actor)
    checkpoint.save(actor, 0)
    path = tmp_path / "iter_0000001" / component
    shutil.rmtree(path) if path.is_dir() else path.unlink()
    resumed = _actor(tmp_path, load=str(tmp_path))
    with pytest.raises(Exception):
        checkpoint.finalize_load(resumed, checkpoint.load(resumed))
    assert resumed.args.start_rollout_id == resumed.global_step == resumed.micro_step == 0


@pytest.mark.parametrize("component", ["model", "optimizer", "lr_scheduler"])
def test_corrupt_dcp_data_aborts_without_cursor_finalize(tmp_path, component):
    actor = _actor(tmp_path)
    _step(actor)
    checkpoint.save(actor, 0)
    for path in (tmp_path / "iter_0000001" / component).glob("*.distcp"):
        path.write_bytes(b"corrupt")
    resumed = _actor(tmp_path, load=str(tmp_path))
    with pytest.raises((Exception, CheckpointException)):
        checkpoint.finalize_load(resumed, checkpoint.load(resumed))
    assert resumed.args.start_rollout_id == resumed.global_step == resumed.micro_step == 0


def test_missing_or_corrupt_explicit_tracker_is_not_optional(tmp_path):
    actor = _actor(tmp_path)
    assert checkpoint.load(actor) is None
    actor.args.load = str(tmp_path)
    with pytest.raises(FileNotFoundError):
        checkpoint.load(actor)
    tracker = tmp_path / checkpoint.CHECKPOINT_TRACKER_FILENAME
    tracker.write_text("broken")
    with pytest.raises(ValueError, match="tracker"):
        checkpoint.load(actor)


def test_rng_is_restored_only_at_finalize(tmp_path):
    actor = _actor(tmp_path)
    checkpoint.save(actor, 0)
    expected = (torch.rand(3), random.random(), np.random.random())
    resumed = _actor(tmp_path, load=str(tmp_path))
    before = torch.get_rng_state().clone()
    payload = checkpoint.load(resumed)
    assert torch.equal(torch.get_rng_state(), before)
    checkpoint.finalize_load(resumed, payload)
    _assert_equal((torch.rand(3), random.random(), np.random.random()), expected)


def test_no_load_rng_preserves_current_rng(tmp_path):
    checkpoint.save(_actor(tmp_path), 0)
    (tmp_path / "iter_0000001" / "rng_rank_00000.pt").write_bytes(b"corrupt")
    resumed = _actor(tmp_path, load=str(tmp_path), no_load_rng=True)
    before = checkpoint._capture_rng()
    checkpoint.finalize_load(resumed, checkpoint.load(resumed))
    after = checkpoint._capture_rng()
    assert torch.equal(after["state"]["torch"], before["state"]["torch"])
    assert after["state"]["python"] == before["state"]["python"]
    np.testing.assert_equal(after["state"]["numpy"], before["state"]["numpy"])


@pytest.mark.parametrize("field", ["rank", "world_size"])
def test_rng_identity_mismatch_aborts_before_cursor(tmp_path, field):
    checkpoint.save(_actor(tmp_path), 0)
    path = tmp_path / "iter_0000001" / "rng_rank_00000.pt"
    state = torch.load(path, weights_only=False)
    state[field] += 1
    torch.save(state, path)
    resumed = _actor(tmp_path, load=str(tmp_path))
    with pytest.raises(ValueError, match="rank/world-size"):
        checkpoint.finalize_load(resumed, checkpoint.load(resumed))
    assert resumed.args.start_rollout_id == 0


def test_corrupt_rng_aborts_before_loading_model(tmp_path):
    checkpoint.save(_actor(tmp_path), 0)
    (tmp_path / "iter_0000001" / "rng_rank_00000.pt").write_bytes(b"corrupt")
    resumed = _actor(tmp_path, load=str(tmp_path))
    before = copy.deepcopy(resumed.model.state_dict())
    with pytest.raises(Exception):
        checkpoint.finalize_load(resumed, checkpoint.load(resumed))
    _assert_equal(resumed.model.state_dict(), before)
    assert resumed.args.start_rollout_id == 0


def test_explicit_step_does_not_need_tracker(tmp_path):
    checkpoint.save(_actor(tmp_path), 0)
    (tmp_path / checkpoint.CHECKPOINT_TRACKER_FILENAME).unlink()
    resumed = _actor(tmp_path, load=str(tmp_path), ckpt_step=1)
    checkpoint.finalize_load(resumed, checkpoint.load(resumed))
    assert resumed.args.start_rollout_id == 1


def test_optimizer_inventory_mismatch_is_not_partial_load(tmp_path):
    checkpoint.save(_actor(tmp_path), 0)
    resumed = _actor(tmp_path, load=str(tmp_path), combined=False)
    with pytest.raises(ValueError, match="optimizer layout"):
        checkpoint.finalize_load(resumed, checkpoint.load(resumed))
    assert resumed.args.start_rollout_id == 0


@pytest.mark.parametrize("buffer_owner", ["saved", "current"])
@pytest.mark.parametrize("no_load_optim", [False, True])
def test_model_exact_inventory_rejects_added_or_removed_buffer(tmp_path, buffer_owner, no_load_optim):
    actor = _actor(tmp_path)
    resumed = _actor(tmp_path, load=str(tmp_path), no_load_optim=no_load_optim)
    owner = actor if buffer_owner == "saved" else resumed
    owner.model.register_buffer("schema_probe", torch.ones(1))
    checkpoint.save(actor, 0)
    before = copy.deepcopy(resumed.model.state_dict())
    with pytest.raises(ValueError, match="model layout"):
        checkpoint.finalize_load(resumed, checkpoint.load(resumed))
    _assert_equal(resumed.model.state_dict(), before)
    assert resumed.args.start_rollout_id == resumed.global_step == resumed.micro_step == 0


@pytest.mark.parametrize("shape,dtype", [
    ((1, 3), torch.float64), ((3, 3), torch.float64),
    ((3, 2), torch.float64), ((2, 3), torch.float32),
])
def test_model_tensor_metadata_mismatch_aborts_before_loading(tmp_path, shape, dtype):
    actor = _actor(tmp_path)
    actor.model.register_buffer("schema_probe", torch.ones(2, 3, dtype=torch.float64))
    checkpoint.save(actor, 0)
    resumed = _actor(tmp_path, load=str(tmp_path), no_load_optim=True)
    resumed.model.register_buffer("schema_probe", torch.zeros(shape, dtype=dtype))
    before = copy.deepcopy(resumed.model.state_dict())
    with pytest.raises(ValueError, match="model tensor global size/dtype mismatch"):
        checkpoint.finalize_load(resumed, checkpoint.load(resumed))
    _assert_equal(resumed.model.state_dict(), before)
    assert resumed.args.start_rollout_id == resumed.global_step == resumed.micro_step == 0


@pytest.mark.parametrize("leaf_name", ["muon", "adamw"])
@pytest.mark.parametrize("change", ["smaller", "larger", "reshaped", "dtype"])
def test_optimizer_tensor_metadata_mismatch_aborts_before_model_load(tmp_path, leaf_name, change):
    actor = _actor(tmp_path)
    _step(actor)
    leaf = actor.optimizer.inner_optimizers()[leaf_name]
    parameter = actor.model.weight if leaf_name == "muon" else actor.model.bias
    key = "momentum_buffer" if leaf_name == "muon" else "exp_avg"
    tensor = leaf.state[parameter][key]
    if change == "smaller":
        changed = tensor[:1].clone()
    elif change == "larger":
        changed = torch.cat((tensor, tensor[:1]))
    elif change == "reshaped":
        changed = tensor.reshape(tuple(reversed(tensor.shape)) if tensor.ndim > 1 else (1, -1))
    else:
        changed = tensor.double()
    leaf.state[parameter][key] = changed
    checkpoint.save(actor, 0)
    resumed = _actor(tmp_path, load=str(tmp_path))
    before = copy.deepcopy(resumed.model.state_dict())
    with pytest.raises(ValueError, match="optimizer tensor global size/dtype mismatch"):
        checkpoint.finalize_load(resumed, checkpoint.load(resumed))
    _assert_equal(resumed.model.state_dict(), before)
    assert resumed.args.start_rollout_id == resumed.global_step == resumed.micro_step == 0
    resumed = _actor(tmp_path, load=str(tmp_path), no_load_optim=True)
    optimizer_before = copy.deepcopy(resumed.optimizer.state_dict())
    scheduler_before = copy.deepcopy(resumed.lr_scheduler.state_dict())
    checkpoint.finalize_load(resumed, checkpoint.load(resumed))
    _assert_equal(resumed.optimizer.state_dict(), optimizer_before)
    _assert_equal(resumed.lr_scheduler.state_dict(), scheduler_before)


@pytest.mark.parametrize("difference", ["saved_only", "current_only", "nested"])
def test_scheduler_exact_inventory_rejects_schema_changes_before_model_load(tmp_path, difference):
    actor = _actor(tmp_path)
    resumed = _actor(tmp_path, load=str(tmp_path))
    if difference in ("saved_only", "nested"):
        actor.lr_scheduler.schema_probe = {"nested": {"saved_key": torch.ones(1)}}
    if difference in ("current_only", "nested"):
        resumed.lr_scheduler.schema_probe = {"nested": {"current_key": torch.ones(1)}}
    checkpoint.save(actor, 0)
    before = copy.deepcopy(resumed.model.state_dict())
    with pytest.raises(ValueError, match="lr_scheduler layout"):
        checkpoint.finalize_load(resumed, checkpoint.load(resumed))
    _assert_equal(resumed.model.state_dict(), before)
    assert resumed.args.start_rollout_id == resumed.global_step == resumed.micro_step == 0


@pytest.mark.parametrize("scheduler_owner", ["saved", "current"])
def test_scheduler_presence_must_match(tmp_path, scheduler_owner):
    actor = _actor(tmp_path)
    resumed = _actor(tmp_path, load=str(tmp_path))
    if scheduler_owner == "saved":
        resumed.lr_scheduler = None
    else:
        actor.lr_scheduler = None
    checkpoint.save(actor, 0)
    before = copy.deepcopy(resumed.model.state_dict())
    with pytest.raises(ValueError, match="lr_scheduler"):
        checkpoint.finalize_load(resumed, checkpoint.load(resumed))
    _assert_equal(resumed.model.state_dict(), before)
    assert resumed.args.start_rollout_id == resumed.global_step == resumed.micro_step == 0


def test_checkpoint_without_scheduler_roundtrips(tmp_path):
    actor = _actor(tmp_path)
    actor.lr_scheduler = None
    checkpoint.save(actor, 0)
    metadata = json.loads((tmp_path / "iter_0000001" / "meta.json").read_text())
    assert metadata["components"]["lr_scheduler"] is False
    assert metadata["lr_scheduler_type"] is None
    resumed = _actor(tmp_path, load=str(tmp_path))
    resumed.lr_scheduler = None
    checkpoint.finalize_load(resumed, checkpoint.load(resumed))
    _assert_equal(resumed.model.state_dict(), actor.model.state_dict())
    assert resumed.args.start_rollout_id == 1


@pytest.mark.parametrize("difference", ["different_schema", "same_schema", "missing_type"])
def test_scheduler_type_is_bound_even_with_identical_inventory(tmp_path, difference):
    class OtherStepLR(torch.optim.lr_scheduler.StepLR):
        pass

    actor = _actor(tmp_path)
    checkpoint.save(actor, 0)
    path = tmp_path / "iter_0000001" / "meta.json"
    metadata = json.loads(path.read_text())
    assert metadata["lr_scheduler_type"] == "torch.optim.lr_scheduler.StepLR"
    resumed = _actor(tmp_path, load=str(tmp_path))
    if difference == "different_schema":
        resumed.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(resumed.optimizer, gamma=0.5)
    elif difference == "same_schema":
        resumed.lr_scheduler = OtherStepLR(resumed.optimizer, step_size=1, gamma=0.5)
        assert resumed.lr_scheduler.state_dict().keys() == actor.lr_scheduler.state_dict().keys()
    else:
        del metadata["lr_scheduler_type"]
        path.write_text(json.dumps(metadata))
    before = copy.deepcopy(resumed.model.state_dict())
    with pytest.raises(ValueError, match="lr_scheduler type"):
        checkpoint.finalize_load(resumed, checkpoint.load(resumed))
    _assert_equal(resumed.model.state_dict(), before)
    assert resumed.args.start_rollout_id == resumed.global_step == resumed.micro_step == 0


@pytest.mark.parametrize("backend,ranks", [("nccl", [0, 1]), ("gloo", [0]), ("gloo", [1, 0])])
def test_control_group_guard_rejects_wrong_backend_membership_or_order(monkeypatch, backend, ranks):
    group = object()
    monkeypatch.setattr(checkpoint, "get_gloo_group", lambda: group)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_backend", lambda actual: backend)
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda actual: ranks)
    with pytest.raises(ValueError, match="full world in rank order"):
        checkpoint._control_group()


def test_failed_save_does_not_publish_tracker(tmp_path, monkeypatch):
    actor = _actor(tmp_path)
    checkpoint.save(actor, 0)
    original = checkpoint.dcp.save

    def fail_optimizer(state, *, checkpoint_id, process_group):
        if checkpoint_id.name == "optimizer":
            raise RuntimeError("injected write failure")
        return original(state, checkpoint_id=checkpoint_id, process_group=process_group)

    monkeypatch.setattr(checkpoint.dcp, "save", fail_optimizer)
    with pytest.raises(RuntimeError, match="injected"):
        checkpoint.save(actor, 1)
    assert (tmp_path / checkpoint.CHECKPOINT_TRACKER_FILENAME).read_text() == "1"
    assert not (tmp_path / "iter_0000002" / "meta.json").exists()


def test_atomic_tracker_is_published_last_and_iterations_are_immutable(tmp_path, monkeypatch):
    actor = _actor(tmp_path)
    writes = []
    original = checkpoint._atomic_write

    def observe(path, text):
        writes.append(path.name)
        if path.name == checkpoint.CHECKPOINT_TRACKER_FILENAME:
            directory = tmp_path / "iter_0000001"
            assert (directory / "meta.json").is_file()
            assert (directory / "rng_rank_00000.pt").is_file()
            assert all((directory / name / ".metadata").is_file() for name in ("model", "optimizer", "lr_scheduler"))
        original(path, text)

    monkeypatch.setattr(checkpoint, "_atomic_write", observe)
    checkpoint.save(actor, 0)
    assert writes == ["meta.json", checkpoint.CHECKPOINT_TRACKER_FILENAME]
    with pytest.raises(FileExistsError):
        checkpoint.save(actor, 0)


def _distributed_rng_worker(rank, root, rendezvous):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        group = init_gloo_group()
        assert group is not dist.group.WORLD
        assert checkpoint._control_group() is group
        calls = {}
        with pytest.MonkeyPatch.context() as patch:
            def observe(namespace, name, group_keyword):
                original = getattr(namespace, name)

                def checked(*args, **kwargs):
                    assert kwargs.get(group_keyword) is group, f"{name} did not use the explicit control group"
                    calls[name] = calls.get(name, 0) + 1
                    return original(*args, **kwargs)

                patch.setattr(namespace, name, checked)

            for name in ("all_gather_object", "gather_object", "broadcast_object_list", "scatter_object_list", "barrier"):
                observe(dist, name, "group")
            for name in ("save", "load"):
                observe(checkpoint.dcp, name, "process_group")

            torch.manual_seed(100)
            actor = _actor(root, combined=False)
            _step(actor)
            torch.manual_seed(200 + rank)
            random.seed(300 + rank)
            np.random.seed(400 + rank)
            checkpoint.save(actor, 0)
            expected = (torch.rand(3), random.random(), np.random.random())
            resumed = _actor(root, combined=False, load=str(root))
            checkpoint.finalize_load(resumed, checkpoint.load(resumed))
            _assert_equal((torch.rand(3), random.random(), np.random.random()), expected)
            _assert_equal(resumed.model.state_dict(), actor.model.state_dict())
            _assert_equal(resumed.optimizer.state_dict(), actor.optimizer.state_dict())
            _assert_equal(resumed.lr_scheduler.state_dict(), actor.lr_scheduler.state_dict())
            assert calls["save"] == calls["load"] == 3
            assert all(calls[name] > 0 for name in ("all_gather_object", "gather_object", "broadcast_object_list", "barrier"))

            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.tensor import Replicate, Shard, distribute_tensor

            mesh = init_device_mesh("cpu", (2,))
            tensor = torch.arange(12, dtype=torch.float32).reshape(4, 3)
            sharded = distribute_tensor(tensor, mesh, [Shard(0)])
            path = root / "dtensor_metadata"
            saved = checkpoint.dcp.save({"model": {"weight": sharded}}, checkpoint_id=path, process_group=group)
            keys = sorted(saved.state_dict_metadata)
            assert tuple(sharded.to_local().shape) == (2, 3)
            for placements in ([Shard(0)], [Replicate()]):
                current = distribute_tensor(tensor, mesh, placements)
                checkpoint._all_ranks(lambda: checkpoint._checked_state("model", {"weight": current}, keys, path))
            for changed in (tensor[:3], tensor.double()):
                current = distribute_tensor(changed, mesh, [Shard(0)])
                with pytest.raises(ValueError, match="model tensor global size/dtype mismatch"):
                    checkpoint._all_ranks(lambda: checkpoint._checked_state("model", {"weight": current}, keys, path))

            dist.barrier(group=group)
            if rank == 0:
                (root / "iter_0000001" / "rng_rank_00001.pt").unlink()
            dist.barrier(group=group)
            resumed = _actor(root, combined=False, load=str(root))
            with pytest.raises((FileNotFoundError, RuntimeError)):
                checkpoint.finalize_load(resumed, checkpoint.load(resumed))
            assert resumed.args.start_rollout_id == 0

        subgroups = [dist.new_group(ranks=[member], backend="gloo") for member in range(2)]
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(checkpoint, "get_gloo_group", lambda: subgroups[rank])
            with pytest.raises(ValueError, match="full world in rank order"):
                checkpoint._control_group()
    finally:
        dist.destroy_process_group()


def test_two_rank_cpu_rng_roundtrip(tmp_path):
    processes = mp.start_processes(
        _distributed_rng_worker, args=(tmp_path, tmp_path / "rendezvous"),
        nprocs=2, join=False, start_method="spawn",
    )
    deadline = time.monotonic() + 60
    try:
        while not processes.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("CPU checkpoint workers timed out")
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    metadata = json.loads((tmp_path / "iter_0000001" / "meta.json").read_text())
    assert metadata["rng_files"] == ["rng_rank_00000.pt", "rng_rank_00001.pt"]
