"""Two-rank CUDA comparison. Run with torchrun --nproc-per-node=2 -m pytest."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from dmodel.training.optimizer.offload import OffloadAdamW
from dmodel.training.optimizer import CombinedOptimizer, MuonWithStackedExperts
from dmodel.training.optimizer.state import set_flattened_optimizer_state_dict
from miles.backends.dmodel_utils.checkpoint import _optimizer_state
from dmodel.training.optimizer.muon_offload import GPUOptimizerOffload
from miles.backends.fsdp_utils.dtensor import gather_full_param


@pytest.mark.parametrize("muon_prefetch", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("WORLD_SIZE") != "2",
                    reason="requires two CUDA torchrun workers")
def test_sharded_update_checkpoint_and_publication(tmp_path, muon_prefetch):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    owns_group = not dist.is_initialized()
    if owns_group:
        store, global_rank, world = next(dist.rendezvous("env://"))
        dist.init_process_group("cpu:gloo,cuda:nccl", store=dist.PrefixStore(f"optimizer-{muon_prefetch}", store),
                                rank=global_rank, world_size=world)
    try:
        _compare_updates(tmp_path, muon_prefetch)
    finally:
        if owns_group:
            dist.destroy_process_group()


def _compare_updates(tmp_path, muon_prefetch):
    mesh = init_device_mesh("cuda", (2,))

    def create():
        model = torch.nn.Module()
        for name, shape in [("dense", (16, 8)), ("experts", (4, 8, 8)), ("bias", (512,))]:
            value = torch.arange(torch.Size(shape).numel(), dtype=torch.float32, device="cuda").reshape(shape) / 1024
            param = distribute_tensor(value, mesh, [Shard(0)])
            model.register_parameter(name, torch.nn.Parameter(param))
        muon = MuonWithStackedExperts([model.dense, model.experts],
                                      param_names={id(p): n for n, p in model.named_parameters()})
        adam = OffloadAdamW([model.bias], lr=0.01, foreach=False)
        return model, CombinedOptimizer(muon, adam)

    reference, baseline = create()
    model, optimizer = create()
    policy = GPUOptimizerOffload(optimizer, max_bucket_bytes=2048, muon_prefetch=muon_prefetch).install()
    # Both ranks must address the same checkpoint directory on shared storage.
    paths = [str(tmp_path) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(paths, src=0)
    for step in range(3):
        for module, opt in [(reference, baseline), (model, optimizer)]:
            for param in module.parameters():
                param.grad = torch.ones_like(param) * (0.3 + step)
            opt.step()
            opt.zero_grad(set_to_none=True)
        tensors = [value for leaf in optimizer.inner_optimizers().values() for state in leaf.state.values()
                   for value in state.values() if torch.is_tensor(value)]
        assert len(tensors) >= 5 and all(value.device.type == "cpu" for value in tensors)
        for saved, actual in zip(reference.parameters(), model.parameters(), strict=True):
            torch.testing.assert_close(gather_full_param(actual), gather_full_param(saved))
        ref_state = _optimizer_state(reference, baseline)
        state = _optimizer_state(model, optimizer)
        for key, item in state.items():
            if isinstance(item, torch.Tensor):
                local = item.to_local() if hasattr(item, "to_local") else item
                expected = ref_state[key]
                expected = expected.to_local() if hasattr(expected, "to_local") else expected
                assert local.device.type == "cpu"
                torch.testing.assert_close(local, expected.cpu())
        if step == 0:
            dcp.save({"optimizer": state}, checkpoint_id=paths[0])
            resumed, optimizer = create()
            policy = GPUOptimizerOffload(optimizer, max_bucket_bytes=2048, muon_prefetch=muon_prefetch).install()
            resumed.load_state_dict(model.state_dict())
            model = resumed
            state = _optimizer_state(model, optimizer)
            dcp.load({"optimizer": state}, checkpoint_id=paths[0])
            set_flattened_optimizer_state_dict(model, optimizer, state)
    if muon_prefetch:
        assert torch.cuda.current_device() in policy.policies["muon"].transfer_streams


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prefetch_empty_local_shard():
    from dmodel.training.optimizer.muon_prefetch import _layout

    shard = torch.empty((0, 16), device="cuda")
    momentum = torch.empty_like(shard, device="cpu")
    entries, size = _layout([(shard, shard, momentum)])
    assert size == 0
    assert entries[0].gpu_view(shard.view(torch.uint8).flatten()).shape == momentum.shape
