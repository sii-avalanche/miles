import torch.distributed as dist

from miles.backends.training_utils.parallel import ParallelState
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.ft_utils.process_group_utils import GroupInfo


def _group_info(mesh, *, gloo_group=None) -> GroupInfo:
    group = mesh.get_group()
    return GroupInfo(
        rank=dist.get_rank(group),
        size=dist.get_world_size(group),
        group=group,
        gloo_group=gloo_group,
    )


def _mesh_gloo_group(mesh):
    """Create a CPU metrics group with exactly the members of the mesh slice."""
    local_ranks = tuple(int(rank) for rank in mesh.mesh.flatten().tolist())
    world_size = dist.get_world_size()
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_ranks, group=get_gloo_group())
    groups = sorted(set(gathered))
    if (len(local_ranks) != len(set(local_ranks)) or
            sorted(rank for ranks in groups for rank in ranks) != list(range(world_size))):
        raise ValueError("DModel metrics mesh does not partition the world")
    selected = None
    for ranks in groups:
        group = dist.new_group(ranks=list(ranks), backend="gloo")
        if ranks == local_ranks:
            selected = group
    if selected is None:
        raise ValueError("DModel rank is absent from its metrics mesh")
    return selected


def create_dmodel_parallel_state(parallelism) -> ParallelState:
    """Expose DModel's real data, context and expert meshes to common Miles code."""
    dp_mesh = parallelism.get_mesh("dp")
    metrics_mesh = parallelism.get_mesh("dp_cp") if parallelism.context_parallel_size > 1 else dp_mesh
    metrics_gloo = _mesh_gloo_group(metrics_mesh)
    dp = _group_info(dp_mesh)
    cp = _group_info(parallelism.get_mesh("cp_ring"))
    dp_cp = _group_info(metrics_mesh, gloo_group=metrics_gloo)
    self_group = dist.new_group([dist.get_rank()])
    self_info = GroupInfo(rank=0, size=1, group=self_group)
    ep = _group_info(parallelism.get_mesh("ep"))

    return ParallelState(
        intra_dp=dp,
        intra_dp_cp=dp_cp,
        cp=cp if parallelism.context_parallel_size > 1 else self_info,
        tp=self_info,
        pp=self_info,
        ep=ep,
        etp=self_info,
        indep_dp=self_info,
        edp=_group_info(parallelism.get_mesh("edp_shard")),
        meshes={
            "dp": parallelism.get_mesh("dp"),
            "fsdp": parallelism.get_dp_mesh(),
            "ep": parallelism.get_mesh("ep"),
            "edp_shard": parallelism.get_mesh("edp_shard"),
        },
    )
