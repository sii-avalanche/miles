from types import SimpleNamespace

import pytest
import torch

from miles.backends.dmodel_utils import parallel


def test_mesh_gloo_group_matches_dp_members(monkeypatch):
    created = []
    groups = [(0, 2), (1, 3)]
    monkeypatch.setattr(parallel, "get_gloo_group", lambda: "world-gloo")
    monkeypatch.setattr(parallel.dist, "get_world_size", lambda: 4)

    def gather(output, local, group):
        assert local == (1, 3)
        assert group == "world-gloo"
        output[:] = [groups[0], groups[1], groups[0], groups[1]]

    def new_group(*, ranks, backend):
        assert backend == "gloo"
        created.append(tuple(ranks))
        return tuple(ranks)

    monkeypatch.setattr(parallel.dist, "all_gather_object", gather)
    monkeypatch.setattr(parallel.dist, "new_group", new_group)
    mesh = SimpleNamespace(mesh=torch.tensor([1, 3]))
    assert parallel._mesh_gloo_group(mesh) == (1, 3)
    assert created == groups


def test_mesh_gloo_group_rejects_non_partition(monkeypatch):
    monkeypatch.setattr(parallel, "get_gloo_group", lambda: "world-gloo")
    monkeypatch.setattr(parallel.dist, "get_world_size", lambda: 4)

    def gather(output, local, group):
        output[:] = [(0, 2), (1, 2), (0, 2), (1, 2)]

    monkeypatch.setattr(parallel.dist, "all_gather_object", gather)
    monkeypatch.setattr(parallel.dist, "new_group", lambda **kwargs: pytest.fail("unexpected group creation"))
    with pytest.raises(ValueError, match="does not partition"):
        parallel._mesh_gloo_group(SimpleNamespace(mesh=torch.tensor([0, 2])))
