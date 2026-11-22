from types import SimpleNamespace

import pytest

from miles.backends.dmodel_utils import actor


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("stale", [False, True])
def test_policy_acknowledgement_is_broadcast_from_driver(monkeypatch, rank, stale):
    calls = []

    class Engine:
        async def get_weight_version(self):
            calls.append(rank)
            return 1 if stale else 2

    def broadcast(rows, src, group):
        assert src == 0 and group == "gloo"
        if rank:
            rows[0] = {"passed": not stale, "version": 2}

    monkeypatch.setattr(actor.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(actor.dist, "broadcast_object_list", broadcast)
    monkeypatch.setattr(actor, "get_gloo_group", lambda: "gloo")
    trainer = SimpleNamespace(weight_updater=SimpleNamespace(weight_version=2))
    verify = actor.DModelTrainRayActor._verify_weight_version
    if stale:
        with pytest.raises(RuntimeError, match="not acknowledged"):
            verify(trainer, [Engine(), Engine()])
    else:
        verify(trainer, [Engine(), Engine()])
    assert calls == ([] if rank else [0, 0])
