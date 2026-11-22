"""Bounded allocator tests; no Ray cluster, sockets, services or GPU work."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from miles.utils import misc
from miles.utils.workers.addr_allocator import PortAllocator


@pytest.fixture(autouse=True)
def clean_range(monkeypatch):
    monkeypatch.delenv("MILES_DYNAMIC_PORT_RANGE", raising=False)


def actor_probe():
    async def probe(**kwargs):
        return misc.NodeProbeMixin._get_free_port_block(**kwargs)

    return SimpleNamespace(_get_free_port_block=SimpleNamespace(remote=AsyncMock(side_effect=probe)))


@pytest.mark.parametrize("value", ["", "24000", "24000:24799", "24799-24000", "0-24000",
                                  "24000-65536", "-1-5", "1-2-3", " 24000-24799", "1-2\n",
                                  "true", "24000.0-24799", "a-b"])
def test_malformed_range_fails_at_construction(monkeypatch, value):
    monkeypatch.setenv("MILES_DYNAMIC_PORT_RANGE", value)
    with pytest.raises(ValueError, match="MILES_DYNAMIC_PORT_RANGE"):
        PortAllocator()


def test_default_rpc_and_start_remain_unchanged(monkeypatch):
    monkeypatch.setattr(misc, "is_port_available", lambda port: True)
    actor = actor_probe()
    allocator = PortAllocator()
    assert asyncio.run(allocator.alloc(actor, node_ip="node", consecutive=5)) == 20000
    actor._get_free_port_block.remote.assert_awaited_once_with(start_port=20000, count=5)
    assert allocator._next_port_of_ip == {"node": 20005}


def test_default_remote_wrap_still_starts_at_20000(monkeypatch):
    monkeypatch.setattr(misc, "is_port_available", lambda port: port == 20000)
    assert misc.NodeProbeMixin._get_free_port_block(start_port=65535, count=1) == 20000
    with pytest.raises(AssertionError):
        misc.get_free_port(start_port=65535, consecutive=2)


def test_inclusive_upper_bound_and_cursor_wrap(monkeypatch):
    monkeypatch.setenv("MILES_DYNAMIC_PORT_RANGE", "24000-24799")
    monkeypatch.setattr(misc, "is_port_available", lambda port: True)
    allocator = PortAllocator()
    allocator._next_port_of_ip["node"] = 24797
    actor = actor_probe()

    async def allocate():
        assert await allocator.alloc(actor, node_ip="node", consecutive=3) == 24797
        assert await allocator.alloc(actor, node_ip="node", consecutive=3) == 24000

    asyncio.run(allocate())
    actor._get_free_port_block.remote.assert_awaited_with(
        start_port=24000, count=3, min_port=24000, max_port=24799
    )


def test_remote_wrap_never_probes_above_or_below_bounds(monkeypatch):
    visited = []

    def available(port):
        visited.append(port)
        assert 24000 <= port <= 24004
        return port in {24000, 24001}

    monkeypatch.setattr(misc, "is_port_available", available)
    assert misc.NodeProbeMixin._get_free_port_block(
        start_port=24003, count=2, min_port=24000, max_port=24004
    ) == 24000
    assert visited == [24003, 24000, 24001]


def test_occupied_range_exhausts_in_bounds_without_cursor_change(monkeypatch):
    monkeypatch.setenv("MILES_DYNAMIC_PORT_RANGE", "24000-24799")
    visited = []

    def occupied(port):
        visited.append(port)
        return False

    monkeypatch.setattr(misc, "is_port_available", occupied)
    allocator = PortAllocator()
    allocator._next_port_of_ip["node"] = 24795
    with pytest.raises(RuntimeError, match=r"\[24000, 24799\]"):
        asyncio.run(allocator.alloc(actor_probe(), node_ip="node", consecutive=5))
    assert len(visited) == len(set(visited)) == 796
    assert min(visited) == 24000 and max(visited) == 24795
    assert allocator._next_port_of_ip == {"node": 24795}
    assert not allocator._lock.locked()


@pytest.mark.parametrize("count", [0, -1, True, 1.5, 801])
def test_invalid_block_rejected_before_rpc(monkeypatch, count):
    monkeypatch.setenv("MILES_DYNAMIC_PORT_RANGE", "24000-24799")
    actor = actor_probe()
    with pytest.raises(ValueError, match="consecutive"):
        asyncio.run(PortAllocator().alloc(actor, node_ip="node", consecutive=count))
    actor._get_free_port_block.remote.assert_not_awaited()


def test_whole_range_is_one_contiguous_block(monkeypatch):
    monkeypatch.setenv("MILES_DYNAMIC_PORT_RANGE", "24000-24799")
    visited = []
    monkeypatch.setattr(misc, "is_port_available", lambda port: visited.append(port) is None)
    assert asyncio.run(PortAllocator().alloc(actor_probe(), node_ip="node", consecutive=800)) == 24000
    assert visited == list(range(24000, 24800))


@pytest.mark.parametrize("returned", [23999, 24799, 24888, None, True])
def test_out_of_bounds_remote_result_rejected(monkeypatch, returned):
    monkeypatch.setenv("MILES_DYNAMIC_PORT_RANGE", "24000-24799")
    actor = SimpleNamespace(_get_free_port_block=SimpleNamespace(remote=AsyncMock(return_value=returned)))
    allocator = PortAllocator()
    with pytest.raises(RuntimeError, match="outside"):
        asyncio.run(allocator.alloc(actor, node_ip="node", consecutive=2))
    assert allocator._next_port_of_ip == {}


def test_range_captured_once_not_reinterpreted_by_actor_environment(monkeypatch):
    monkeypatch.setenv("MILES_DYNAMIC_PORT_RANGE", "24000-24799")
    allocator = PortAllocator()
    monkeypatch.setenv("MILES_DYNAMIC_PORT_RANGE", "30000-30010")
    monkeypatch.setattr(misc, "is_port_available", lambda port: True)
    assert asyncio.run(allocator.alloc(actor_probe(), node_ip="node")) == 24000


def test_concurrent_allocations_and_independent_nodes(monkeypatch):
    monkeypatch.setenv("MILES_DYNAMIC_PORT_RANGE", "24000-24799")
    monkeypatch.setattr(misc, "is_port_available", lambda port: True)
    allocator = PortAllocator()
    actor = actor_probe()

    async def allocate():
        values = await asyncio.gather(*[allocator.alloc(actor, node_ip="a", consecutive=2) for _ in range(4)])
        assert values == [24000, 24002, 24004, 24006]
        assert await allocator.alloc(actor, node_ip="b", consecutive=2) == 24000

    asyncio.run(allocate())


@pytest.mark.parametrize("bounds", [{"min_port": 0}, {"min_port": 25000, "max_port": 24000},
                                   {"max_port": 65536}, {"min_port": True}])
def test_native_probe_rejects_invalid_bounds(monkeypatch, bounds):
    monkeypatch.setattr(misc, "is_port_available", lambda port: pytest.fail("Invalid bounds reached probe"))
    with pytest.raises(ValueError, match="Port bounds"):
        misc.NodeProbeMixin._get_free_port_block(start_port=24000, count=1, **bounds)


@pytest.mark.parametrize("start,count", [(23999, 1), (24800, 1), (24799, 2)])
def test_native_probe_rejects_nonfitting_start_before_search(monkeypatch, start, count):
    monkeypatch.setattr(misc, "is_port_available", lambda port: pytest.fail("Invalid start reached probe"))
    with pytest.raises(AssertionError, match="leaves no room"):
        misc.NodeProbeMixin._get_free_port_block(
            start_port=start, count=count, min_port=24000, max_port=24799
        )
