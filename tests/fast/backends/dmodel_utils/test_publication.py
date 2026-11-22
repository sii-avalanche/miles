import asyncio

import pytest

from miles.backends.training_utils.weight_update.version import acknowledge_versions


class Engine:
    def __init__(self, version):
        self.version, self.calls = version, 0

    async def get_weight_version(self):
        self.calls += 1
        return self.version


def test_all_engines_acknowledged_without_random_sampling():
    engines = [Engine(3), Engine("3"), Engine(3)]
    assert asyncio.run(acknowledge_versions(engines, 3)) == dict(version=3, engines=3, versions=["3"] * 3)
    assert [e.calls for e in engines] == [1, 1, 1]


def test_single_stale_engine_rejected():
    engines = [Engine(3), Engine(2), Engine(3)]
    with pytest.raises(RuntimeError, match="Weight version mismatch"):
        asyncio.run(acknowledge_versions(engines, 3))
    assert [e.calls for e in engines] == [1, 1, 1]


@pytest.mark.parametrize("version", [True, 0, -1, None, "1"])
def test_invalid_version_not_queried(version):
    engine = Engine(1)
    with pytest.raises(ValueError):
        asyncio.run(acknowledge_versions([engine], version))
    assert engine.calls == 0


def test_empty_engine_set_not_accepted():
    with pytest.raises(ValueError):
        asyncio.run(acknowledge_versions([], 1))


def test_timeout_cancels_pending_version_query():
    cancelled = []

    class Hung:
        async def get_weight_version(self):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(True)

    with pytest.raises(TimeoutError):
        asyncio.run(acknowledge_versions([Hung()], 1, timeout=.01))
    assert cancelled == [True]
