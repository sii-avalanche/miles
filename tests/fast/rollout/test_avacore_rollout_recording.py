from types import SimpleNamespace

import pytest

from miles.rollout.avacore_rollout import _weight_version


def node(*, generated: bool, segments: tuple = (), weight_version=None):
    metadata = {} if weight_version is None else {"weight_version": weight_version}
    return SimpleNamespace(
        generated=generated,
        segments=segments,
        message=SimpleNamespace(metadata=metadata),
    )


def test_weight_version_survives_trailing_tool_result() -> None:
    trace = SimpleNamespace(
        prefixes=[
            node(generated=False),
            node(generated=True, segments=(object(),), weight_version="7"),
            node(generated=False),
        ]
    )

    assert _weight_version(trace) == 7


def test_weight_version_ignores_unrendered_generated_turn() -> None:
    trace = SimpleNamespace(
        prefixes=[
            node(generated=True, segments=(object(),), weight_version=8),
            node(generated=True, weight_version=9),
        ]
    )

    assert _weight_version(trace) == 8


def test_default_weight_version_is_recorded_as_initial_version() -> None:
    trace = SimpleNamespace(
        prefixes=[
            node(generated=True, segments=(object(),), weight_version="default"),
        ]
    )

    assert _weight_version(trace) == 0


def test_weight_version_requires_token_level_policy_turn() -> None:
    trace = SimpleNamespace(prefixes=[node(generated=False)])

    with pytest.raises(ValueError, match="no generated weight_version"):
        _weight_version(trace)
