import copy
import hashlib
import json

import pytest

from miles.utils.recording.outbox import DurableRolloutOutbox, _atomic_json
from miles.rollout.screening_resume import (
    completed_groups, inherit_records, validate_recording_sources, validate_reused_recording,
)


def source(tmp_path):
    directory = tmp_path / "old" / "rollout_records"
    (directory / "committed").mkdir(parents=True)
    manifest = {"model": "model", "run": "old", "collection": "data"}
    (directory / "run.json").write_text(json.dumps(manifest))
    record = {
        "query_id": "7", "trial_id": 0, "status": "completed", "reward": {"score": 1, "reason": "yes"},
        "instance": {"prompt": "q"}, "trace_graph": {"version": 1, "nodes": [], "tips": []},
        "metadata": {"weight_version": None},
    }
    name = hashlib.sha256(b"7:0").hexdigest() + ".json.gz"
    _atomic_json(directory / "committed" / name, record, compressed=True)
    groups = [{"source_index": 7, "rewards": [1], "recording_source": str(directory)}]
    return directory, groups, record


async def test_inheritance_preserves_snapshot_and_is_idempotent(tmp_path):
    directory, groups, original = source(tmp_path)
    snapshots = {p.name: p.read_bytes() for p in (directory / "committed").iterdir()}
    manifests = validate_recording_sources(groups, collection="data")
    writes = []

    async def write(record):
        writes.append(copy.deepcopy(record))

    outbox = DurableRolloutOutbox(tmp_path / "new", write, workers=1)
    await outbox.start()
    try:
        assert await inherit_records(outbox, groups, manifests) == 1
        assert await outbox.flush(2)
        assert await inherit_records(outbox, groups, manifests) == 1
        assert await outbox.flush(2)
        assert len(writes) == 1
        restored = writes[0]
        origin = restored["metadata"].pop("inherited_from")
        assert origin["run"] == "old"
        assert origin["query_id"] == "7"
        assert restored == original
        assert snapshots == {p.name: p.read_bytes() for p in (directory / "committed").iterdir()}
    finally:
        await outbox.close(2)


async def test_inheritance_rejects_score_mismatch(tmp_path):
    _, groups, _ = source(tmp_path)
    groups[0]["rewards"] = [0]
    manifests = validate_recording_sources(groups, collection="data")
    outbox = DurableRolloutOutbox(tmp_path / "new", None)
    with pytest.raises(ValueError, match="does not match"):
        await inherit_records(outbox, groups, manifests)
    assert not list(outbox.pending.iterdir())


def test_missing_snapshot_or_wrong_collection_stops_before_generation(tmp_path):
    _, groups, _ = source(tmp_path)
    with pytest.raises(ValueError, match="collection"):
        validate_recording_sources(groups, collection="different")
    groups[0]["rewards"] = [1, 1]
    with pytest.raises(FileNotFoundError, match="Missing confirmed"):
        validate_recording_sources(groups, collection="data")


@pytest.mark.parametrize("inherited_only", [False, True])
async def test_same_run_resume_preserves_committed_records_and_allocates_new_trials(tmp_path, inherited_only):
    from argparse import Namespace
    from unittest.mock import AsyncMock
    from miles.rollout.avacore_rollout import AvaCoreRollout
    from miles.utils.types import Sample

    directory, groups, _ = source(tmp_path)
    manifests = validate_recording_sources(groups, collection="data")
    original = {p.name: p.read_bytes() for p in (directory / "committed").iterdir()}
    writes = []

    async def write(record):
        writes.append(record)

    adapter = object.__new__(AvaCoreRollout)
    adapter.args = Namespace(prompt_data="data.jsonl")
    adapter.recording = {"model": "model", "run": "old"}
    if inherited_only:
        # An interrupted run may have only imported groups from its ancestors.
        adapter.recording["expected_run_id"] = 6907
        manifests = {path: {**manifest, "run": "ancestor"} for path, manifest in manifests.items()}
    adapter._outbox = DurableRolloutOutbox(directory, write, workers=1)
    adapter._shutdown_timeout = 2
    adapter._prepare_recording = AsyncMock()
    await adapter._outbox.seed_trials([("7", 0), ("7", 1000001)], 8)
    await adapter._outbox.start()
    try:
        assert await adapter.inherit_completed(groups, manifests, {}) == 1
        assert writes == []
        assert original == {p.name: p.read_bytes() for p in (directory / "committed").iterdir()}
        sample = Sample(index=56, epoch=0, group_index=7, metadata={"_index": 7})
        assert await adapter._outbox.allocate_trial(sample, 8) > 1000001
    finally:
        await adapter._outbox.close(2)


def test_same_run_resume_requires_the_original_spool_and_all_settled_trials(tmp_path):
    directory, groups, _ = source(tmp_path)
    identity = {"model": "model", "run": "old", "collection": "data"}
    assert validate_reused_recording(groups, directory, identity) == 1
    with pytest.raises(ValueError, match="different database run"):
        validate_reused_recording(groups, directory, {**identity, "model": "other"})
    groups[0]["rewards"] = [1, 0]
    with pytest.raises(FileNotFoundError, match="Missing confirmed"):
        validate_reused_recording(groups, directory, identity)


async def test_nonzero_retry_trial_survives_later_resume_and_copy(tmp_path):
    directory, groups, original = source(tmp_path)
    next((directory / "committed").iterdir()).unlink()
    original["trial_id"] = 16
    name = hashlib.sha256(b"7:16").hexdigest() + ".json.gz"
    _atomic_json(directory / "committed" / name, original, compressed=True)
    groups[0]["trial_ids"] = [16]
    manifests = validate_recording_sources(groups, collection="data")
    saved = []

    async def write(record):
        saved.append(record)

    outbox = DurableRolloutOutbox(tmp_path / "next", write, workers=1)
    await outbox.start()
    try:
        await inherit_records(outbox, groups, manifests)
        assert await outbox.flush(2)
        assert saved[0]["trial_id"] == 16
        assert saved[0]["metadata"]["inherited_from"]["trial_id"] == 16
    finally:
        await outbox.close(2)


@pytest.mark.parametrize("change", ["hash", "count", "duplicate", "partial", "decision"])
def test_corrupt_or_incompatible_group_history_is_rejected(tmp_path, change):
    summary = {"source_sha256": "hash", "total": 3, "samples_per_prompt": 2, "early_stop": True}
    group = {"source_index": 0, "rewards": [1, 1], "decision": "all_one"}
    if change == "hash":
        summary["source_sha256"] = "other"
    elif change == "count":
        summary["samples_per_prompt"] = 3
    elif change == "partial":
        group["rewards"] = [1]
    elif change == "decision":
        group["decision"] = "all_zero"
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    lines = [group, group] if change == "duplicate" else [group]
    (tmp_path / "groups.jsonl").write_text("".join(json.dumps(g) + "\n" for g in lines))
    with pytest.raises(ValueError):
        completed_groups(tmp_path, source_sha256="hash", total=3, n_samples=2, early_stop=True)
