import copy
import hashlib
import json

import pytest

from miles.rollout.rollout_recording import DurableRolloutOutbox, _atomic_json
from miles.rollout.screening_resume import completed_groups, inherit_records, validate_recording_sources


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
