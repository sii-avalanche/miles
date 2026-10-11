import json

import pytest
from fastapi.testclient import TestClient

from miles.dashboard.dump_reader import DumpReader
from miles.dashboard.group_history import GroupHistory, build_group_history
from miles.dashboard.server import make_app
from miles.dashboard.store import MetricStore, RuntimeEvent


def event(name, ts, *, details=None, batch=None, end=None, status="completed", role="rollout"):
    return {
        "role": role,
        "name": name,
        "rollout_id": batch,
        "t0": ts,
        "ts": end or ts,
        "t1": end if name == "train" else ts,
        "status": status,
        "details": details,
    }


def identity(attempt=1, gid=7):
    return {"query_id": "1677", "group_index": gid, "generation_attempt": attempt, "trainer_model_id": None}


def test_retry_count_and_later_training_require_a_matching_selected_group(tmp_path):
    events = [
        event("buffer_reject", 10, details={**identity(), "reason": "stale", "action": "retry"}),
        event("buffer_reject", 20, details={**identity(2), "reason": "stale", "action": "retry"}),
        event("training_batch_selected", 30, batch=4, details={"groups": [identity(3)]}),
        event("train", 31, batch=4, end=40, role="driver"),
        event("train", 31, batch=4, status="running", role="driver"),
        event("buffer_reject", 15, details={**identity(gid=8), "reason": "stale"}),
    ]
    result = build_group_history(events, {}, tmp_path)
    rows = {row["ts"]: row for row in result["rows"]}
    assert rows[10]["stale_count_at_event"] == 1
    assert rows[10]["stale_count_total"] == 2
    assert rows[20]["stale_count_at_event"] == 2
    assert rows[10]["training"] == {
        "status": "training_completed",
        "batch_id": 4,
        "generation_attempt": 3,
        "selected_at": 30,
        "completed_at": 40,
    }
    assert rows[15]["training"]["status"] == "unknown", "same query in another group is not this retry"


@pytest.mark.parametrize(
    "status,end,expected",
    [("running", None, "training_running"), ("failed", 40, "training_failed"), (None, None, "selected")],
)
def test_selection_does_not_claim_training_completed_without_success(tmp_path, status, end, expected):
    events = [event("training_batch_selected", 30, batch=4, details={"groups": [identity()]})]
    if status:
        events.append(event("train", 31, batch=4, end=end, status=status, role="driver"))
    assert build_group_history(events, {}, tmp_path)["rows"][-1]["training"]["status"] == expected


def test_old_events_remain_unknown_and_dynamic_completion_is_not_a_reject_time(tmp_path):
    events = [
        event("buffer_reject", 10, details={"reason": "stale"}),
        event("buffer_reject", 20, details={"reason": "dynamic_filter"}),
    ]
    legacy = {
        "rejections": [{"ts": 10, **identity()}],
        "unbound_dynamic_groups": [
            {"ts": 19, **identity(gid=9), "reason": "dynamic_filter", "timestamp_kind": "group_completed"}
        ],
    }
    result = build_group_history(events, legacy, tmp_path)
    assert result["counts"]["dynamic_filter"] == 1, "recovered identities do not double count raw reject events"
    assert result["counts"]["identified_dynamic"] == 1
    assert result["rows"][0]["training"]["status"] == "unknown"
    assert result["rows"][0]["stale_count_total"] is None
    assert result["rows"][-1]["identity_source"] == "recovered_fifo"


def test_history_window_keeps_whole_run_counts_and_artifacts_cannot_escape(tmp_path):
    root = tmp_path / "analysis/rollout_history"
    (root / "prompts").mkdir(parents=True)
    (root / "prompts/group_7_attempt_1.html").write_text("<p>prompt details</p>")
    store = MetricStore(tmp_path / "logs/dashboard")
    for ts, attempt in [(10, 1), (20, 2)]:
        store.records["runtime_events"].append(
            RuntimeEvent(**event("buffer_reject", ts, details={**identity(attempt), "reason": "stale"}))
        )
    client = TestClient(make_app(store, DumpReader(tmp_path / "logs")))
    data = client.get("/api/runtime/group_history?t0=9&t1=11").json()
    assert len(data["rows"]) == 1 and data["rows"][0]["stale_count_total"] == 2
    assert data["rows"][0]["artifact"] == "prompts/group_7_attempt_1.html"
    assert client.get("/analysis/prompts/group_7_attempt_1.html").text == "<p>prompt details</p>"
    history = GroupHistory(root)
    outside = tmp_path / "private.html"
    outside.write_text("private")
    (root / "escape.html").symlink_to(outside)
    for path in ["../../private.html", "escape.html", "group_history.json"]:
        with pytest.raises(FileNotFoundError):
            history.artifact_path(path)
    assert client.get("/api/runtime/group_history?t0=20&t1=10").status_code == 400


def test_sidecar_refreshes_without_new_runtime_events(tmp_path):
    history = GroupHistory(tmp_path)
    assert not history.window([], t0=None, t1=None, limit=100)["rows"]
    (tmp_path / "group_history.json").write_text(
        json.dumps(
            {
                "unbound_dynamic_groups": [
                    {"ts": 10, **identity(), "reason": "dynamic_filter", "timestamp_kind": "group_completed"}
                ]
            }
        )
    )
    assert len(history.window([], t0=None, t1=None, limit=100)["rows"]) == 1
