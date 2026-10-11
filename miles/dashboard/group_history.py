"""Group decisions and confirmed training selection, including legacy exports."""

import json
from collections import Counter
from pathlib import Path
from threading import Lock


def _identity(row: dict) -> tuple | None:
    # A group keeps its identity across retries; the same prompt in another epoch
    # has a different group_index and must not be confused with this retry chain.
    if row.get("group_index") is None:
        return None
    return row.get("trainer_model_id"), row["group_index"]


def _final_events(events: list[dict]) -> list[dict]:
    final = {}
    for event in events:
        key = (event["role"], event.get("rollout_id"), event["name"], event["t0"])
        if key not in final or event.get("t1") is not None:
            final[key] = event
    return sorted(final.values(), key=lambda event: event["t0"])


def _training_status(selection: dict, events: list[dict]) -> dict:
    matches = [
        event
        for event in events
        if event["name"] == "train"
        and event.get("rollout_id") == selection["batch_id"]
        and event["t0"] >= selection["ts"]
        and (event.get("details") or {}).get("trainer_model_id") == selection.get("trainer_model_id")
    ]
    event = matches[-1] if matches else None
    status = "selected"
    if event:
        status = (
            "training_completed"
            if event.get("t1") is not None and event["status"] == "completed"
            else ("training_failed" if event["status"] == "failed" else "training_running")
        )
    return {
        "status": status,
        "batch_id": selection["batch_id"],
        "generation_attempt": selection.get("generation_attempt"),
        "selected_at": selection["ts"],
        "completed_at": event.get("t1") if event else None,
    }


def build_group_history(events: list[dict], legacy: dict, analysis_dir: Path) -> dict:
    events = _final_events(events)
    recovered = {entry["ts"]: entry for entry in legacy.get("rejections", [])}
    rows, selections = [], []
    for event in events:
        details = event.get("details") or {}
        if event["name"] == "buffer_reject":
            row = {
                **recovered.get(event["t0"], {}),
                **details,
                "ts": event["t0"],
                "event": "buffer_reject",
                "timestamp_kind": "decision",
                "identity_source": "runtime" if "group_index" in details else "unknown",
            }
            if row.get("group_index") is not None and "group_index" not in details:
                row["identity_source"] = "recovered_fifo"
            rows.append(row)
        elif event["name"] == "training_batch_selected":
            selections.extend(
                {
                    **identity,
                    "ts": event["t0"],
                    "event": "training_batch_selected",
                    "batch_id": event["rollout_id"],
                    "trainer_model_id": details.get("trainer_model_id"),
                    "timestamp_kind": "decision",
                    "identity_source": "runtime",
                }
                for identity in details.get("groups", [])
            )
    selections.extend(legacy.get("selections", []))
    rows.extend(legacy.get("unbound_dynamic_groups", []))
    selected_by_group = {}
    for selection in sorted(selections, key=lambda entry: entry["ts"]):
        selected_by_group.setdefault(_identity(selection), []).append(selection)
    totals = Counter(_identity(row) for row in rows if row.get("reason") == "stale" and _identity(row) is not None)
    stale_seen = Counter()
    for row in sorted(rows + selections, key=lambda entry: entry["ts"]):
        key = _identity(row)
        if row.get("reason") == "stale" and key is not None:
            stale_seen[key] += 1
        row["stale_count_at_event"] = stale_seen[key] if key is not None else None
        row["stale_count_total"] = totals[key] if key is not None else None
        later = next((selection for selection in selected_by_group.get(key, []) if selection["ts"] >= row["ts"]), None)
        row["training"] = _training_status(later, events) if later else {"status": "unknown"}
        gid, attempt = row.get("group_index"), row.get("generation_attempt")
        candidates = [f"prompts/group_{gid}_attempt_{attempt}.html", f"prompts/group_{gid}_saved_trials.html"]
        row["artifact"] = next(
            (path for path in candidates if gid is not None and (analysis_dir / path).is_file()), None
        )
    counts = Counter(row.get("reason") for row in rows if row.get("timestamp_kind") == "decision")
    return {
        "rows": sorted(rows + selections, key=lambda row: row["ts"], reverse=True),
        "counts": {
            "stale": counts["stale"],
            "dynamic_filter": counts["dynamic_filter"],
            "identified_stale": sum(row.get("reason") == "stale" and _identity(row) is not None for row in rows),
            "identified_dynamic": sum(
                row.get("reason") == "dynamic_filter" and _identity(row) is not None for row in rows
            ),
            "selected_groups": len(selections),
        },
        "coverage_note": legacy.get(
            "coverage_note", "Older decision events may not include prompt or group identities."
        ),
        "last_ts": max((event["t0"] for event in events), default=None),
    }


class GroupHistory:
    """Cache the small decision index; never read rollout tensors on refresh."""

    def __init__(self, analysis_dir: Path):
        self.analysis_dir = analysis_dir.resolve()
        self._lock = Lock()
        self._cache_key = None
        self._result = None

    def window(self, events, *, t0: float | None, t1: float | None, limit: int) -> dict:
        path = self.analysis_dir / "group_history.json"
        with self._lock:
            key = (len(events), path.stat().st_mtime_ns if path.is_file() else None)
            if key != self._cache_key:
                legacy = json.loads(path.read_text()) if path.is_file() else {}
                self._result = build_group_history([event.to_dict() for event in events], legacy, self.analysis_dir)
                self._cache_key = key
            result = self._result
        rows = [row for row in result["rows"] if (t0 is None or row["ts"] >= t0) and (t1 is None or row["ts"] <= t1)]
        return {**result, "rows": rows[:limit], "truncated": len(rows) > limit, "window_rows": len(rows)}

    def artifact_path(self, artifact: str) -> Path:
        path = (self.analysis_dir / artifact).resolve()
        if not path.is_relative_to(self.analysis_dir) or path.suffix not in {
            ".html",
            ".png",
            ".pdf",
            ".md",
            ".csv",
            ".json",
            ".jsonl",
            ".gz",
        }:
            raise FileNotFoundError("Unknown analysis artifact")
        if not path.is_file():
            raise FileNotFoundError("Analysis artifact not found")
        return path
