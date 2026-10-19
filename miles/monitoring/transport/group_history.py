"""Group decisions and confirmed training selection, including legacy exports."""

import json
from pathlib import Path
from threading import Lock

from miles.monitoring.aggregation.runtime import group_history


def resolve_artifact(directory: Path, artifact: str) -> Path:
    """Allow supported analysis files within the resolved export directory."""
    directory = directory.resolve()
    path = (directory / artifact).resolve()
    if (
        not path.is_relative_to(directory)
        or path.suffix
        not in {
            ".html",
            ".png",
            ".pdf",
            ".md",
            ".csv",
            ".json",
            ".jsonl",
            ".gz",
        }
        or not path.is_file()
    ):
        raise FileNotFoundError("Analysis artifact not found")
    return path


def build_group_history(events: list[dict], legacy: dict, analysis_dir: Path) -> dict:
    result = group_history(events, legacy)
    for row in result["rows"]:
        gid, attempt = row.get("group_index"), row.get("generation_attempt")
        candidates = [f"prompts/group_{gid}_attempt_{attempt}.html", f"prompts/group_{gid}_saved_trials.html"]
        row["artifact"] = next(
            (path for path in candidates if gid is not None and (analysis_dir / path).is_file()), None
        )
    return result


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
        return resolve_artifact(self.analysis_dir, artifact)
