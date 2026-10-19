"""Read and write run manifests without importing training or platform clients."""

import hashlib
import json
import logging
import os
import re
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

MANIFEST = "dashboard_run.json"
_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SUBMISSION_ID = re.compile(r"[a-f0-9]{32}\Z")
logger = logging.getLogger(__name__)


def _created_at(path: Path, data: dict) -> float:
    if isinstance(data.get("created_at"), (int, float)):
        return data["created_at"]
    # Legacy runs encode their creation time in the directory name. Directory
    # modification times change when logs/checkpoints are added later.
    try:
        return datetime.strptime(path.name, "run_%Y%m%d_%H%M%S").timestamp()
    except ValueError:
        return path.stat().st_mtime


def validate_job_id(job_id: str) -> None:
    if not _JOB_ID.fullmatch(job_id):
        raise ValueError("Expected a job ID containing only letters, digits, _, . and -")


def run_id(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode()).hexdigest()[:20]


def _atomic_json(path: Path, data: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".dashboard-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as output:
            json.dump(data, output)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _submitted_job(runs_dir: Path, submission_id: str | None) -> str | None:
    if not isinstance(submission_id, str) or not _SUBMISSION_ID.fullmatch(submission_id):
        return None
    path = runs_dir / ".dashboard_submissions" / f"{submission_id}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    job_id = data["job_id"]
    validate_job_id(job_id)
    return job_id


def write_manifest(
    run_dir: Path, *, job_id: str | None = None, state: str | None = None, submission_id: str | None = None,
) -> None:
    """Atomically replace metadata; never copy config, environment or credentials."""
    if job_id is not None:
        validate_job_id(job_id)
    path = run_dir / MANIFEST
    data = json.loads(path.read_text()) if path.is_file() else {"created_at": time.time()}
    if job_id is not None:
        data["job_id"] = job_id
    if state is not None:
        data["state"] = state
    if submission_id is not None:
        if not _SUBMISSION_ID.fullmatch(submission_id):
            raise ValueError("Invalid submission token")
        data["submission_id"] = submission_id
    data["updated_at"] = time.time()
    if os.getenv("AVATRAIN_DASHBOARD_POSTGRES"):
        # Keep existing file-backed bookmarks when enabling DB publication later.
        identity = run_id(f"{run_dir.parent.name}/{run_dir.name}") if path.exists() else uuid.uuid4().hex
        data.setdefault("dashboard_id", identity)
        data.setdefault("owner", os.getenv("AVATRAIN_DASHBOARD_OWNER", os.getenv("USER", "unknown")))
    _atomic_json(path, data)
    if os.getenv("AVATRAIN_DASHBOARD_POSTGRES"):
        # Optional persistence must not make a successful training run fail.
        from miles.dashboard.publisher import publish_manifest

        try:
            publish_manifest(run_dir)
        except Exception as error:
            logger.warning("Dashboard registration pending (%s); local manifest retained", type(error).__name__)


def discover_runs(runs_dir: Path) -> list[dict]:
    """Read only the two-level run catalog, not configs, checkpoints or log trees."""
    root = runs_dir.resolve()
    items = []
    for candidate in root.glob("*/run_*"):
        try:
            path = candidate.resolve()
            if not path.is_dir() or not path.is_relative_to(root):
                continue
            manifest = path / MANIFEST
            data = json.loads(manifest.read_text()) if manifest.is_file() else {}
            if not isinstance(data, dict):
                raise ValueError("Run manifest must be an object")
            relative = path.relative_to(root).as_posix()
            telemetry = path / "logs" / "dashboard"
            # Do not follow telemetry symlinks outside the allowed tree either.
            ready = telemetry.is_dir() and telemetry.resolve().is_relative_to(root)
            items.append({
                "id": data.get("dashboard_id", run_id(relative)),
                "name": path.name,
                "config_name": path.parent.name,
                "relative_path": relative,
                "job_id": data.get("job_id") or _submitted_job(root, data.get("submission_id")),
                "state": data.get("state", "unknown"),
                "created_at": _created_at(path, data),
                "telemetry_ready": ready,
            })
        except (OSError, ValueError, TypeError, KeyError):
            logger.warning("Skipping unreadable run catalog entry: %s", candidate.name)
    return sorted(items, key=lambda item: (item["created_at"], item["relative_path"]), reverse=True)
