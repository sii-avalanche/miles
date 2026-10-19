"""Retryable file-to-database publication, outside the training/flush thread."""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from pathlib import Path

from miles.monitoring.transport.catalog import MANIFEST, _atomic_json, _created_at, _submitted_job, run_id
from miles.monitoring.transport.postgres import DashboardDatabase, block_summary

logger = logging.getLogger(__name__)
DSN_ENV = "AVATRAIN_DASHBOARD_POSTGRES"
BLOCK_BYTES = 1024 * 1024


def run_catalog(run_dir: Path, *, new_identity: bool = False):
    path = run_dir / MANIFEST
    manifest = json.loads(path.read_text()) if path.exists() else {}
    if new_identity and not manifest.get("dashboard_id"):
        manifest["dashboard_id"] = (
            run_id(f"{run_dir.parent.name}/{run_dir.name}") if path.exists() else uuid.uuid4().hex
        )
        manifest["owner"] = os.getenv("AVATRAIN_DASHBOARD_OWNER", os.getenv("USER", "unknown"))
        _atomic_json(path, manifest)
    relative = f"{run_dir.parent.name}/{run_dir.name}"
    return dict(
        id=manifest.get("dashboard_id", run_id(relative)),
        name=run_dir.name,
        config_name=run_dir.parent.name,
        relative_path=relative,
        job_id=manifest.get("job_id") or _submitted_job(run_dir.parent.parent, manifest.get("submission_id")),
        state=manifest.get("state", "unknown"),
        created_at=_created_at(run_dir, manifest),
        updated_at=manifest.get("updated_at", _created_at(run_dir, manifest)),
        owner=manifest.get("owner", os.getenv("AVATRAIN_DASHBOARD_OWNER", "unknown")),
        telemetry_ready=(run_dir / "logs/dashboard/meta.json").is_file(),
    )


def publish_manifest(run_dir: Path):
    dsn = os.getenv(DSN_ENV)
    if dsn:
        DashboardDatabase(dsn).publish_run(run_catalog(run_dir))


class TelemetryPublisher:
    def __init__(self, database: DashboardDatabase, run_dir: Path, *, telemetry_dir: Path | None = None):
        self.database = database
        self.run_dir = run_dir
        self.directory = telemetry_dir or run_dir / "logs/dashboard"
        self._stop = threading.Event()
        self._thread = None

    def sync(self):
        # Import only when telemetry is used, keeping CLI/catalog startup light.
        from miles.monitoring.transport.config import _SNAPSHOT_KEYS
        from miles.monitoring.transport.status import reward_info, training_strategy
        from miles.monitoring.transport.store import Stream

        catalog = run_catalog(self.run_dir)
        meta = json.loads((self.directory / "meta.json").read_text())
        meta["args"] = {key: value for key, value in meta.get("args", {}).items() if key in _SNAPSHOT_KEYS}
        lane_file = self.directory / "lane_catalog.json"
        metadata = dict(
            meta=meta,
            reward_info=reward_info(self.run_dir / "config.toml"),
            training_strategy=training_strategy(self.run_dir, meta.get("args", {})),
            lanes=json.loads(lane_file.read_text()).get("lanes", {}) if lane_file.exists() else {},
        )
        # Do not upload config.toml, environment, credentials, or tensor dumps.
        self.database.publish_run(dict(catalog, telemetry_ready=True), metadata)
        _, remote = self.database.describe(catalog["id"])
        uploaded = 0
        for stream in Stream:
            files = [self.directory / stream.filename, *sorted((self.directory / stream.value).glob("*.jsonl"))]
            for path in files:
                if not path.is_file():
                    continue
                relative = path.relative_to(self.directory).as_posix()
                offset = remote.get(relative, {}).get("size", 0)
                if path.stat().st_size < offset:
                    raise ValueError("Published telemetry was truncated; use a new run identity")
                with path.open("rb") as source:
                    source.seek(offset)
                    while not self._stop.is_set():
                        raw = source.read(BLOCK_BYTES)
                        if not raw:
                            break
                        end = raw.rfind(b"\n") + 1
                        if end == 0:
                            if len(raw) == BLOCK_BYTES:
                                raise ValueError("Dashboard record exceeds the 1 MiB publication limit")
                            break  # retain an incomplete final record for the next pass
                        raw = raw[:end]
                        stamps, keys = block_summary(raw)
                        self.database.append(catalog["id"], relative, offset, raw, stamps=stamps, keys=keys)
                        offset += len(raw)
                        uploaded += len(raw)
                        source.seek(offset)
        return uploaded

    def start(self):
        self._thread = threading.Thread(target=self._loop, name="dashboard-postgres", daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.sync()
            except Exception as error:
                # Some database exceptions include connection credentials; report the
                # type only. The original files retain everything needed for replay.
                logger.warning(
                    "Dashboard publication failed (%s); files retained, retrying in 5s", type(error).__name__
                )
            self._stop.wait(5)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._thread and self._thread.is_alive():
            logger.warning("Dashboard publisher still finishing; replay remaining files with sync")
            return
        # The collector has already flushed its final records. Use a bounded
        # background drain; shutdown must not hold training indefinitely.
        self._stop.clear()
        self._thread = threading.Thread(target=self._final_sync, name="dashboard-postgres-final", daemon=True)
        self._thread.start()
        self._thread.join(timeout=10)
        self._stop.set()

    def _final_sync(self):
        try:
            self.sync()
        except Exception as error:
            logger.warning("Dashboard final publication incomplete (%s); replay files with sync", type(error).__name__)


def start_publisher(telemetry_dir: str):
    dsn = os.getenv(DSN_ENV)
    if not dsn:
        return None
    directory = Path(telemetry_dir).resolve()
    run_dir = Path(os.environ["RUN_DIR"]) if os.getenv("RUN_DIR") else directory.parent.parent
    run_catalog(run_dir, new_identity=True)
    publisher = TelemetryPublisher(DashboardDatabase(dsn), run_dir, telemetry_dir=directory)
    publisher.start()
    return publisher
