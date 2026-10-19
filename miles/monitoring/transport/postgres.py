"""Shared dashboard catalog and append-only telemetry blocks in PostgreSQL.

The schema is deliberately separate from AvaCore's rollout tables. JSONL bytes
are compressed in bounded blocks so the existing dashboard keeps its exact
metric/event semantics; readers fetch only the requested stream partitions.
"""

from __future__ import annotations

import hashlib
import json
import zlib
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

SCHEMA = """
CREATE TABLE IF NOT EXISTS miles_dashboard.runs (
    id text PRIMARY KEY, catalog jsonb NOT NULL, metadata jsonb NOT NULL DEFAULT '{}',
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS miles_dashboard.files (
    run_id text NOT NULL REFERENCES miles_dashboard.runs(id), path text NOT NULL,
    size bigint NOT NULL DEFAULT 0, first_ts double precision, last_ts double precision,
    keys jsonb NOT NULL DEFAULT '[]', PRIMARY KEY (run_id, path)
);
CREATE TABLE IF NOT EXISTS miles_dashboard.blocks (
    run_id text NOT NULL, path text NOT NULL, start_offset bigint NOT NULL,
    end_offset bigint NOT NULL, sha256 text NOT NULL, payload bytea NOT NULL,
    PRIMARY KEY (run_id, path, start_offset),
    FOREIGN KEY (run_id, path) REFERENCES miles_dashboard.files(run_id, path)
);
"""


class DashboardDatabase:
    def __init__(self, dsn: str):
        self._dsn = dsn

    @contextmanager
    def connect(self):
        # Bound network/lock waits; never include the DSN in log messages.
        with psycopg.connect(
            self._dsn,
            connect_timeout=5,
            client_encoding="utf8",
            row_factory=dict_row,
            application_name="miles-dashboard",
            options="-c statement_timeout=15000 -c lock_timeout=5000",
        ) as connection:
            yield connection

    def initialize(self):
        with self.connect() as connection:
            # CREATE SCHEMA IF NOT EXISTS still requires database CREATE even
            # when the schema exists. A DBA can instead provision this schema
            # for a writer that only owns the dashboard schema.
            if connection.execute("SELECT to_regnamespace('miles_dashboard') AS schema").fetchone()["schema"] is None:
                connection.execute("CREATE SCHEMA miles_dashboard")
            connection.execute(SCHEMA)

    def publish_run(self, catalog: dict, metadata: dict | None = None):
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO miles_dashboard.runs(id,catalog,metadata) VALUES (%s,%s,%s) "
                "ON CONFLICT(id) DO UPDATE SET catalog=CASE WHEN "
                "COALESCE((EXCLUDED.catalog->>'updated_at')::double precision,0) >= "
                "COALESCE((miles_dashboard.runs.catalog->>'updated_at')::double precision,0) "
                "THEN EXCLUDED.catalog ELSE miles_dashboard.runs.catalog END, "
                "metadata=miles_dashboard.runs.metadata || EXCLUDED.metadata, updated_at=now()",
                (catalog["id"], Jsonb(catalog), Jsonb(metadata or {})),
            )

    def list_runs(self):
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT catalog, metadata ? 'meta' AS ready FROM miles_dashboard.runs ORDER BY "
                "(catalog->>'created_at')::double precision DESC"
            ).fetchall()
        return [dict(row["catalog"], storage="postgres", telemetry_ready=bool(row["ready"])) for row in rows]

    def describe(self, run_id: str):
        with self.connect() as connection:
            run = connection.execute("SELECT metadata FROM miles_dashboard.runs WHERE id=%s", (run_id,)).fetchone()
            if run is None:
                raise KeyError(run_id)
            files = connection.execute(
                "SELECT path,size,first_ts,last_ts,keys FROM miles_dashboard.files WHERE run_id=%s ORDER BY path",
                (run_id,),
            ).fetchall()
        return run["metadata"], {row["path"]: row for row in files}

    def tail(self, run_id: str, files: dict, stream: str, *, max_bytes: int = 262144):
        paths = sorted(path for path in files if path.startswith(stream + "/"))[-2:]
        paths = paths or ([stream + ".jsonl"] if stream + ".jsonl" in files else [])
        result = []
        for path in paths:
            size = files[path]["size"]
            offset = max(0, size - max_bytes)
            raw = self.read(run_id, path, offset, size)
            if offset:
                raw = raw.partition(b"\n")[2]
            result.extend(json.loads(line) for line in raw.splitlines())
        return result

    def append(self, run_id: str, path: str, offset: int, raw: bytes, *, stamps: list, keys: list):
        digest = hashlib.sha256(raw).hexdigest()
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO miles_dashboard.files(run_id,path) VALUES (%s,%s) ON CONFLICT DO NOTHING", (run_id, path)
            )
            current = connection.execute(
                "SELECT size,keys FROM miles_dashboard.files WHERE run_id=%s AND path=%s FOR UPDATE", (run_id, path)
            ).fetchone()
            if current["size"] != offset:
                previous = connection.execute(
                    "SELECT sha256 FROM miles_dashboard.blocks WHERE run_id=%s AND path=%s AND start_offset=%s",
                    (run_id, path, offset),
                ).fetchone()
                if previous and previous["sha256"] == digest:
                    return  # committed before an acknowledgement was lost
                raise ValueError("Telemetry changed or another publisher owns this run")
            connection.execute(
                "INSERT INTO miles_dashboard.blocks VALUES (%s,%s,%s,%s,%s,%s)",
                (run_id, path, offset, offset + len(raw), digest, zlib.compress(raw, 1)),
            )
            connection.execute(
                "UPDATE miles_dashboard.files SET size=%s, first_ts=LEAST(first_ts,%s), "
                "last_ts=GREATEST(last_ts,%s), keys=%s WHERE run_id=%s AND path=%s",
                (
                    offset + len(raw),
                    min(stamps) if stamps else None,
                    max(stamps) if stamps else None,
                    Jsonb(sorted(set(current["keys"]) | set(keys))),
                    run_id,
                    path,
                ),
            )

    def read(self, run_id: str, path: str, offset: int, size: int):
        """Read a contiguous committed prefix, respecting the caller's snapshot."""
        chunks = []
        with self.connect() as connection:
            with connection.cursor(name="dashboard_blocks") as cursor:
                cursor.execute(
                    "SELECT start_offset,end_offset,sha256,payload FROM miles_dashboard.blocks "
                    "WHERE run_id=%s AND path=%s AND end_offset>%s AND start_offset<%s "
                    "ORDER BY start_offset",
                    (run_id, path, offset, size),
                )
                for block in cursor:
                    raw = zlib.decompress(block["payload"])
                    if hashlib.sha256(raw).hexdigest() != block["sha256"]:
                        raise ValueError("Corrupt dashboard block")
                    lo = max(offset, block["start_offset"]) - block["start_offset"]
                    hi = min(size, block["end_offset"]) - block["start_offset"]
                    chunks.append(raw[lo:hi])
        result = b"".join(chunks)
        if len(result) != size - offset:
            raise ValueError("Missing dashboard block")
        return result


def block_summary(raw: bytes):
    stamps, keys = [], set()
    for line in raw.splitlines():
        row = json.loads(line)
        if "ts" in row:
            stamps.append(row["ts"])
        elif "t0" in row:
            stamps.append(row["t0"])
            if not row.get("open", False):
                stamps.append(row["t1"])
        if "metric" in row:
            keys.add(row["metric"])
    return stamps, sorted(keys)
