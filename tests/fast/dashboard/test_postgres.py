"""Integration checks against an isolated PostgreSQL database supplied by the caller."""

import json
import os
import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from miles.dashboard.catalog import discover_runs, write_manifest
from miles.dashboard.postgres import DashboardDatabase, block_summary
from miles.dashboard.postgres_reader import PostgresMetricStore
from miles.dashboard.publisher import TelemetryPublisher, run_catalog
from miles.dashboard.store import Meta, MetricsRecord, MetricStore, Stream
from miles.dashboard.web.service import make_service
from tests.fast.dashboard.dummy_telemetry import BASE_TS, dump_dummy_telemetry


@pytest.fixture
def database():
    dsn = os.getenv("MILES_DASHBOARD_TEST_POSTGRES")
    if not dsn:
        pytest.skip("Set MILES_DASHBOARD_TEST_POSTGRES to an isolated test database")
    database = DashboardDatabase(dsn)
    database.initialize()
    with database.connect() as connection:
        connection.execute("TRUNCATE miles_dashboard.runs CASCADE")
    return database


def make_run(tmp_path):
    run = tmp_path / "source" / "experiment" / "run_20261019_120000"
    dump_dummy_telemetry(run / "logs", steps=2)
    write_manifest(run, state="running", job_id="j-test")
    return run


def test_dashboard_reads_database_without_training_mount(database, tmp_path):
    run = make_run(tmp_path)
    key = run_catalog(run)["id"]
    publisher = TelemetryPublisher(database, run)
    assert publisher.sync() > 0
    assert publisher.sync() == 0
    local = make_service(run.parent.parent, cache_dir=tmp_path / "local-cache", password=None)
    shared = make_service(tmp_path / "unmounted", cache_dir=tmp_path / "shared-cache",
                          password="test", postgres=os.environ["MILES_DASHBOARD_TEST_POSTGRES"])
    paths = ["api/meta", "api/timeline/topology",
             f"api/timeline/phases?t0={BASE_TS}&t1={BASE_TS + 250}",
             f"api/timeline/gpu?t0={BASE_TS}&t1={BASE_TS + 250}"]
    with TestClient(local) as client:
        expected = {path: client.get(f"/runs/{key}/{path}") for path in paths}
    shutil.rmtree(run.parent.parent)
    with TestClient(shared) as client:
        assert client.get("/api/runs").status_code == 401
        client.auth = ("admin", "test")
        row = client.get("/api/runs").json()["runs"][0]
        assert row["id"] == key and row["job_id"] == "j-test"
        for path, reference in expected.items():
            response = client.get(f"/runs/{key}/{path}")
            assert reference.status_code == response.status_code == 200, (path, response.text)
            assert response.json() == reference.json(), path
        assert client.get(f"/runs/{key}/").status_code == 200
        assert client.get(f"/runs/{key}/api/runtime/status").status_code == 200
        assert 'job_id="j-test"' in client.get("/metrics").text


def test_retry_partial_lines_and_incremental_reads(database, tmp_path):
    run = make_run(tmp_path)
    publisher = TelemetryPublisher(database, run)
    publisher.sync()
    store = PostgresMetricStore(database, run_catalog(run)["id"], tmp_path / "reader")
    before = len(store.records[Stream.METRICS])
    file = run / "logs/dashboard/metrics.jsonl"
    row = MetricsRecord(ts=BASE_TS + 999, step=99, step_key="step", metrics={"test/new": 42}).to_dict()
    encoded = json.dumps(row).encode()
    with file.open("ab") as output:
        output.write(encoded[:20])
    assert publisher.sync() == 0
    with file.open("ab") as output:
        output.write(encoded[20:] + b"\n")
    assert publisher.sync() > 0
    assert publisher.sync() == 0
    store.follow()
    store.follow()
    assert len(store.records[Stream.METRICS]) == before + 1
    assert store.records[Stream.METRICS][-1].metrics == {"test/new": 42}


def test_lost_acknowledgement_does_not_duplicate(database, tmp_path):
    run = make_run(tmp_path)
    catalog = run_catalog(run)
    database.publish_run(catalog)
    raw = b'{"ts":1,"step":0,"step_key":"step","metrics":{"loss":1}}\n'
    stamps, keys = block_summary(raw)
    for _ in range(2):
        database.append(catalog["id"], "metrics.jsonl", 0, raw, stamps=stamps, keys=keys)
    assert database.read(catalog["id"], "metrics.jsonl", 0, len(raw)) == raw
    with pytest.raises(ValueError):
        database.append(catalog["id"], "metrics.jsonl", 0, raw.replace(b'loss', b'gain'), stamps=stamps, keys=keys)


def test_user_run_identity_and_metadata_redaction(database, tmp_path, monkeypatch):
    monkeypatch.setenv("AVATRAIN_DASHBOARD_POSTGRES", os.environ["MILES_DASHBOARD_TEST_POSTGRES"])
    identities = []
    for owner in ("alice", "bob"):
        monkeypatch.setenv("AVATRAIN_DASHBOARD_OWNER", owner)
        run = make_run(tmp_path / owner)
        catalog = run_catalog(run)
        identities.append(catalog["id"])
        assert discover_runs(run.parent.parent)[0]["id"] == catalog["id"]
        assert catalog["owner"] == owner
        meta = run / "logs/dashboard/meta.json"
        data = json.loads(meta.read_text())
        data["args"].update(password="must-not-upload", postgres="secret-url")
        meta.write_text(json.dumps(data))
        TelemetryPublisher(database, run).sync()
        uploaded, _ = database.describe(catalog["id"])
        assert "must-not-upload" not in json.dumps(uploaded) and "secret-url" not in json.dumps(uploaded)
    assert identities[0] != identities[1]
    assert len(database.list_runs()) == 2


def test_windowed_database_reader_matches_files(database, tmp_path):
    run = make_run(tmp_path)
    TelemetryPublisher(database, run).sync()
    file = MetricStore.load(run / "logs/dashboard")
    shared = PostgresMetricStore(database, run_catalog(run)["id"], tmp_path / "reader")
    assert shared.lanes() == file.lanes()
    assert shared.time_range() == file.time_range()
    assert shared.engine_metric_names() == file.engine_metric_names()
    for stream in MetricStore.PARTITIONED_STREAMS:
        expected = file._readers[stream].window(BASE_TS, BASE_TS + 250)
        actual = shared._readers[stream].window(BASE_TS, BASE_TS + 250)
        if stream in MetricStore.COLUMNAR_STREAMS:
            assert actual.equals(expected)
        else:
            assert actual == expected


def test_stale_publisher_does_not_revert_final_status(database, tmp_path):
    run = make_run(tmp_path)
    catalog = run_catalog(run)
    database.publish_run(dict(catalog, state="completed", updated_at=100))
    database.publish_run(dict(catalog, state="running", updated_at=99))
    assert database.list_runs()[0]["state"] == "completed"


def test_collector_automatically_publishes_new_run(database, tmp_path, monkeypatch):
    from miles.dashboard.collector import CollectorConfig, DashboardCollector

    monkeypatch.setenv("AVATRAIN_DASHBOARD_POSTGRES", os.environ["MILES_DASHBOARD_TEST_POSTGRES"])
    monkeypatch.setenv("AVATRAIN_DASHBOARD_OWNER", "teammate")
    run = tmp_path / "independent-workspace" / "experiment" / "run_20261019_130000"
    run.mkdir(parents=True)
    monkeypatch.setenv("RUN_DIR", str(run))
    monkeypatch.setattr("miles.dashboard.collector._list_gpu_nodes", lambda: [])
    write_manifest(run, state="running")
    assert database.list_runs()[0]["telemetry_ready"] is False
    collector = DashboardCollector(config=CollectorConfig(
        dashboard_dir=str(run / "logs/dashboard"), run_name="new-team-run", start_ts=1,
        flush_interval_seconds=0.05,
    ))
    collector.start()
    try:
        collector.push_metrics(MetricsRecord(ts=2, step=0, step_key="step", metrics={"train/loss": 0.5}))
        collector.flush()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            _, files = database.describe(run_catalog(run)["id"])
            if files.get("metrics.jsonl", {}).get("size", 0):
                break
            time.sleep(0.1)
        else:
            pytest.fail("Collector did not automatically upload telemetry")
    finally:
        collector.shutdown()
    write_manifest(run, state="completed")
    assert database.list_runs()[0]["state"] == "completed"
    assert database.list_runs()[0]["owner"] == "teammate"


def test_database_outage_keeps_files_for_retry(database, tmp_path, monkeypatch):
    run = make_run(tmp_path)
    publisher = TelemetryPublisher(database, run)
    original = database.append
    monkeypatch.setattr(database, "append", lambda *a, **kw: (_ for _ in ()).throw(ConnectionError()))
    with pytest.raises(ConnectionError):
        publisher.sync()
    assert (run / "logs/dashboard/metrics.jsonl").stat().st_size > 0
    monkeypatch.setattr(database, "append", original)
    assert publisher.sync() > 0
    assert publisher.sync() == 0


def test_enabling_database_preserves_existing_bookmark(database, tmp_path, monkeypatch):
    run = make_run(tmp_path)
    before = run_catalog(run)["id"]
    monkeypatch.setenv("AVATRAIN_DASHBOARD_POSTGRES", os.environ["MILES_DASHBOARD_TEST_POSTGRES"])
    write_manifest(run, state="completed")
    assert run_catalog(run)["id"] == before
    assert discover_runs(run.parent.parent)[0]["id"] == before
