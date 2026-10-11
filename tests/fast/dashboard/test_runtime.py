import time
from argparse import Namespace
from types import SimpleNamespace

from fastapi.testclient import TestClient

from miles.dashboard.collector import CollectorConfig, DashboardCollector
from miles.dashboard.dump_reader import DumpReader
from miles.dashboard.server import make_app
from miles.dashboard.store import (
    EngineInfo,
    EngineSample,
    GpuSample,
    MetricStore,
    PhaseEvent,
    RuntimeEvent,
    TopologySnapshot,
)


def test_fleet_gauges_exclude_stale_and_unassigned_gpus(tmp_path):
    collector = DashboardCollector(
        config=CollectorConfig(
            dashboard_dir=str(tmp_path / "dashboard"),
            run_name="test",
            start_ts=0,
            args_snapshot={"actor_num_nodes": 1, "actor_num_gpus_per_node": 2, "rollout_num_gpus": 1},
        )
    )
    now = time.time()
    collector.push_phases([PhaseEvent(name="train", t0=now, t1=-1, node="node", gpus=[0, 1], rank=0, role="train")])
    collector.update_topology(
        TopologySnapshot(
            ts=now,
            engines=[
                EngineInfo(
                    addr="http://engine", worker_type="rollout", engine_rank=0, gpus=[["node", 2]], gpu_uuids=[]
                )
            ],
        )
    )
    collector.push_gpu_samples(
        "node",
        [
            GpuSample(ts=now, node="node", gpu=0, util=90, mem_mb=1024, power_w=200),
            GpuSample(ts=now - 100, node="node", gpu=1, util=80, mem_mb=1024, power_w=200),
            GpuSample(ts=now, node="node", gpu=2, util=50, mem_mb=2048, power_w=200),
            GpuSample(ts=now, node="node", gpu=3, util=100, mem_mb=4096, power_w=200),
        ],
    )
    values = collector.runtime_snapshot()
    assert values["train/gpu_util_mean_pct"] == 90
    assert values["train/coverage_fraction"] == 0.5
    assert values["inference/gpu_util_mean_pct"] == 50
    assert values["inference/mapped_gpus"] == 1
    collector._append(
        EngineSample(ts=now - 100, addr="http://engine", metric="sglang_num_queue_reqs", labels={}, value=100)
    )
    assert "inference/queued_requests" not in collector.runtime_snapshot()

    collector._append(EngineSample(ts=now, addr="http://engine", metric="sglang_mamba_usage", labels={}, value=0.9))
    assert collector.runtime_snapshot()["inference/mamba_usage_mean"] == 0.9
    assert collector.runtime_snapshot()["inference/mamba_usage_max"] == 0.9


def test_runtime_events_roundtrip_and_api_work_without_sample_dumps(tmp_path):
    collector = DashboardCollector(
        config=CollectorConfig(dashboard_dir=str(tmp_path / "dashboard"), run_name="test", start_ts=1)
    )
    event = RuntimeEvent(
        ts=3,
        role="driver",
        name="wait_rollout",
        rollout_id=7,
        t0=1,
        t1=3,
        status="failed",
        error_type="ActorDiedError",
    )
    collector.push_runtime_events([event])
    collector.flush()
    store = MetricStore.load(tmp_path / "dashboard")
    client = TestClient(make_app(store, DumpReader(tmp_path), follow=False))
    assert client.get("/api/runtime/events").json()["events"] == [event.to_dict()]
    assert client.get("/api/meta").status_code == 200
    series = client.get(
        "/api/metrics", params={"keys": "runtime/rollout/record/pending_writes", "x": "runtime/time_s"}
    ).json()
    assert series["runtime/rollout/record/pending_writes"]["y"] == []
    assert client.get("/api/runtime/events", params={"limit": 5001}).status_code == 422


def test_runtime_event_windows_keep_overlapping_phases_and_decision_details(tmp_path):
    store = MetricStore(tmp_path / "dashboard")
    records = [
        RuntimeEvent(ts=1, role="driver", name="update_weights", rollout_id=1, t0=1, t1=None, status="running"),
        RuntimeEvent(ts=5, role="driver", name="update_weights", rollout_id=1, t0=1, t1=5, status="completed"),
        RuntimeEvent(
            ts=6,
            role="rollout",
            name="buffer_reject",
            rollout_id=None,
            t0=6,
            t1=6,
            status="completed",
            details={"reason": "stale", "action": "drop", "oldest_version": 2, "current_version": 6},
        ),
        RuntimeEvent(ts=9, role="driver", name="train", rollout_id=2, t0=8, t1=9, status="completed"),
    ]
    for record in records:
        store.append(record)
    store.flush()
    client = TestClient(make_app(MetricStore.load(tmp_path / "dashboard"), DumpReader(tmp_path)))
    result = client.get("/api/runtime/events", params={"t0": 4, "t1": 7, "limit": 2}).json()
    assert result["truncated"] is False
    assert [e["name"] for e in result["events"]] == ["update_weights", "buffer_reject"]
    assert result["events"][-1]["details"]["oldest_version"] == 2
    assert client.get("/api/runtime/events", params={"t0": 4, "t1": 7, "limit": 1}).json()["truncated"] is True
    later = client.get("/api/runtime/events", params={"t0": 7, "t1": 10}).json()["events"]
    assert [e["name"] for e in later] == ["train"]
    assert client.get("/api/runtime/events", params={"t0": 7, "t1": 4}).status_code == 400


def test_runtime_monitor_to_tracking_collector_files_and_http(tmp_path, monkeypatch):
    # Real tracking/backend/store/API, with only the Ray transport replaced.
    # No cluster, CUDA, W&B service or database is started for this check.
    from miles.dashboard import backend
    from miles.utils import perf_monitor
    from miles.utils.tracking_utils import tracking

    collector = DashboardCollector(
        config=CollectorConfig(
            dashboard_dir=str(tmp_path / "dashboard"),
            run_name="runtime-wiring",
            start_ts=time.time(),
        )
    )
    handle = SimpleNamespace(
        push_metrics=SimpleNamespace(remote=collector.push_metrics),
        push_runtime_events=SimpleNamespace(remote=collector.push_runtime_events),
    )

    def connect(args, **kwargs):
        monkeypatch.setattr(backend, "_handle", handle)
        monkeypatch.setattr(backend, "_is_primary", False)

    monkeypatch.setattr(backend, "init_dashboard", connect)
    monkeypatch.setattr(perf_monitor, "_fleet_snapshot", collector.runtime_snapshot)
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    args = Namespace(use_miles_dashboard=True, perf_monitor_interval=10)
    tracking.init_tracking(args)
    try:
        monitor = perf_monitor.get_monitor(args, "driver")
        with monitor.phase("wait_rollout", rollout_id=2):
            time.sleep(0.001)
    finally:
        tracking.finish_tracking()
    collector.flush()

    store = MetricStore.load(tmp_path / "dashboard")
    client = TestClient(make_app(store, DumpReader(tmp_path), follow=False))
    key = "runtime/driver/phase/wait_rollout/count_total"
    values = client.get("/api/metrics", params={"keys": key, "x": "runtime/time_s"}).json()
    assert values[key]["y"][-1] == 1
    events = client.get("/api/runtime/events").json()["events"]
    assert {event["status"] for event in events} == {"running", "completed"}
    assert all(event["rollout_id"] == 2 for event in events)
    assert list((tmp_path / "logs" / "perf").glob("driver-*.jsonl"))
