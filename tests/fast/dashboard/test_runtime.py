import time

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
