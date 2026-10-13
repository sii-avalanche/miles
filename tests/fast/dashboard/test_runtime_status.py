import json

from fastapi.testclient import TestClient

from miles.dashboard.dump_reader import DumpReader
from miles.dashboard.runtime_status import reward_info, runtime_status
from miles.dashboard.server import make_app
from miles.dashboard.store import MetricsRecord, MetricStore

R = "runtime/rollout/"
D = "runtime/driver/"


def record(ts, metrics):
    return MetricsRecord(ts=ts, step_key="runtime/time_s", step=ts, metrics=metrics)


def test_stages_use_latest_snapshot_not_ingestion_order_or_viewport():
    records = [
        record(
            90,
            {
                R + "sample/generate/waiting": 0,
                R + "sample/reward/waiting": 4,
                R + "sample/reward/oldest_wait_seconds": 600,
                R + "sample/reward/count_total": 100,
                R + "record/pending_writes": 3,
                R + "record/in_flight_writes": 0,
            },
        ),
        record(80, {R + "sample/generate/waiting": 20, R + "sample/reward/waiting": 1}),
    ]
    result = runtime_status(records, now=100)
    stages = {s["id"]: s for s in result["rollout"]["stages"]}
    assert result["rollout"]["fresh"] is True
    assert stages["generate"]["active"] == 0  # newer zero clears earlier busy state
    assert stages["reward"]["active"] == 4
    assert stages["reward"]["oldest_seconds"] == 600
    assert stages["reward"]["finished_total"] == 100
    assert stages["persist"]["active"] is None  # absent is different from zero
    assert stages["database"]["pending"] == 3
    assert stages["database"]["active"] == 0  # backlog may exist without an active writer


def test_fresh_driver_does_not_make_stale_rollout_live():
    records = [
        record(30, {R + "sample/reward/waiting": 4}),
        record(99, {D + "phase/train/waiting": 1, D + "phase/train/oldest_wait_seconds": 10}),
    ]
    result = runtime_status(records, now=100)
    assert result["rollout"]["fresh"] is False
    assert result["driver"]["fresh"] is True
    assert result["driver"]["active_phases"] == [{"name": "train", "active": 1, "oldest_seconds": 10}]
    missing = runtime_status([], now=100)["rollout"]
    assert missing["observed_at"] is None
    assert missing["fresh"] is False


def test_concurrent_stages_and_nonfinite_metrics():
    result = runtime_status(
        [
            record(
                99,
                {
                    R + "sample/generate/waiting": 3,
                    R + "sample/reward/waiting": 7,
                    R + "record/persist/waiting": float("nan"),
                },
            )
        ],
        now=100,
    )
    stages = result["rollout"]["stages"]
    assert [s["active"] for s in stages[:3]] == [3, 7, None]
    json.dumps(result, allow_nan=False)


def test_api_reads_judge_identity_from_config_without_exposing_credentials(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    config = tmp_path / "config.toml"
    config.write_text(
        '[reward]\nkind="lmaaj"\n[reward.model]\nname="Kimi-K3"\n'
        'endpoint="http://private"\napi_key="secret"\ntimeout=600\nmax_retries=3\nconcurrency=32\n'
        '[env]\nPASSWORD="private"\n'
    )
    client = TestClient(make_app(MetricStore(logs / "dashboard"), DumpReader(logs)))
    assert client.get("/api/meta").json()["reward_info"] == {
        "kind": "lmaaj",
        "model_name": "Kimi-K3",
        "timeout": 600,
        "max_retries": 3,
        "concurrency": 32,
    }
    response = client.get("/api/runtime/status")
    assert response.status_code == 200
    assert "secret" not in response.text and "private" not in response.text
    assert response.json()["rollout"]["observed_at"] is None
    config.write_text('[reward]\nkind="math_verify"\n')
    assert reward_info(config) == {"kind": "math_verify"}
    config.write_text("invalid toml")
    assert reward_info(config) == {}
