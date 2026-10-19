import json
import subprocess
import sys

from fastapi.testclient import TestClient

from miles.dashboard.__main__ import main
from miles.dashboard.catalog import MANIFEST


def test_help_does_not_import_training_dependencies():
    code = """
import runpy, sys
sys.argv = ['miles.dashboard', '--help']
try:
    runpy.run_module('miles.dashboard', run_name='__main__')
finally:
    assert 'torch' not in sys.modules and 'ray' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "serve" in result.stdout and "view" in result.stdout and "register" in result.stdout


def test_serve_dispatch_preserves_authentication(tmp_path, monkeypatch):
    apps = []
    monkeypatch.delenv("AVATRAIN_DASHBOARD_PASSWORD_FILE", raising=False)
    monkeypatch.setenv("AVATRAIN_DASHBOARD_PASSWORD", "test-password")
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: apps.append(app))
    main(["serve", "--runs-dir", str(tmp_path)])
    with TestClient(apps[0]) as client:
        assert client.get("/api/runs").status_code == 401
        response = client.get("/api/runs", auth=("admin", "test-password"))
        assert response.json() == {"runs": [], "total": 0}


def test_view_dispatch_serves_telemetry_with_follow_lifecycle(tmp_path, monkeypatch):
    from miles.dashboard.store import Meta, MetricStore

    directory = tmp_path / "logs" / "dashboard"
    MetricStore(directory).write_meta(Meta(run_name="cli-view", start_ts=1, args={}))
    apps = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: apps.append(app))
    main(["view", "--dashboard-dir", str(directory), "--follow", "--low-mfu-threshold", "0"])
    with TestClient(apps[0]) as client:
        metadata = client.get("/api/meta").json()
        assert metadata["run_name"] == "cli-view"
        assert metadata["mode"] == "follow"


def test_register_dispatch_writes_manifest(tmp_path):
    main(["register", "j-example", str(tmp_path)])
    assert json.loads((tmp_path / MANIFEST).read_text())["job_id"] == "j-example"
