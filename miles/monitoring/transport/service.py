"""One authenticated HTTP service for runs, telemetry, Prometheus and Grafana."""

import asyncio
import base64
import binascii
import contextlib
import hashlib
import hmac
import html
import json
import logging
import math
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
from collections import OrderedDict, defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import Mount, get_route_path

from miles.monitoring.frontend import STATIC_DIR
from miles.monitoring.transport.catalog import discover_runs, validate_job_id
from miles.monitoring.transport.group_history import resolve_artifact

logger = logging.getLogger(__name__)
_HOME = STATIC_DIR / "runs.html"
ASSET_DIR = STATIC_DIR / "analysis"


@asynccontextmanager
async def follow_updates(update):
    """Tail shared files while a single-run or multi-run app is serving."""

    async def follow():
        while True:
            await asyncio.sleep(2)
            await run_in_threadpool(update)

    task = asyncio.create_task(follow())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


class BasicAuth:
    """Guard the entire app, including mounted static assets and metric APIs."""

    def __init__(self, app, *, username: str, password: str):
        self.app = app
        self.expected = f"{username}:{password}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or get_route_path(scope) == "/healthz":
            return await self.app(scope, receive, send)
        authorization = dict(scope["headers"]).get(b"authorization", b"")
        scheme, _, token = authorization.partition(b" ")
        try:
            supplied = base64.b64decode(token, validate=True) if scheme.lower() == b"basic" else b""
        except (ValueError, binascii.Error):
            supplied = b""
        if not hmac.compare_digest(supplied, self.expected):
            response = JSONResponse(
                {"detail": "Authentication required"},
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="AvaTrain Dashboard", charset="UTF-8"'},
            )
            return await response(scope, receive, send)
        return await self.app(scope, receive, send)


class DashboardCache:
    """Load dashboards only when visited and cap resident runs with an LRU."""

    def __init__(self, runs_dir: Path, cache_dir: Path, max_runs: int, *, grafana: bool = False, database=None):
        self.runs_dir = runs_dir
        self.cache_dir = cache_dir
        self.max_runs = max_runs
        self.grafana = grafana
        self.database = database
        self.loaded = OrderedDict()
        self.lock = threading.Lock()
        self.catalog = []
        self.catalog_read_at = float("-inf")

    def list_runs(self):
        with self.lock:
            now = time.monotonic()
            if now - self.catalog_read_at >= 5:
                local = discover_runs(self.runs_dir)
                if self.database:
                    try:
                        remote = self.database.list_runs()
                    except Exception as error:
                        logger.warning("Shared dashboard catalog unavailable (%s)", type(error).__name__)
                        if not local and not self.catalog:
                            raise HTTPException(503, "Shared dashboard database unavailable") from None
                        remote = [entry for entry in self.catalog if entry.get("storage") == "postgres"]
                    merged = {entry["id"]: entry for entry in local}
                    merged.update({entry["id"]: entry for entry in remote})
                    local = sorted(merged.values(), key=lambda entry: entry["created_at"], reverse=True)
                self.catalog = local
                self.catalog_read_at = now
            return self.catalog

    def get(self, entry: dict):
        # Defer torch/dashboard imports until the first actual dashboard visit.
        from miles.monitoring.transport.api import make_app
        from miles.monitoring.transport.dump_reader import DumpReader
        from miles.monitoring.transport.store import MetricStore

        with self.lock:
            key = entry["id"]
            if key in self.loaded:
                self.loaded.move_to_end(key)
                return self.loaded[key][0]
            run_metadata = None
            if entry.get("storage") == "postgres":
                from miles.monitoring.transport.postgres_reader import PostgresMetricStore

                if not entry["telemetry_ready"]:
                    raise HTTPException(409, "This run has no dashboard telemetry yet")
                # A private empty artifact directory prevents DB catalog fields
                # from resolving paths in the server's training filesystem.
                run_dir = self.cache_dir / "database" / hashlib.sha256(key.encode()).hexdigest()
                run_dir.mkdir(parents=True, exist_ok=True)
                try:
                    store = PostgresMetricStore(self.database, key, run_dir / "logs/dashboard")
                except Exception as error:
                    logger.warning("Shared telemetry unavailable (%s)", type(error).__name__)
                    raise HTTPException(503, "Shared dashboard telemetry unavailable") from None
                run_metadata = store.metadata
            else:
                run_dir = (self.runs_dir / entry["relative_path"]).resolve()
                telemetry = (run_dir / "logs" / "dashboard").resolve()
                if not run_dir.is_relative_to(self.runs_dir) or not telemetry.is_relative_to(self.runs_dir):
                    raise HTTPException(404, "Run not found")
                if not telemetry.is_dir():
                    raise HTTPException(409, "This run has no dashboard telemetry yet")
                store = MetricStore.load(telemetry)
            # Tensor dumps are not the run logs; keep them opt-in and separate.
            app = make_app(
                store,
                DumpReader(run_dir / "logs", cache_dir=self.cache_dir / key, tensor_lru=1),
                follow=True,
                use_utilization_overview=True,
                run_metadata=run_metadata,
                observability={"grafana_url": f"../../grafana/d/miles-overview/rl-observability?var-run={key}"}
                if self.grafana
                else None,
            )
            self.loaded[key] = (app, store)
            while len(self.loaded) > self.max_runs:
                self.loaded.popitem(last=False)
            return app

    def follow(self):
        with self.lock:
            stores = [store for _, store in self.loaded.values()]
        for store in stores:
            try:
                store.follow()
            except Exception as error:
                logger.warning("Could not refresh a dashboard stream (%s)", type(error).__name__)


class RunDispatcher:
    def __init__(self, cache: DashboardCache):
        self.cache = cache
        self.static_dir = STATIC_DIR
        self.static = StaticFiles(directory=self.static_dir)

    async def __call__(self, scope, receive, send):
        path = get_route_path(scope).lstrip("/")
        key, slash, remainder = path.partition("/")
        try:
            entries = await run_in_threadpool(self.cache.list_runs)
            entry = next((item for item in entries if item["id"] == key), None)
            if entry is None:
                raise HTTPException(404, "Run not found")
            if not slash:
                response = RedirectResponse(scope["root_path"] + "/" + key + "/")
                return await response(scope, receive, send)
            child_scope = dict(scope, root_path=scope["root_path"] + "/" + key)
            # Render the page immediately: heavy data imports/loads only belong to API calls.
            if remainder == "":
                response = FileResponse(self.static_dir / "index.html", headers={"Cache-Control": "no-cache"})
                return await response(child_scope, receive, send)
            if remainder.startswith("static/"):
                static_scope = dict(child_scope, root_path=child_scope["root_path"] + "/static")

                async def static_send(message):
                    if message["type"] == "http.response.start":
                        message = dict(
                            message, headers=list(message.get("headers", [])) + [(b"cache-control", b"no-cache")]
                        )
                    await send(message)

                return await self.static(static_scope, receive, static_send)
            if remainder.startswith("analysis/"):
                if entry.get("storage") == "postgres":
                    raise HTTPException(404, "This run has no published analysis artifact")
                run_dir = (self.cache.runs_dir / entry["relative_path"]).resolve()
                if not run_dir.is_relative_to(self.cache.runs_dir):
                    raise HTTPException(404, "Run not found")
                prefix = scope["root_path"].removesuffix("/runs")
                response = await run_in_threadpool(
                    analysis_artifact,
                    run_dir,
                    remainder.removeprefix("analysis/"),
                    asset_base=prefix + "/analysis-static",
                )
                return await response(child_scope, receive, send)
            app = await run_in_threadpool(self.cache.get, entry)
        except HTTPException as error:
            return await JSONResponse({"detail": error.detail}, status_code=error.status_code)(scope, receive, send)
        await app(child_scope, receive, send)


def make_service(
    runs_dir: Path,
    *,
    cache_dir: Path,
    password: str | None,
    username: str = "admin",
    max_runs: int = 3,
    monitoring: bool = False,
    monitoring_dir: Path | None = None,
    port: int = 7788,
    root_path: str = "",
    postgres: str | None = None,
) -> FastAPI:
    runs_dir = runs_dir.resolve()
    if not runs_dir.is_dir() and not postgres:
        raise ValueError(f"Runs directory does not exist: {runs_dir}")
    if max_runs < 1:
        raise ValueError("max-runs must be positive")
    database = None
    if postgres:
        from miles.monitoring.transport.postgres import DashboardDatabase

        database = DashboardDatabase(postgres)
    cache = DashboardCache(runs_dir, cache_dir, max_runs, grafana=monitoring, database=database)
    exporter = PrometheusExporter(runs_dir, cache.list_runs, database=database)
    stack = None
    if monitoring:
        stack = MonitoringStack(
            monitoring_dir or cache_dir / "monitoring",
            port=port,
            username=username,
            password=password,
            root_path=root_path,
        )

    @asynccontextmanager
    async def lifespan(app):
        if stack:
            await stack.start()
        try:
            async with follow_updates(cache.follow):
                yield
        finally:
            if stack:
                await stack.stop()

    app = FastAPI(title="AvaTrain Dashboard", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    if password:
        app.add_middleware(BasicAuth, username=username, password=password)

    @app.get("/healthz")
    def health():
        if stack and not stack.healthy():
            return JSONResponse({"status": "monitoring unavailable"}, status_code=503)
        return {"status": "ok"}

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics():
        return Response(exporter.render(), headers={"Content-Type": "text/plain; version=0.0.4; charset=utf-8"})

    @app.get("/api/observability")
    def observability():
        return {"grafana": monitoring, "metrics_path": "metrics"}

    @app.get("/")
    def home():
        return FileResponse(_HOME, headers={"Cache-Control": "no-cache"})

    @app.get("/api/runs")
    def runs(q: str = Query("", max_length=128), limit: int = Query(50, ge=1, le=100)):
        items = cache.list_runs()
        search = q.strip().lower()
        selected = [
            item
            for item in items
            if search in f"{item['job_id'] or ''} {item['relative_path']} {item.get('owner', '')}".lower()
        ]
        return {"runs": selected[:limit], "total": len(selected)}

    @app.get("/api/jobs/{job_id}")
    def job_runs(job_id: str):
        try:
            validate_job_id(job_id)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        items = [item for item in cache.list_runs() if item["job_id"] == job_id]
        if not items:
            raise HTTPException(404, "Job is not registered. Associate its run with the register command.")
        return {"job_id": job_id, "runs": items}

    app.mount("/analysis-static", StaticFiles(directory=ASSET_DIR))
    if monitoring:
        app.mount("/grafana", GrafanaProxy("http://127.0.0.1:3000", username))
    app.router.routes.append(Mount("/runs", app=RunDispatcher(cache)))
    return app


def analysis_artifact(run_dir: Path, artifact: str, *, asset_base: str):
    try:
        path = resolve_artifact(run_dir / "analysis" / "rollout_history", artifact)
    except FileNotFoundError as error:
        raise HTTPException(404, "Analysis artifact not found") from error
    headers = {"Cache-Control": "no-cache"}
    if path.suffix != ".html":
        return FileResponse(path, headers=headers)
    source = path.read_text(encoding="utf-8")
    base = html.escape(asset_base.rstrip("/") + "/v1", quote=True)
    assets = (
        f'<link rel="stylesheet" href="{base}/katex/katex.min.css">'
        f'<link rel="stylesheet" href="{base}/analysis_math.css">'
        f'<script defer src="{base}/katex/katex.min.js"></script>'
        f'<script defer src="{base}/katex/auto-render.min.js"></script>'
        f'<script defer src="{base}/analysis_math.js"></script>'
    )
    boundary = re.search(r"</head\s*>|<body(?:\s[^>]*)?>", source, flags=re.IGNORECASE)
    position = boundary.start() if boundary else len(source)
    return HTMLResponse(source[:position] + assets + source[position:], headers=headers)


ASSETS = Path(__file__).parent / "provisioning"
_HOP_HEADERS = {
    "host",
    "connection",
    "keep-alive",
    "transfer-encoding",
    "content-length",
    "upgrade",
    "accept-encoding",
}

TAIL_BYTES = 1024 * 1024
MAX_RUNS = 12
ENGINE_METRICS = frozenset(
    {
        "sglang_num_running_reqs",
        "sglang_num_queue_reqs",
        "sglang_gen_throughput",
        "sglang_token_usage",
        "sglang_mamba_usage",
        "sglang_cache_hit_rate",
    }
)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _tail(path: Path, root: Path) -> list[dict]:
    """Never scan a whole event log or accept a path outside the run tree."""
    if not path.resolve().is_relative_to(root):
        return []
    try:
        with path.open("rb") as file:
            size = file.seek(0, 2)
            offset = max(0, size - TAIL_BYTES)
            file.seek(offset)
            raw = file.read(TAIL_BYTES)
    except FileNotFoundError:
        return []
    if offset:
        raw = raw.partition(b"\n")[2]
    # A partial final line is not a committed sample.
    raw = raw[: raw.rfind(b"\n") + 1]
    rows = []
    for line in raw.splitlines():
        try:
            row = json.loads(line)
            if isinstance(row, dict) and _number(row.get("ts")):
                rows.append(row)
        except (ValueError, UnicodeDecodeError):
            continue
    return rows


def _latest_stream(directory: Path, stream: str, root: Path) -> list[dict]:
    partition = directory / stream
    if not partition.resolve().is_relative_to(root):
        return []
    files = sorted(partition.glob("*.jsonl"))[-2:]
    return [row for path in files or [directory / f"{stream}.jsonl"] for row in _tail(path, root)]


def _label(value) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


class _Metrics:
    def __init__(self):
        self.families = defaultdict(dict)
        self.types = {}

    def add(self, name, value, labels=None, *, kind="gauge"):
        if not _number(value):
            return
        self.types[name] = kind
        # De-duplicate per-engine DP samples and repeated runtime records.
        self.families[name][tuple(sorted((labels or {}).items()))] = float(value)

    def render(self):
        lines = []
        for name, samples in sorted(self.families.items()):
            lines.extend(
                [
                    f"# HELP {name} AvaTrain {name.removeprefix('avatrain_').replace('_', ' ')}.",
                    f"# TYPE {name} {self.types[name]}",
                ]
            )
            for labels, value in sorted(samples.items()):
                suffix = "{" + ",".join(f'{key}="{_label(val)}"' for key, val in labels) + "}" if labels else ""
                lines.append(f"{name}{suffix} {value}")
        return "\n".join(lines) + "\n"


def _runtime_metrics(output: _Metrics, rows: list[dict], run_id: str, now: float, max_age: float):
    latest = {}
    for row in rows:
        metrics = row.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if not _number(value) or not isinstance(key, str):
                continue
            if key.startswith(("runtime/rollout/", "runtime/driver/")):
                _, component, metric = key.split("/", 2)
            elif key.startswith(("train/", "rollout/", "perf/")):
                component, metric = "training", key
            else:
                continue
            pair = (component, metric)
            if pair not in latest or row["ts"] >= latest[pair][0]:
                latest[pair] = row["ts"], value
    observed = {}
    for (component, metric), (ts, value) in latest.items():
        observed[component] = max(observed.get(component, 0), ts)
        labels = dict(run_id=run_id, component=component, metric=metric)
        if 0 <= now - ts <= max_age:
            if metric.endswith("_total") and value >= 0:
                output.add("avatrain_runtime_total", value, labels, kind="counter")
            else:
                output.add("avatrain_runtime_value", value, labels)
    for component, ts in observed.items():
        _freshness(output, run_id, component, ts, now, max_age)


def _freshness(output, run_id, component, ts, now, max_age):
    labels = dict(run_id=run_id, component=component)
    output.add("avatrain_telemetry_timestamp_seconds", ts, labels)
    output.add("avatrain_telemetry_fresh", int(0 <= now - ts <= max_age), labels)


def _hardware_metrics(output, directory, root, run_id, now, max_age):
    _hardware_rows(
        output,
        _latest_stream(directory, "gpu_util", root),
        _latest_stream(directory, "engine_series", root),
        run_id,
        now,
        max_age,
    )


def _hardware_rows(output, gpu_rows, engine_rows, run_id, now, max_age):
    gpu = {}
    for row in gpu_rows:
        key = (str(row.get("node", "unknown")), str(row.get("gpu", "unknown")))
        if key not in gpu or row["ts"] >= gpu[key]["ts"]:
            gpu[key] = row
    for (node, device), row in gpu.items():
        if not 0 <= now - row["ts"] <= max_age:
            continue
        labels = dict(run_id=run_id, node=node, gpu=device)
        for field, name, scale in (
            ("util", "utilization_ratio", 0.01),
            ("mem_mb", "memory_bytes", 1024**2),
            ("power_w", "power_watts", 1),
        ):
            if _number(row.get(field)):
                output.add(f"avatrain_gpu_{name}", row[field] * scale, labels)
    if gpu:
        _freshness(output, run_id, "gpu", max(row["ts"] for row in gpu.values()), now, max_age)
    engines = {}
    for row in engine_rows:
        if row.get("metric") not in ENGINE_METRICS or not _number(row.get("value")):
            continue
        # Keep DP identities: request counts must not be silently overwritten.
        dp = str(row.get("labels", {}).get("dp_rank", "0"))
        key = (str(row.get("addr", "unknown")), dp, row["metric"])
        if key not in engines or row["ts"] >= engines[key]["ts"]:
            engines[key] = row
    for (engine, dp, metric), row in engines.items():
        if 0 <= now - row["ts"] <= max_age:
            output.add(
                "avatrain_engine_value", row["value"], dict(run_id=run_id, engine=engine, dp_rank=dp, metric=metric)
            )
    if engines:
        _freshness(output, run_id, "engine", max(row["ts"] for row in engines.values()), now, max_age)


class PrometheusExporter:
    """Cache bounded file/database tails; scraping never opens tensor dumps."""

    def __init__(self, root: Path, catalog, *, database=None):
        self.root = root.resolve()
        self.catalog = catalog
        self.database = database
        self.lock = threading.Lock()
        self.cached_at = float("-inf")
        self.cached = ""
        self.errors = 0

    def render(self, *, now=None):
        with self.lock:
            if now is None and time.monotonic() - self.cached_at < 5:
                return self.cached
            wall_time = time.time() if now is None else now
            output = _Metrics()
            entries = sorted(
                (entry for entry in self.catalog() if entry["telemetry_ready"]),
                key=lambda entry: entry["created_at"],
                reverse=True,
            )[:MAX_RUNS]
            for entry in entries:
                if entry.get("storage") == "postgres":
                    try:
                        self._database_metrics(output, entry, wall_time)
                    except Exception:
                        self.errors += 1
                    continue
                directory = self.root / entry["relative_path"] / "logs" / "dashboard"
                if not directory.resolve().is_relative_to(self.root):
                    continue
                try:
                    meta_path = directory / "meta.json"
                    if not meta_path.resolve().is_relative_to(self.root):
                        continue
                    metadata = json.loads(meta_path.read_text())
                    interval = metadata.get("args", {}).get("perf_monitor_interval", 10)
                    max_age = max(30, 3 * interval) if _number(interval) else 30
                    output.add(
                        "avatrain_run_info",
                        1,
                        dict(
                            run_id=entry["id"],
                            run_name=entry["name"],
                            job_id=entry["job_id"] or "",
                            state=entry["state"],
                        ),
                    )
                    _runtime_metrics(
                        output, _tail(directory / "metrics.jsonl", self.root), entry["id"], wall_time, max_age
                    )
                    _hardware_metrics(output, directory, self.root, entry["id"], wall_time, max_age)
                except (OSError, ValueError, TypeError, KeyError):
                    self.errors += 1
            output.add("avatrain_exporter_read_errors_total", self.errors, kind="counter")
            output.add("avatrain_exporter_runs", len(entries))
            self.cached = output.render()
            self.cached_at = time.monotonic()
            return self.cached

    def _database_metrics(self, output, entry, now):
        metadata, files = self.database.describe(entry["id"])
        interval = metadata.get("meta", {}).get("args", {}).get("perf_monitor_interval", 10)
        max_age = max(30, 3 * interval) if _number(interval) else 30
        output.add(
            "avatrain_run_info",
            1,
            dict(run_id=entry["id"], run_name=entry["name"], job_id=entry["job_id"] or "", state=entry["state"]),
        )
        rows = {
            stream: self.database.tail(entry["id"], files, stream)
            for stream in ("metrics", "gpu_util", "engine_series")
        }
        _runtime_metrics(output, rows["metrics"], entry["id"], now, max_age)
        _hardware_rows(output, rows["gpu_util"], rows["engine_series"], entry["id"], now, max_age)


def _write_yaml(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    path.chmod(0o600)


def _postgres_source(url):
    parsed = urlsplit(url)
    if parsed.scheme not in ("postgres", "postgresql") or not parsed.hostname or not parsed.username:
        raise ValueError("AVATRAIN_GRAFANA_POSTGRES_URL must be a PostgreSQL URL for a read-only account")
    ssl = parse_qs(parsed.query).get("sslmode", ["require"])[0]
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    return dict(
        name="AvaCore rollouts (read only)",
        uid="ava-rollouts",
        type="postgres",
        access="proxy",
        url=f"{host}:{parsed.port or 5432}",
        user=unquote(parsed.username),
        jsonData=dict(
            database=unquote(parsed.path.lstrip("/")),
            sslmode=ssl,
            postgresVersion=1500,
            maxOpenConns=3,
            maxIdleConns=1,
            connMaxLifetime=300,
        ),
        secureJsonData=dict(password=unquote(parsed.password or "")),
        editable=False,
    )


class MonitoringStack:
    """The single dashboard replica owns persistent TSDB/SQLite state and child processes."""

    def __init__(self, directory: Path, *, port: int, username: str, password: str | None, root_path=""):
        self.directory = directory
        self.port = port
        self.username = username
        self.password = password
        self.root_path = root_path.rstrip("/")
        self.processes = []
        self.logs = []

    def _configure(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        provision = self.directory / "provisioning"
        for folder in (
            provision / "plugins",
            provision / "alerting",
            self.directory / "plugins",
            self.directory / "logs",
        ):
            folder.mkdir(parents=True, exist_ok=True)
        config = {"global": dict(scrape_interval="15s", evaluation_interval="15s")}
        scrape = dict(job_name="miles", static_configs=[dict(targets=[f"127.0.0.1:{self.port}"])])
        if self.password:
            secret = self.directory / "scrape-password"
            secret.write_text(self.password)
            secret.chmod(0o600)
            scrape["basic_auth"] = dict(username=self.username, password_file=str(secret))
        config.update(scrape_configs=[scrape], rule_files=[str(ASSETS / "alerts.yml")])
        _write_yaml(self.directory / "prometheus.yml", config)
        sources = [
            dict(
                name="Miles Prometheus",
                uid="miles-prometheus",
                type="prometheus",
                access="proxy",
                url="http://127.0.0.1:9090",
                isDefault=True,
                editable=False,
                jsonData=dict(httpMethod="POST", timeInterval="15s"),
            )
        ]
        postgres = os.getenv("AVATRAIN_GRAFANA_POSTGRES_URL")
        if postgres:
            sources.append(_postgres_source(postgres))
        _write_yaml(provision / "datasources" / "miles.yml", dict(apiVersion=1, datasources=sources))
        _write_yaml(
            provision / "dashboards" / "miles.yml",
            dict(
                apiVersion=1,
                providers=[
                    dict(
                        name="Miles",
                        orgId=1,
                        folder="Miles",
                        type="file",
                        disableDeletion=True,
                        allowUiUpdates=False,
                        options=dict(path=str(self.directory / "dashboards")),
                    ),
                ],
            ),
        )
        (self.directory / "dashboards").mkdir(exist_ok=True)
        shutil.copyfile(ASSETS / "overview.json", self.directory / "dashboards" / "overview.json")
        if postgres:
            shutil.copyfile(ASSETS / "rollouts.json", self.directory / "dashboards" / "rollouts.json")
        else:
            (self.directory / "dashboards" / "rollouts.json").unlink(missing_ok=True)
        ini = self.directory / "grafana.ini"
        ini.write_text(f"""[server]
http_addr = 127.0.0.1
http_port = 3000
root_url = http://localhost:3000{self.root_path}/grafana/
serve_from_sub_path = true
[paths]
data = {self.directory / "grafana"}
logs = {self.directory / "logs"}
plugins = {self.directory / "plugins"}
provisioning = {provision}
[auth]
disable_login_form = true
[auth.basic]
enabled = false
[security]
allow_embedding = true
[auth.proxy]
enabled = true
header_name = X-WEBAUTH-USER
header_property = username
auto_sign_up = true
whitelist = 127.0.0.1
headers = Role:X-WEBAUTH-ROLE
[users]
allow_sign_up = false
[analytics]
reporting_enabled = false
check_for_updates = false
check_for_plugin_updates = false
[dashboards]
default_home_dashboard_path = {self.directory / "dashboards" / "overview.json"}
[live]
max_connections = 0
[log]
mode = console
level = warn
""")
        return ini

    async def start(self):
        ini = self._configure()
        env = {**os.environ, "GF_SECURITY_ADMIN_PASSWORD": secrets.token_urlsafe(32)}
        commands = [
            [
                os.getenv("AVATRAIN_PROMETHEUS_BINARY", "/opt/monitoring/prometheus"),
                f"--config.file={self.directory / 'prometheus.yml'}",
                f"--storage.tsdb.path={self.directory / 'prometheus'}",
                "--storage.tsdb.retention.time=15d",
                "--storage.tsdb.retention.size=2GB",
                "--web.listen-address=127.0.0.1:9090",
            ],
            [
                os.getenv("AVATRAIN_GRAFANA_BINARY", "/opt/monitoring/grafana/bin/grafana"),
                "server",
                "--homepath=/opt/monitoring/grafana",
                f"--config={ini}",
            ],
        ]
        try:
            for name, command in zip(("prometheus", "grafana"), commands, strict=False):
                log = (self.directory / f"{name}.log").open("a")
                self.logs.append(log)
                self.processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
            async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline:
                    if any(p.poll() is not None for p in self.processes):
                        raise RuntimeError("Monitoring process exited; inspect monitoring service logs")
                    try:
                        checks = await asyncio.gather(
                            client.get("http://127.0.0.1:9090/-/ready"),
                            client.get("http://127.0.0.1:3000" + self.root_path + "/grafana/api/health"),
                        )
                        if all(r.status_code == 200 for r in checks):
                            return
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(0.5)
            raise RuntimeError("Monitoring startup did not become ready")
        except BaseException:
            await self.stop()
            raise

    def healthy(self):
        return len(self.processes) == 2 and all(p.poll() is None for p in self.processes)

    async def stop(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
        for process in self.processes:
            try:
                await asyncio.to_thread(process.wait, timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait)
        for log in self.logs:
            log.close()
        self.processes.clear()
        self.logs.clear()


class GrafanaProxy:
    """Expose Grafana through the already authenticated Miles entrypoint."""

    def __init__(self, upstream: str, username: str):
        self.upstream = upstream.rstrip("/")
        self.username = username
        self.local = urlsplit(upstream).hostname in ("localhost", "127.0.0.1")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await Response(status_code=404)(scope, receive, send)
        path = get_route_path(scope)
        query = scope.get("query_string", b"").decode("ascii")
        url = self.upstream + "/grafana" + path + ("?" + query if query else "")
        # Grafana includes UTF-8 dashboard titles in request headers. Preserve
        # their original bytes rather than asking httpx to re-encode as ASCII.
        headers = [
            (key, value)
            for key, value in scope["headers"]
            if key.decode() not in _HOP_HEADERS and not key.lower().startswith(b"x-webauth-")
        ]
        if self.local:
            headers = [(key, value) for key, value in headers if key.lower() != b"authorization"]
            headers.extend([(b"x-webauth-user", self.username.encode()), (b"x-webauth-role", b"Admin")])
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if not message.get("more_body"):
                break
        try:
            async with httpx.AsyncClient(trust_env=False, timeout=30, follow_redirects=False) as client:
                upstream = await client.request(scope["method"], url, headers=headers, content=bytes(body))
            response = Response(upstream.content, status_code=upstream.status_code)
            response.raw_headers = []
            for key, value in upstream.headers.raw:
                if key.decode().lower() in _HOP_HEADERS or key.lower() in (b"content-encoding", b"x-frame-options"):
                    continue
                if key.lower() == b"location" and value.startswith(self.upstream.encode()):
                    value = value[len(self.upstream.encode()) :]
                response.raw_headers.append((key.lower(), value))
            response.raw_headers.append((b"x-frame-options", b"SAMEORIGIN"))
            response.raw_headers.append((b"content-length", str(len(upstream.content)).encode()))
        except httpx.HTTPError:
            response = JSONResponse({"detail": "Grafana is temporarily unavailable"}, status_code=502)
        await response(scope, receive, send)
