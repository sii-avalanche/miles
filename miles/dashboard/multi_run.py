"""Read-only multi-run entry for the current Miles dashboard."""
import asyncio
import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from miles.dashboard.dump_reader import DumpReader
from miles.dashboard.server import make_app
from miles.dashboard.store import MetricStore

ROOT = Path(os.environ.get('MILES_RUNS_DIR', 'runs')).resolve()
PATTERNS = ('*/run_*/logs/dashboard/meta.json', '*/run_*/dashboard/meta.json',
            '*/run_*/details/dashboard/meta.json', '*/details/dashboard/meta.json',
            '*/details/*/dashboard/meta.json')
lock = threading.RLock()
cache = OrderedDict()
catalogue = {}
catalogue_time = 0


def discover():
    global catalogue, catalogue_time
    with lock:
        if time.monotonic() - catalogue_time < 15:
            return catalogue
        found = {}
        for pattern in PATTERNS:
            for meta in ROOT.glob(pattern):
                directory = meta.parent.parent.resolve()
                if not directory.is_relative_to(ROOT):
                    continue
                try:
                    data = json.loads(meta.read_text())
                except (OSError, ValueError):
                    continue
                relative = str(directory.relative_to(ROOT))
                key = hashlib.sha256(relative.encode()).hexdigest()[:20]
                found[key] = {'id': key, 'name': data.get('run_name') or directory.name,
                              'path': relative, 'start_ts': data.get('start_ts') or 0}
        catalogue, catalogue_time = found, time.monotonic()
        return catalogue


def get_app(key):
    with lock:
        item = discover().get(key)
        if item is None:
            return None
        if key not in cache:
            root = ROOT / item['path']
            store = MetricStore.load(root / 'dashboard')
            reader = DumpReader(root, cache_dir=Path('/tmp/miles-dashboard-cache') / key, tensor_lru=1)
            cache[key] = (make_app(store, reader, follow=True), store, 0)
            while len(cache) > 4:
                cache.popitem(last=False)
        app, store, updated = cache[key]
        if time.monotonic() - updated >= 2:
            store.follow()
            cache[key] = (app, store, time.monotonic())
        cache.move_to_end(key)
        return app


index = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

@index.get('/healthz')
def health():
    return {'status': 'ok'}

@index.get('/')
def home():
    return FileResponse(Path(__file__).with_name('runs.html'), headers={'Cache-Control': 'no-cache'})

@index.get('/api/runs')
def runs():
    available = sorted(discover().values(), key=lambda row: row['start_ts'], reverse=True)
    # Legacy runs may lack telemetry entirely; list them without fabricating metrics.
    missing = []
    known = {str((ROOT / r['path']).resolve()) for r in available}
    for config in ROOT.glob('*/run_*/config.toml'):
        import tomllib
        try:
            settings = tomllib.loads(config.read_text())
            paths = [v.get('dump_details') for v in settings.values() if isinstance(v, dict)]
            if any(p and str(Path(p).resolve()) in known for p in paths):
                continue
            if any(str(p.resolve()) in known for p in (config.parent, config.parent/'logs', config.parent/'details')):
                continue
            missing.append(str(config.parent.relative_to(ROOT)))
        except (OSError, ValueError):
            continue
    return {'runs': available, 'without_telemetry': sorted(missing, reverse=True)}


async def app(scope, receive, send):
    path = scope.get('path', '')
    if scope['type'] == 'http' and path.startswith('/runs/'):
        parts = path.split('/', 3)
        selected = await asyncio.to_thread(get_app, parts[2])
        if selected is None:
            return await JSONResponse({'detail': 'Run not found'}, status_code=404)(scope, receive, send)
        if len(parts) == 3:
            return await RedirectResponse(path + '/')(scope, receive, send)
        child = dict(scope, path='/' + parts[3], raw_path=('/' + parts[3]).encode())
        return await selected(child, receive, send)
    return await index(scope, receive, send)


def main():
    import argparse

    global ROOT
    parser = argparse.ArgumentParser(description="Serve dashboards for runs in a directory")
    parser.add_argument("--runs", type=Path, default=ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7788)
    args = parser.parse_args()
    ROOT = args.runs.resolve()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
