"""Export the native dashboard timeline to one HTML file and publish it to W&B."""

import hashlib
import html
import json
import logging
import re
import threading
from dataclasses import asdict
from pathlib import Path

from miles.dashboard.advisory import compute_advisories, mfu_summary
from miles.dashboard.store import MetricStore

logger = logging.getLogger(__name__)
STATIC = Path(__file__).with_name("static")
ENGINE_METRICS = ("sglang_num_running_reqs", "sglang_gen_throughput", "sglang_token_usage", "sglang_cache_hit_rate")


def snapshot(store: MetricStore) -> dict:
    """Use the same queries as the HTTP dashboard, without running a server."""
    bounds = store.time_range()
    if bounds is None:
        raise ValueError("Dashboard has no telemetry yet")
    t0, t1 = bounds
    t0 = max(t0, t1 - MetricStore.MAX_WINDOW_S)
    window = dict(t0=t0, t1=max(t0 + 1, t1))
    mfu = mfu_summary(store)
    engines = {name: store.engine_series(name, **window, max_points=4000) for name in ENGINE_METRICS}
    return dict(
        meta=dict(
            mode="static",
            start_ts=store.meta.start_ts if store.meta else t0,
            time_range=[window["t0"], window["t1"]],
            data_buffer_length=store.latest_data_buffer_length(),
            engine_interval_s=store.meta.args.get("dashboard_sglang_scrape_interval", 2) if store.meta else 2,
            capabilities=dict(
                has_timeline=True,
                has_engine_series=any(engines.values()),
                use_utilization_overview=True,
                max_window_s=MetricStore.MAX_WINDOW_S,
            ),
        ),
        topology=dict(lanes=store.lane_index(), windows=store.topology_windows()),
        phases=store.phases_by_lane(**window),
        gpu=store.gpu_series(**window, max_points=4000),
        processes=store.gpu_processes(**window),
        fleet=store.fleet(**window, x_buckets=1200),
        bubbles=store.bubbles(),
        engines=engines,
        data_pool=store.data_buffer_series(**window),
        advisory=dict(mfu=mfu, advisories=[asdict(a) for a in compute_advisories(store, **window, mfu=mfu)]),
        outliers={
            name: store.outliers(name, **window, top_k=64) for name in ("lowest_util", "slowest_phase:update_weights")
        },
    )


def render_report(store: MetricStore, *, title: str, run_url: str = "") -> str:
    modules = []
    for name in ("app", "api", "charts", "carpet", "fleet", "data_pool", "views_timeline"):
        source = (STATIC / ("report_api.js" if name == "api" else f"{name}.js")).read_text()
        if name == "app":
            # Reuse the native UI helpers; the standalone page has no SPA routing.
            source = source.split("function parseRoute()", 1)[0]
            source = re.sub(r"^import .*?;\n", "", source, flags=re.MULTILINE)
        source = re.sub(
            r'import\s*\{([^}]+)\}\s*from\s*"\./([^\"]+)\.js";',
            lambda m: f'const {{{m[1]}}} = modules["{m[2]}"];',
            source,
        )
        exports = re.findall(r"export\s+(?:async\s+)?(?:function|const|let)\s+(\w+)", source)
        source = re.sub(r"\bexport\s+", "", source)
        modules.append(f'modules["{name}"] = (() => {{\n{source}\nreturn {{{",".join(exports)}}};\n}})();')
    data = json.dumps(snapshot(store), separators=(",", ":"), allow_nan=False)
    script = "const REPORT=" + data + ";const modules={};\n" + "\n".join(modules)
    script += '\nmodules.views_timeline.renderTimeline(document.getElementById("view"), REPORT.meta, {}).catch(e=>{document.getElementById("view").textContent=String(e);console.error(e)});'
    script = script.replace("</", "<\\/")
    style = re.sub(r"^@import[^\n]*$", "", (STATIC / "style.css").read_text(), flags=re.MULTILINE)
    link = f'<a href="{html.escape(run_url, quote=True)}">W&amp;B run</a>' if run_url else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{style}</style></head><body>
<header id="topbar" style="padding:12px 20px;color:var(--topbar-text)"><strong>{html.escape(title)}</strong>{link}</header>
<p class="muted" style="padding:0 20px">Miles timeline snapshot · latest {MetricStore.MAX_WINDOW_S / 3600:g} hours · wheel to zoom, drag to pan. Refresh to load a newer snapshot. Overview bins and outlier ordering are fixed at export time.</p>
<main id="view">Loading…</main><script>{script}</script></body></html>"""


class ReportPublisher:
    """One driver thread reads the native files; no second collector or telemetry archive."""

    def __init__(self, dashboard_dir: str):
        self.dashboard_dir = dashboard_dir
        self.media_path = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="wandb-rl-report", daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.wait(60):
            self.publish()

    def close(self):
        self._stop.set()
        self._thread.join()
        self.publish()

    def publish(self):
        try:
            self._publish()
        except Exception:
            logger.warning("RL report upload failed; training continues", exc_info=True)

    def _publish(self):
        # W&B is optional and only used when both existing tracking backends are enabled.
        import wandb

        if wandb.run is None:
            return
        store = MetricStore.load(self.dashboard_dir)
        if store.time_range() is None:
            return
        path = Path(wandb.run.dir) / "rl-report.html"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(render_report(store, title=wandb.run.name or wandb.run.id, run_url=wandb.run.url or ""))
        temporary.replace(path)
        wandb.run.save(str(path), base_path=wandb.run.dir, policy="live")
        if self.media_path is None:
            media = wandb.Html(str(path), inject=False)
            wandb.run.log({"rl_report": media})
            self.media_path = media.to_json(wandb.run)["path"]
        media_file = Path(wandb.run.dir) / self.media_path
        media_file.write_bytes(path.read_bytes())
        wandb.run.save(str(media_file), base_path=wandb.run.dir, policy="live")
        wandb.run.summary["rl_report"] = {
            "_type": "html-file",
            "path": self.media_path,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }
        if wandb.run.url:
            url = f"{wandb.run.url.split('?')[0].rstrip('/')}/panel/rlreportpanel?nw=rlreport"
            wandb.run.summary["rl_report_url"] = url
            if "[RL timeline]" not in (wandb.run.notes or ""):
                wandb.run.notes = (wandb.run.notes or "") + f"\n\n[RL timeline]({url})"
