"""Self-contained, per-run report using the original Miles timeline template."""

import html
import json
import math
import re
from pathlib import Path

STATIC = Path(__file__).with_name("static")
DASHBOARD_STATIC = Path(__file__).resolve().parents[2] / "dashboard" / "static"
PERFORMANCE_KEYS = {"perf/actor_train_mfu", "perf/mfu_peak_tflops"}
# Keep only values consumed by the report. Runtime diagnostics no longer become
# thousands of independent W&B history series.
REPORT_SUFFIXES = (
    "/gpu_util_pct",
    "/gpu_memory_gb",
    "/running_requests",
    "/generated_tokens_per_s",
    "/kv_usage_mean",
    "/gpu_util_mean_pct",
    "/gpu_memory_max_gb",
    "/coverage_fraction",
    "/queued_groups",
    "/capacity_groups",
    "/current_weight_version",
    "/waiting",
    "/oldest_wait_seconds",
    "/completed_rollouts",
    "/target_rollouts",
    "/fraction",
    "/ready_groups",
    "/target_groups",
    "/progress_fraction",
)


def report_metrics(metrics):
    return {
        k: v
        for k, v in metrics.items()
        if isinstance(v, (int, float))
        and math.isfinite(v)
        and (
            k in PERFORMANCE_KEYS
            or k == "runtime/time_s"
            or k.startswith("runtime/")
            and (k.endswith(REPORT_SUFFIXES) or "/versions/" in k)
        )
    }


def report_payload(records, interval):
    series, events = {}, []
    for record in sorted(records, key=lambda r: r.get("ts", r.get("t0", 0))):
        if record["kind"] != "metrics":
            events.append(record)
            continue
        for key, value in report_metrics(record["metrics"]).items():
            column = series.setdefault(key, {"ts": [], "y": []})
            column["ts"].append(round(record["ts"], 3))
            column["y"].append(value)
    return {"series": series, "events": events, "interval": interval}


def render_report(records, *, title, run_url="", interval=10):
    """Embed data, styles and original modules; no server, CDN or credentials."""
    modules = []
    for name in ("app", "api", "charts", "async_flow", "carpet", "fleet", "views_timeline"):
        assets = DASHBOARD_STATIC if name == "carpet" else STATIC
        source = (assets / f"{name}.js").read_text()
        source = re.sub(
            r'import\s*\{([^}]+)\}\s*from\s*"\./([^\"]+)\.js";',
            lambda m: f'const {{{m[1]}}} = modules["{m[2]}"];',
            source,
        )
        exports = re.findall(r"export\s+(?:async\s+)?(?:function|const|let)\s+(\w+)", source)
        source = re.sub(r"\bexport\s+", "", source)
        modules.append(f'modules["{name}"] = (() => {{\n{source}\nreturn {{{",".join(exports)}}};\n}})();')
    data = json.dumps(report_payload(records, interval), separators=(",", ":"), allow_nan=False)
    script = "const REPORT=" + data + "; const modules={};\n" + "\n".join(modules)
    script += '\nmodules.views_timeline.renderTimeline(document.getElementById("view"), modules.api.meta, {}).catch(e=>{document.getElementById("view").textContent=String(e);console.error(e)});'
    script = script.replace("</", "<\\/")
    style = re.sub(r"^@import[^;]+;", "", (DASHBOARD_STATIC / "style.css").read_text(), flags=re.MULTILINE)
    style += (STATIC / "style.css").read_text()
    link = f'<a href="{html.escape(run_url, quote=True)}">W&amp;B run ↗</a>' if run_url else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{style}</style></head><body>
<header id="topbar"><strong>{html.escape(title)}</strong>{link}</header>
<p style="padding:0 20px" class="muted">Independent run snapshot · Miles timeline template · wheel to zoom, drag to pan, double-click to reset. Missing telemetry remains blank. Refresh this page to load a newer snapshot.</p>
<p class="muted" style="padding:0 20px">Latest reported values (hover for sample time)</p><div id="snapshot-summary" class="statgrid" style="padding:0 20px"></div><main id="view">Loading…</main><script>{script}</script></body></html>"""


def report_url(run_url):
    """Project-scoped saved view installed by the Miles report workspace command."""
    return f"{run_url.split('?')[0].rstrip('/')}/panel/rlreportpanel?nw=rlreport"
