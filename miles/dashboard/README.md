# miles dashboard

Training-dynamics and efficiency dashboards for miles runs, built on top of
`--dump-details` plus a live telemetry collector.

User-facing documentation lives at `docs/user-guide/dashboard.md`; this file is the
implementer's summary.

## Collect

```bash
python train.py ... \
    --dump-details /path/to/dump \
    --use-miles-dashboard \
    --use-rollout-entropy        # optional: per-token entropy in the token view
```

`--use-miles-dashboard` adds a named collector actor on the driver node, one
NVML sampler per GPU node, per-rank phase interval sinks on the existing
`Timer` instrumentation, and an sglang engine-metric scraper. Everything is
appended under `{dump-details}/dashboard/` (append-only JSONL). Overhead on
the training path is a few milliseconds per step; all pushes are
fire-and-forget and a dead collector never affects training.

`--dump-details` alone (no `--use-miles-dashboard`) still writes the rollout
`.pt`, a per-token column mirror (`dashboard_columns/`, so the token view
never full-loads the `.pt`), and — for session / multi-turn runs — a
raw-conversation sidecar (`trajectory/`, for the conversation view).

## View

```bash
pip install fastapi uvicorn polars   # already present in the training image
python -m miles.dashboard.serve --dump-details /path/to/dump [--follow] [--port 7788]
```

Works on any machine that can see the directory (login node over NFS, the
training node itself); `--follow` tails a still-running job, typically over an
SSH port-forward. `--use-utilization-overview` forces the fleet overview
(auto-enabled above 64 lanes). Runs recorded *without* `--use-miles-dashboard`
still get the training-dynamics views; only the timeline is absent (metrics
fall back to dump-derived `dump/*` aggregates).

Views:

- **Overview** — default run page: whole-fleet GPU utilization and phase composition,
  then generation, reward, recording, batch collection, trainer waits, training,
  weight sync and Data Pool on one wall-clock window.
- **GPU details** — per-GPU curves, memory, engine overlays, MFU, lane selection,
  utilization distribution and outlier quick-picks. Also accepts legacy `#/timeline`.
- **Data & samples** — pool versions, workload composition, batch selection,
  rejections and retries; links into rollout samples and HTML / Trace artifacts.
- **Training metrics** — learning curves for the current run, with raw metric
  families retained for inspection. W&B links are for comparing experiments.
- **Monitoring & alerts** — optional same-origin embedded Grafana, scoped to the
  run and the shared time window; explicit fallback when monitoring is absent.

The wall-clock window persists across these views in the current document.
Grafana receives the window on entry or **Apply shared time window**; changes
inside Grafana do not update Miles. Legacy `#/runtime` opens Overview.

- **Rollouts** — per-step trajectory table and scatter, GRPO group degeneracy
  (`zero_std`), average weight-version staleness, eval tab. Opens on the newest
  step that has samples, skipping one still being dumped; a step named in the
  URL is always shown as asked.
- **sample view** — a `conversation` tab (role-tagged turns with thinking /
  tool calls, from the trajectory sidecar) and a lazily-loaded `tokens` tab
  (the whole sequence in one scrollable metric-colored strip, opened at the
  first generated token, + a selectable metric-vs-position chart, loss-masked
  regions dimmed).

## Develop

```bash
python -m miles.dashboard.serve --demo    # generated demo data, no cluster needed
python -m pytest tests/fast/dashboard/ -q
MILES_DASHBOARD_REALDATA_DIR=/path/to/real/dump python -m pytest tests/fast/dashboard/ -q
```

Architecture:

```
producers (Timer sinks / rollout hooks / NVML samplers / sglang scraper)
    -> DashboardCollector (named actor, driver node)          [live JSONL streams]
dump .pt + dashboard_columns/*.parquet + trajectory/*.jsonl   [written by training]
    -> serve.py: MetricStore + DumpReader -> FastAPI -> static SPA
```

Runtime instrumentation lives in `miles/utils/perf_monitor.py`, including its
private JSONL and host-resource helpers. Rollout MoE summaries are part of
`miles/ray/rollout/metrics.py`; the executor only invokes the logging boundary.
Buffer queue telemetry uses one cached queue and one set of workload/version
counts, from which fleet totals are derived when sampled.
