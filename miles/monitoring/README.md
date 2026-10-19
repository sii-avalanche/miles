# Monitoring data flow

Business code records observations through `collection.runtime.get_monitor()` or
`collection.hooks`. Monitoring never owns scheduling, filtering, the training
loss, or weight synchronization.

```text
Trainer / Rollout / Buffer / AvaCore / Host / NVML / SGLang
    -> collection: observations, phase/event hooks and raw record schemas
    -> aggregation: bounded state, distributions and pure query transforms
    -> transport: existing Tracking, Ray collector, JSONL/Postgres and HTTP
    -> frontend: shared Overview, Async Flow, GPU timeline and other pages
```

The arrows describe data flow, not a requirement to pass every sample through
four objects. Raw GPU/SGLang records are persisted as-is; query-time aggregation
receives windowed records or frames. `RuntimeMonitor` wires its statistics and
writer together. Aggregation imports neither transport implementations nor
frontend code. Diagnostic rules use a small read-only query protocol, which both
local and shared stores implement.

## Where to look

| Layer | Files | Responsibility |
| --- | --- | --- |
| collection | `runtime.py` | Process monitor registry; lightweight increment, observe, wait, phase and provider APIs. |
| collection | `buffer.py`, `rollout.py` | Observe buffer decisions and producer/batch progress; own probe lifecycle, not queue or task scheduling. |
| collection | `hooks.py`, `records.py` | Existing Timer/trajectory/topology hooks and shared observation schemas. |
| collection | `gpu_sampler.py`, `sglang_scraper.py`, `host.py` | Reused NVML/AMD and SGLang sampling; host/cgroup and AvaCore outbox/pool gauges. |
| aggregation | `runtime.py` | Counters, gauges, duration windows, rates, open waits, bounded events, runtime status and group-history reconciliation. |
| aggregation | `buffer.py`, `workload.py` | Queue version distributions and per-step selection statistics; workload progress and optional MoE expert-load math. |
| aggregation | `fleet.py`, `timeline.py` | Fleet latest-value state; GPU coverage, lane/phase assembly, percentiles, engine derivatives, heatmaps and downsampling. |
| aggregation | `diagnostics.py` | Existing stall, rollout, engine and MFU advisory rules against read-only queries. |
| transport | `runtime.py`, `backend.py`, `collector.py` | Existing background writer, Tracking/Ray delivery and collector lifecycle; statistics are delegated to aggregation. |
| transport | `store.py`, `dump_reader.py` | JSONL partition/cache/query access and existing tensor/Parquet readers. Pure timeline calculations are delegated to aggregation. |
| transport | `postgres.py`, `postgres_reader.py`, `publisher.py` | Optional shared storage and incremental publication. Core probes do not import database clients. |
| transport | `api.py`, `service.py`, `catalog.py` | Existing HTTP contracts, mounted multi-run service, authentication, discovery and Prometheus/Grafana integration. |
| transport | `group_history.py`, `status.py` | File-backed metadata/artifact lookup; runtime calculations are delegated to aggregation. |
| transport | `config.py`, `cli.py`, `logging_utils.py`, `provisioning/` | Existing flags, launch commands, throttled delivery warnings and monitoring service configuration. |
| frontend | `static/` | All original pages and shared resources, including Metrics, Rollout, Tokens, Monitoring, Overview and Async Flow. |

## Concrete paths

* `train_async.py` records `monitor.phase("train")`; `RuntimeStatistics` maintains
  waits/durations and the event queue; `RuntimeWriter` writes JSONL and calls the
  existing Tracking backend with `step_key="runtime/time_s"`. Dashboard delivery
  goes through the same Ray collector and `MetricStore` used by other metrics.
* `DefaultDataBuffer` calls `BufferTelemetry` around the existing FIFO/filter
  operations. `BufferStatistics` owns queued version/workload counts;
  `BufferWindow` owns the existing per-step statistics and reset. The staleness
  decision still uses the caller's `current_version`. Telemetry's published
  version is only a reporting reference.
* `FullyAsyncRolloutFn` reports submission/completion and batch observations to
  `RolloutTelemetry`. The helper uses the existing heartbeat task; the scheduler
  still owns capacity, submission, cancellation and retries.
* GPU/SGLang collectors submit raw records to `DashboardCollector`. Its existing
  ingest lock guards `FleetStatistics`. The existing Prometheus forwarding path
  and the driver's fleet gauges reuse that state instead of adding samplers.
* `MetricStore` reads a time window, passes observations to `aggregation.timeline`,
  and returns the result through `transport.api`. `frontend/static/views_overview.js`
  shares one refresh/window with `async_flow.js` and the GPU components.

## Compatibility and limits

* Old `miles.dashboard.*` and `miles.utils.perf_monitor` imports are thin aliases
  to the same implementation objects, including process-local collector handles.
* `python -m miles.dashboard`, `python -m miles.dashboard.serve`, existing flags,
  JSONL schemas, API routes, root-path/mounted URLs, and metric names are retained.
  Runtime metrics use `runtime/time_s`; rollout/MoE metrics use `rollout/step`.
* The old checkout asset directories are symlinks to the sole resource copies.
  Wheels package the resources under `miles.monitoring.frontend`; HTTP serves
  that location. Legacy runtime/timeline view exports delegate to Overview.
* No monitoring framework, production dependency, lock or background thread was
  added. Duration history remains bounded to 256 observations per tracked stage,
  and the runtime event queue remains bounded to 2048 events.
* MoE and configured workload diagnostics remain opt-in. Postgres publication
  remains optional; core monitor imports do not load Torch, Ray, NumPy, Polars,
  W&B or Psycopg.

## Refactor verification (2026-10-19)

* Before deleting the five requested temporary tests: 414 Python tests passed,
  13 skipped, one real-Ray case excluded from the successful rerun. Both temporary
  frontend suites and the existing time-window suite passed.
* After deletion: 418 existing dashboard/fully-async/AvaCore-recording tests passed,
  13 skipped and one real-Ray case excluded. AvaTrain integration tests: 14 passed.
* The exact scoped commit candidate, excluding unrelated local recording/resume
  edits, also passed 411 tests (13 skipped, one real-Ray case excluded).
* The real-Ray case was attempted separately: node startup timed out inside
  `ray.init()` before dashboard activation. A retry without proxy environment
  variables reached the 240-second command timeout. This does not establish
  distributed/GPU runtime correctness.
* Ruff, compile/import checks, old/new module identity checks, aggregation import
  boundaries, mounted-URL frontend checks and wheel resource validation passed.
  The wheel contains all 90 frontend assets, provisioning files and the old CLI.
* No live training, W&B experiment, database migration or dashboard deployment
  was performed for this structural refactor.
