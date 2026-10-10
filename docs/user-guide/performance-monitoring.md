# RL performance monitoring

This monitor separates useful production, batch consumption, GPU activity and
infrastructure waits. Continuous `runtime/*` metrics use elapsed wall seconds;
existing `rollout/*`, `train/*`, `perf/*` and new `moe/*` metrics retain their step
axes. Do not add rollout and training durations together when they overlap.

## Enable and view

In an AvaTrain TOML, add to `[train]`:

```toml
use_miles_dashboard = true
perf_monitor_interval = 10.0
dashboard_gpu_sample_interval = 2.0
dashboard_sglang_scrape_interval = 5.0
dashboard_flush_interval = 5.0
# Optional: existing routing replay is required. Layer numbers start at 1.
perf_monitor_expert_layer = 9
```

`avatrain rl train` supplies `dashboard_dir = <run>/logs/dashboard` unless an
explicit location or `dump_details` was supplied. `dump_details` is unnecessary
for telemetry and remains off; no large train/rollout tensors are dumped.
Generic Miles launches must set `--dashboard-dir` or `--dump-details` explicitly.
To only publish runtime metrics to W&B/local logs, omit `use_miles_dashboard`;
fleet GPU/engine sampling requires the dashboard collector.

```bash
./.venv/bin/python -m miles.dashboard.serve \
  --dashboard-dir runs/xy1008/<run>/logs/dashboard \
  --host 127.0.0.1 --port 7788 --follow --use-utilization-overview
```

Open `http://127.0.0.1:7788/#/runtime` for **RL Bottlenecks** or `#/timeline`
for per-GPU utilization and trainer phase lanes. The page refreshes every five
seconds in follow mode. For a remote server, forward the port with SSH.
W&B receives the same runtime counters through the existing shared run; use
`runtime/time_s` as X and choose the fields below as Y.

`rl submit` executes code installed in its selected image. It does not upload
local source changes. Build/publish an image containing the modified `avatrain`
and `miles` and select it with `--image` before using the new TOML options with
submit. Existing running jobs keep their original code/configuration.

## Bounds and what to measure

| Limit or bottleneck | Concrete control / operation | Evidence to collect |
| --- | --- | --- |
| Valid batch supply | `rollout_batch_size` groups × `n_samples_per_prompt`; dynamic, abort, missing-reward and staleness filters | collection progress; generated vs selected group rates; each rejection count; existing raw reward and staleness |
| Client concurrency | `async_max_concurrent_samples`, whole-group submission and sample/group submission scheduler | actual in-flight samples, sample budget, unfinished groups, completed groups awaiting delivery |
| Inference capacity | per-engine `sglang_max_running_requests`, KV/Mamba memory budget, prefill chunk size, router queues | running/queued requests, KV utilization, tokens/s, TTFT/inter-token latency, GPU utilization/memory |
| Long-tail generation | a group waits for its slowest answer; very long output/context | oldest in-flight group, completed-group latency p95, response-length p95/max, time since last completion |
| Producer backpressure | finished-group buffer has `floor(capacity_factor × rollout_batch_size)` slots | buffer fill, live full-buffer wait, time in full-buffer wait per wall second |
| Trainer starvation | driver awaiting `RolloutExecutor.get`; consumer awaiting a valid group | driver rollout wait, live empty-buffer wait, valid collection progress and selected rate |
| Trainer compute/memory | log-probability forwards, reference forwards, actor forward/backward, optimizer; `max_tokens_per_gpu`, recomputation | existing phase times, tokens/s, MFU, microbatch count, device utilization/memory |
| Trainer communication | CP exchanges context/KV or linear-attention states; EP dispatches tokens, gradient reductions | GPU activity and phase times identify a candidate; CUDA/NCCL traces must measure communication separately |
| Data transfer | sample conversion, object-store put, trainer object-store get and tensor preprocessing | `convert_batch`, `object_store_put`, existing `perf/data_preprocess_time` |
| Weight synchronization | generation pause/retraction, trainer→inference weight transfer, resume | driver `update_weights` span; generation pause and engine activity; existing update-weight phase timings |
| Recording | background record tasks, local pool, connection establishment, serialization, SQL/transaction | pending/oldest writes, pool waiting/available/size/errors, successful/failed writes, write latency |
| CPU/container resources | Python generation/reward/serialization, native threads, host/cgroup memory | CPU seconds/s, event-loop lag, executor RSS, node memory available, cgroup usage/limit/OOM-kill counter |
| Checkpoints, evaluation and startup | save/checkpoint storage, shared-fleet evaluation, model/engine initialization | driver spans, GPU lanes; eval dispatch is submission/blocking time, not the duration of a dedicated async eval job |

For the present `xy1008.toml`, a batch is **16 groups × 16 = 256 samples**;
the default completed-group buffer holds **32 groups**. The client sample
budget is **1024**, training has **32 GPUs with TP=1, PP=1, CP=32, EP=32**, and
inference has **32 GPUs / 2 per engine = 16 engines**. Sample-backfill scheduling
can have more than 64 unfinished groups when each retains a slow sample; 64 is
the sample budget expressed in whole groups, not a hard cap on unfinished groups.
`max_tokens_per_gpu` is a batching control, not a guarantee against GPU OOM.

A useful supply comparison is selected groups per second versus
`16 / (trainer time + synchronization + blocking checkpoint time)`. Compute
this over a sufficiently long wall-clock window. Raw generated tokens/s can
rise while valid-group supply falls due to filtering or long-tail answers.
Driver `wait_rollout` measures the exposed wait after overlap, while fully-async
`perf/rollout_time` measures batch collection and may overlap with other work.

## Metric map

Prefixes below are `runtime/rollout/` unless otherwise specified.
`*_total` counts since process start; `*_per_s` is a delta over the last emitted
interval. Duration p50/p95/max use the last **256 completed operations**, not
all operations or a fixed five-minute window. Open waits are visible before
they complete; `seconds_per_s` includes the change in ongoing wait time.

| Metric | Meaning / useful comparison |
| --- | --- |
| `drain/progress_fraction`, `drain/target_groups`, `drain/collected_groups` | fraction and counts of valid groups in the requested batch |
| `producer/in_flight_samples`, `producer/sample_budget` | scheduler sample slots in use / effective group-aligned limit; sample metric exists for the sample-backfill scheduler |
| `producer/in_flight_groups`, `producer/completed_pending_groups` | unfinished group tasks vs finished tasks the producer has yet to deliver |
| `producer/oldest_in_flight_seconds`, `producer/group_generation/p95_seconds` | current long tail vs recent completed group latency |
| `buffer/fill_fraction`, `buffer/queued_groups`, `buffer/capacity_groups` | completed groups held in the default buffer, independent of database records |
| `buffer/full_wait/waiting`, `buffer/full_wait/oldest_wait_seconds`, `buffer/full_wait/seconds_per_s` | producer blocked by a full buffer; prolonged values suggest consumption is slower than supply |
| `buffer/empty_wait/*` | consumer waiting for a group; pair with driver wait and filter rates |
| `buffer/residence/p95_seconds` | time a popped group spent queued, including groups subsequently rejected as stale |
| `buffer/groups_seen_per_s`, `buffer/groups_consumed_per_s` | group-level input and valid output rates; rates are not token throughput |
| `buffer/{dynamic_rejected,stale_rejected,aborted,missing_reward}_groups_per_s` | reasons production is failing to become training data |
| `producer/recycled_groups_total` | original prompt groups returned for retry, not reused old generated answers |
| `sample/generate/*`, `sample/reward/*`, `event_loop/lag_seconds` | generation end-to-end time, reward time and scheduling delay; generation includes API/engine/network waits |
| `record/pending_writes`, `record/oldest_pending_seconds` | detached writes still unfinished; no new queue or connection limit is introduced |
| `record/{success,failed,pool_timeout,other_errors}_total` | completed outcomes; a successful rollout does not imply a successful database record |
| `record/pool/{pool_size,pool_available,requests_waiting,connections_errors}` | actual psycopg pool gauges/counters; fields appear after the pool opens |
| `record/open_wait/*` | awaiting the shared store/run initialization task |
| `record/create_rollout/*`, `record/end_to_end/*` | complete create call vs complete record operation; both include waits/serialization/SQL, not isolated server SQL execution |
| `host/process_cpu_seconds_per_s` | process CPU core equivalents: 1 means approximately one fully busy CPU core |
| `host/process_rss_gb`, `host/container_memory_gb`, `host/container_memory_limit_gb` | physical process/container memory in GiB; device VRAM is measured separately |
| `host/process_virtual_memory_gb`, `host/process_threads` | virtual address reservations and native thread count; virtual size is not physical RAM usage |
| `runtime/driver/phase/{wait_rollout,train,update_weights,checkpoint}/*` | controller wall spans, including worker/RPC waits and retry overhead |
| `runtime/driver/fleet/{train,inference}/gpu_util_mean_pct` | mean of fresh GPU activity samples assigned to that role; inference includes registered eval engines if present |
| `runtime/driver/fleet/{train,inference}/coverage_fraction` | sampled / expected-or-mapped GPUs; low coverage makes averages partial |
| `runtime/driver/fleet/inference/{running_requests,queued_requests,kv_usage_mean,kv_usage_max}` | fresh engine gauges; reporting-engine counts reveal incomplete scrape coverage |
| `moe/rollout_layer_9/{cv,max_over_mean,cold_experts_fraction,sample_coverage}` | optional selected-batch routing statistics: std/mean; max/mean; fraction below 0.1×mean; replay coverage |

Per-policy default buffers use `buffer/<trainer_model_id>/...`; the Metrics
view exposes these keys even when the overview panels use the single-policy
names. Custom buffers retain their interface and filtering policy; their
internal waits require equivalent instrumentation in that implementation.

## Reading the three kinds of charts

* **Expert-load and failure timeline:** layer numbers start at 1. Expert counts
  are observed inference assignments (prompt plus generated positions) in the
  selected training batch. They cannot prove trainer-side router balance, nor
  the benefit of freezing a router without a controlled comparison. The driver
  timeline records stage, rollout ID, start/end and exception class. An
  `ActorDiedError` during collection identifies the failure stage; distinguishing
  kernel OOM, signal, network failure and segfault still needs worker/kernel logs.
* **Collection / workload share:** the default workload is `all`. For mixed
  datasets, provide `[train] perf_monitor_workload_key = "source"` and
  `perf_monitor_workload_targets = { math = 0.7, code = 0.3 }`, with labels in
  `Sample.metadata["source"]`. Progress is `selected groups / (batch target ×
  target fraction)` and can exceed 1. The in-flight group share is a scheduling
  proxy, not measured GPU occupancy. Targets do not enforce a new scheduler.
  Unknown labels enter one `other` bucket; arbitrary sample text never becomes
  metric names. Up to 32 configured workloads are accepted.
* **GPU utilization / waits:** correlate trainer and inference GPU activity with
  full/empty-buffer waits and weight-sync phases on wall time. Device activity
  is not CUDA SM occupancy, memory-bandwidth utilization or MFU. High activity
  can include NCCL kernels. For CP/EP diagnosis, use the existing short-window
  PyTorch profiler (`profile_target = ["train_overall"]`) or Nsight Systems.

For Postgres, `pool_size=4`, `pool_available=0`, increasing waiting requests and
few connection errors suggest occupied connections or slow transactions. A
shrinking/empty pool with increasing `connections_errors` suggests connection
establishment/recovery failure. These are clues: compare server logs, SQL lock
waits and connection health. The monitor does not run SQL probes or load tests.

## Artifacts and limits

* `<run>/logs/dashboard/`: existing per-GPU phase and utilization streams, engine
  series, metrics and `runtime_events.jsonl`; live/offline views share this data.
* `<run>/logs/perf/<role>-<hostname>-<pid>.jsonl`: independent scalar snapshots
  and stage events, useful even when W&B/database access fails. The writer
  emits on the configured cadence and on orderly shutdown. A hard kill can
  lose the final interval; an open span means its end was not observed.
* Telemetry is off when `perf_monitor_interval=0`. Histories of durations and
  pending phase events are bounded. Monitoring failures are rate-limited and
  do not abort training. GPU/engine sampling reuses the existing collector,
  does not initialize a new CUDA context, and introduces no database writes.
* GPU values older than the freshness window are omitted, not replaced with
  zero. Sparse sampling can miss brief activity; use per-GPU lanes and profiler
  traces for detailed claims. Clocks across nodes need normal synchronization.
* Recording behavior, staleness policy, task concurrency and training losses
  are unchanged by enabling these fields. Monitor `telemetry/expert_counts`
  duration when assessing the optional routing-count overhead.
