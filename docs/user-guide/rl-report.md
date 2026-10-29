# Per-run RL report

The report uses Miles' existing dashboard collector, GPU sampler, Timer events,
engine metrics and timeline template. FSDP registers through the same hook as
Megatron. Async data pools add queue/version snapshots and batch/rejection events
to the existing `data_buffer` stream; no new node or process metrics are logged.

## Enable and open

Use the existing dashboard options in AvaTrain:

```toml
[train]
use_wandb = true
use_miles_dashboard = true
dump_details = "/persistent/path/unique-to-this-run/details"
```

`dump_details` also enables Miles' normal debug dumps. Use a distinct directory
for each run. Sampling uses `dashboard_gpu_sample_interval` and
`dashboard_sglang_scrape_interval`; the separate `perf_monitor_*` options have
been removed in favor of these existing settings.

With both backends enabled, the driver exports and uploads a snapshot about once
per minute. Open **Overview → Notes → RL timeline**, use summary `rl_report_url`,
or download `rl-report.html` from Files. Refresh to load a newer snapshot.
No HTTP server is started. Older runs need collected dashboard files to export.

The HTML embeds the latest four hours. It supports hover, wheel zoom, drag pan
and GPU selection. GPU/engine series are downsampled by the native queries;
fleet overview bins and outlier ordering are fixed at export time. The original
files remain under `<dump_details>/dashboard/` for longer-history inspection.

## Set up the W&B view once

Install the optional `wandb-workspaces==0.4.13`, then run:

```bash
python -m miles.utils.rl_report --entity ENTITY --project PROJECT
```

Use `--url 'WORKSPACE_URL'` to replace an existing saved view (backed up first),
or `--dry-run` to preview. Training does not need this optional package.
W&B retains the existing training/rollout/performance scalars and one HTML report;
there is no separate Data pool channel or duplicate engine-metric scrape.
