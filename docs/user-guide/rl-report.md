# Per-run RL report

The report uses Miles' existing dashboard collector, GPU sampler, Timer events,
engine metrics and timeline template. Training phase events use Megatron's existing
hook. Async data pools add queue/version snapshots and batch/rejection events
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
per minute. In a project with the saved RL report view, open
**Overview → Notes → RL timeline** or use summary `rl_report_url`.
The link targets the `rlreport` view and its `rlreportpanel` HTML panel, displaying
the `rl_report` media key. Existing saved W&B views remain independent of this code.
Without that view, download `rl-report.html` from Files and open it locally.
Refresh to load a newer snapshot.
No HTTP server is started. Older runs need collected dashboard files to export.

The HTML embeds the latest four hours. It supports hover, wheel zoom, drag pan
and GPU selection. GPU/engine series are downsampled by the native queries;
fleet overview bins and outlier ordering are fixed at export time. The original
files remain under `<dump_details>/dashboard/` for longer-history inspection.

W&B retains the existing training/rollout/performance scalars and one HTML report;
there is no separate Data pool channel or duplicate engine-metric scrape.
