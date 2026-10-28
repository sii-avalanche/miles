# Per-run RL report

The report reuses the Miles timeline template for engine/GPU utilization, data
pool and collect/train/weight-sync stages. Hover, wheel zoom and drag pan work
in one self-contained HTML file; no separate service or Data pool channel is needed.

## Enable and open

For AvaTrain, add to `[train]` (native Miles uses the corresponding CLI flags):

```toml
use_wandb = true
perf_monitor_interval = 10.0
# Optional sampling intervals, in seconds:
perf_monitor_gpu_interval = 2.0
perf_monitor_engine_interval = 5.0
```

Monitoring defaults to off. The training image must contain this Miles version.
After the first upload, open the run's **Overview → Notes → RL timeline**,
use its summary field `rl_report_url`, or download `rl-report.html` from Files.
The background publisher updates about once per minute; refresh the page to
load a newer snapshot. Runs without collected telemetry have no report.

## Set up W&B project views once

Install the optional `wandb-workspaces==0.4.13` in the command environment, then:

```bash
python -m miles.utils.rl_report --entity ENTITY --project PROJECT
```

This installs the full-screen report view and a compact experiment-comparison
workspace. To replace an existing saved workspace, add `--url 'WORKSPACE_URL'`;
it is backed up before saving. Use `--dry-run` to preview the sections.
Training jobs do not need this optional package or repeat workspace setup.

## Recorded data and limits

- W&B keeps `train/*`, `rollout/*` and `perf/*` scalars. Selected `runtime/*`
  telemetry goes into the report archive instead of separate W&B curves.
- The HTML contains the latest 20,000 records. `rl-telemetry.jsonl.gz` retains
  the collected report telemetry beyond that window. Other tracking backends
  continue receiving runtime scalars on the `runtime/time_s` axis.
- Missing samples stay missing; GPU activity is not MFU. Phase strips describe
  role-level orchestration, not per-GPU kernel timing. Pool counts are groups.
- Old `--use-miles-dashboard` and dashboard service flags are removed. Replace
  them with `--perf-monitor-interval 10`; no dashboard server command is needed.
