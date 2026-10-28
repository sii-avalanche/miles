# Per-run RL report

The report reuses the original Miles `dashboard/static` timeline template from
`10ec633a^`: the rollout/data-pool/sync view, fleet overview, GPU lanes, colours,
hover details, wheel zoom and drag pan. All data, scripts and styles are embedded
in one HTML file. No dashboard HTTP service, database channel or external CDN is
needed. Data pool is part of this page, not a separate W&B channel.

W&B authenticates and displays the report in a dedicated **single-run full-screen
HTML page**. Its ordinary Files tab shows HTML source, and its HTML sandbox does
not allow popup windows, so we use the full-screen page rather than an expiring
raw-file link. Downloaded HTML also works directly in a browser without W&B.

## Install the project views

Run these commands in an environment with Miles installed. Only the workspace
editor needs the optional `wandb-workspaces==0.4.13` package; automatic report generation
and upload use the existing W&B dependency. Install it in the command
environment before configuring a workspace:

```bash
pip install wandb-workspaces==0.4.13
```

```bash
python -m miles.utils.rl_report \
  --project avatrain-math \
  --url 'http://117.135.59.58:28080/avalanche/avatrain-math?nw=unenmu8izsn' \
  --all-runs
```

This backs up the existing RL workspace, replaces its monitoring/data-pool panels
with 13 comparison plots plus an entry guide, and installs the project-scoped
`rlreport` view with one HTML panel. Neither view is pinned to a particular run.
W&B retains reward, response length, truncation, reward-group quality, loss, KL,
clip fraction, gradient norm, learning rate, throughput, MFU, step duration and
evaluation plots. Runtime details are not recreated as independent scalar panels.

## New runs

```toml
[train]
use_wandb = true
perf_monitor_interval = 10.0
```

The monitoring default is still zero; runs must enable it to produce telemetry.
The driver and rollout monitors send samples and exact phase events to the existing
fleet Ray actor. The driver's background monitor writes the report about once per
minute and on orderly shutdown. It registers one native HTML media record, then
replaces that same media file, avoiding cumulative copies of every HTML snapshot.
The Notes field gets a **RL timeline** link and summary gets `rl_report_url`:

```text
<run-url>/panel/rlreportpanel?nw=rlreport
```

The HTML shows the latest 20,000 records, explicitly labelled when earlier records
are omitted. All received, selected telemetry is also retained in
`rl-telemetry.jsonl.gz`, preserving the selected telemetry beyond the HTML window. Queues are
bounded; collector failures cannot block training and are reported in logs.
Sudden process death can leave the last uploaded snapshot up to a minute behind.
Refresh the report page to fetch a newer snapshot; it is not a live socket.

W&B no longer records `runtime/*` scalar history. Only report-consumed runtime
values and phase/buffer events are kept in the report archive. Training/evaluation
metrics, other tracking backends and W&B alert delivery are preserved. Existing
historical scalar data is not deleted. No separate Data pool view is created.

Missing telemetry stays missing. Historical sampled phase spans are explicitly
labelled `sampled`; unobserved short syncs cannot be reconstructed. GPU phase
strips represent role-level orchestration, not measured per-rank kernel timings.
Pool buckets are groups classified by weight version, not token counts.

## Existing runs

Reports already uploaded to W&B remain available through their run links. Runs
that did not collect monitoring telemetry cannot gain a full timeline by switching
branches; enable monitoring and use an image containing this Miles version for
future runs. The run's Files tab also contains `rl-report.html`, which can be
downloaded and opened directly in a browser.

## Validation

From the Miles checkout:

```bash
python3 -m unittest discover -s tests/fast/utils -p test_rl_report.py -v
python3 -m unittest discover -s tests/fast/utils -p test_fleet_dashboard.py -v
```

The report's static assets are packaged in the Miles wheel. Local tests do not
replace validation of telemetry collection in a real distributed GPU training job.
