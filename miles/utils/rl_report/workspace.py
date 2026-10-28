"""Keep W&B focused on experiment comparisons; runtime details live in each run's HTML report."""

import os
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import wandb_workspaces.reports.v2 as reports
from wandb_workspaces import workspaces

# Deliberate small set; detailed runtime, per-engine, per-GPU and pool plots are
# served by the single-run report. No separate Data pool workspace is created.
COMPARISONS = {
    "Quality": [
        ("Reward", "rollout/step", ["rollout/raw_reward", "rollout/raw_reward_unfiltered", "rollout/rewards"]),
        ("Response length", "rollout/step", ["rollout/response_len/mean", "rollout/response_len/max"]),
        ("Truncation", "rollout/step", ["rollout/truncated_ratio"]),
        ("Degenerate reward groups", "rollout/step", ["rollout/zero_std/all_zero_percentage", "rollout/zero_std/all_one_percentage"]),
    ],
    "Optimization": [
        ("Loss", "train/step", ["train/loss", "train/pg_loss", "train/entropy_loss"]),
        ("Policy KL", "train/step", ["train/ppo_kl", "train/train_rollout_kl"]),
        ("Clip fraction", "train/step", ["train/pg_clipfrac", "train/pg_clipfrac_lower"]),
        ("Gradient norm", "train/step", ["train/grad_norm"]),
        ("Learning rate", "train/step", ["train/lr-pg_0"]),
    ],
    "Efficiency": [
        ("Throughput", "rollout/step", ["perf/tokens_per_gpu_per_sec", "perf/effective_tokens_per_gpu_per_sec", "perf/actor_train_tok_per_s"]),
        ("Model FLOPs utilization", "rollout/step", ["perf/actor_train_mfu"]),
        ("Step duration", "rollout/step", ["perf/step_time"]),
    ],
}


def sections():
    result = [
        workspaces.Section(
            name="RL · Run report",
            panels=[
                reports.MarkdownPanel(
                    markdown=(
                        "Open a run → **Overview → Notes → RL timeline** for its dedicated full-screen Miles page. "
                        "The run summary also contains **rl_report_url**. Engine/GPU utilization, "
                        "data pool and collect/train/sync stages share one report; no separate Data pool channel."
                    )
                )
            ],
        )
    ]
    for name, entries in COMPARISONS.items():
        result.append(
            workspaces.Section(
                name=f"RL · {name}",
                panels=[
                    reports.LinePlot(
                        title=title,
                        x=axis,
                        y=keys,
                        max_runs_to_show=3,
                        legend_template="${run:name} · ${metricName}",
                    )
                    for title, axis, keys in entries
                ],
                layout_settings=workspaces.SectionLayoutSettings(columns=2),
            )
        )
    result.append(
        workspaces.Section(
            name="RL · Evaluation",
            panels=[
                reports.LinePlot(
                    title="Evaluation",
                    x="eval/step",
                    y=[],
                    metric_regex=r"^eval/(?!step$|skipped_|lag_steps$|duration_seconds$|export_time_seconds$)[^/]+$",
                    max_runs_to_show=3,
                )
            ],
        )
    )
    for section in result:
        section.is_open = True
        for panel in section.panels:
            panel._is_auto = False
    return result


def install_report_view(entity, project):
    panel = reports.MediaBrowser(
        title="RL timeline", media_keys=["rl_report"], num_columns=1
    )
    panel._id = "rlreportpanel"
    panel._is_auto = False
    view = workspaces.Workspace(
        entity=entity, project=project, name="RL report", auto_generate_panels=False,
        sections=[workspaces.Section(
            name="Run timeline", is_open=True, panels=[panel],
            layout_settings=workspaces.SectionLayoutSettings(columns=1),
        )],
    )
    # Stable project-scoped name and panel ID allow every run to construct its
    # own authenticated full-screen URL, without a hardcoded run or signed URL.
    view._internal_name = "nw-rlreport-v"
    view.save()


def configure_workspace(args):
    if args.host is None and args.url:
        parsed = urlsplit(args.url)
        args.host = f"{parsed.scheme}://{parsed.netloc}"
    if args.host:
        os.environ["WANDB_BASE_URL"] = args.host
    workspace = (
        workspaces.Workspace.from_url(args.url)
        if args.url
        else workspaces.Workspace(
            entity=args.entity,
            project=args.project,
            name="RL — selected metrics",
            auto_generate_panels=False,
        )
    )
    if args.url and not args.dry_run:
        directory = Path(args.backup_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        (directory / f"{args.project}-{stamp}.json").write_text(
            workspace._to_model().model_dump_json(by_alias=True)
        )
    workspace.sections = sections()
    workspace._auto_generate_panels = False
    workspace.settings.max_runs = 3
    if args.all_runs:
        workspace.runset_settings.filters = []
        workspace.runset_settings.pinned_runs = []
    if args.dry_run:
        print([(section.name, len(section.panels)) for section in workspace.sections])
        return
    install_report_view(workspace.entity, workspace.project)
    workspace.save()
    print(workspace.url)
