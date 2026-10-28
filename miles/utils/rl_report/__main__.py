"""Set up W&B project views for Miles reports and experiment comparisons."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="python -m miles.utils.rl_report", description=__doc__)
    parser.add_argument("--entity", default="avalanche")
    parser.add_argument("--project", required=True)
    parser.add_argument("--url", help="Existing saved view to replace; backed up before saving")
    parser.add_argument("--host")
    parser.add_argument("--all-runs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backup-dir", type=Path, default=Path("archives/wandb-workspace-backups"))
    args = parser.parse_args()
    # The workspace editor is an administration-only optional dependency.
    try:
        from miles.utils.rl_report.workspace import configure_workspace
    except ModuleNotFoundError as error:
        if error.name != "wandb_workspaces":
            raise
        parser.error("workspace setup requires wandb-workspaces==0.4.13; install it in the command's environment")
    configure_workspace(args)


if __name__ == "__main__":
    main()
