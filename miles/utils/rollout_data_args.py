"""Explicit local rollout recording and offline training options."""

from pathlib import Path
from string import Formatter


def add_rollout_data_arguments(parser):
    parser.add_argument(
        "--save-rollout-data",
        type=str,
        default=None,
        help="Save each complete training rollout batch before training. Path must contain {rollout_id}, e.g. /shared/run/rollouts/{rollout_id}.pt.",
    )
    parser.add_argument(
        "--load-rollout-data",
        type=str,
        default=None,
        help="Train from locally recorded batches without generation or reward calls. Path must contain {rollout_id}. Requires --use-tis and the same tokenizer and samples-per-prompt setting as the recording.",
    )


def resolve_rollout_data_arguments(args):
    for option in ("save_rollout_data", "load_rollout_data"):
        value = getattr(args, option, None)
        if value is not None:
            fields = [field for _, field, _, _ in Formatter().parse(value) if field is not None]
            if fields != ["rollout_id"]:
                raise ValueError(f"--{option.replace('_', '-')} must contain exactly one {{rollout_id}} field")
    if not args.load_rollout_data:
        return
    if args.load_debug_rollout_data is not None or args.load_debug_rollout_data_subsample is not None:
        raise ValueError("--load-rollout-data cannot be combined with debug rollout loading or subsampling")
    if not args.use_tis:
        raise ValueError("--load-rollout-data requires --use-tis to correct for the recorded behavior policy")
    for target in (args.save_rollout_data, getattr(args, "save_debug_rollout_data", None)):
        if target and Path(args.load_rollout_data).resolve() == Path(target).resolve():
            raise ValueError("Record replayed batches to a different path; do not overwrite the source recording")
    if getattr(args, "megatron_config", None) is not None or getattr(args, "multi_lora", False):
        raise ValueError("--load-rollout-data currently supports single-policy runs without --megatron-config")
    # Reuse the established train-only lifecycle, including disabling inference workers.
    # Keep the public option so the loader can enforce the recording contract.
    args.load_debug_rollout_data = args.load_rollout_data
