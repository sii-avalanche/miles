"""Describe scheduling from resolved launch metadata, with a fallback for old AvaTrain runs."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_FULLY_ASYNC = "miles.rollout.fully_async_rollout.FullyAsyncRolloutFn"
_BATCH = {
    "miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn",
    "miles.rollout.sglang_rollout.generate_rollout",
}
_KEYS = (
    "fully_async",
    "rollout_function_path",
    "num_steps_per_rollout",
    "update_weights_interval",
    "max_weight_staleness",
    "use_tis",
    "partial_rollout",
    "colocate",
    "pause_generation_mode",
)
_RESOLVED_PATH = re.compile(r"import ([A-Za-z_][\w.]*) as generate_rollout function\.")


def _config_train(run_dir: Path) -> dict:
    try:
        with (run_dir / "config.toml").open("rb") as file:
            train = tomllib.load(file).get("train", {})
        return train if isinstance(train, dict) else {}
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _resolved_path(run_dir: Path) -> str | None:
    try:
        # The import is logged at actor startup. Avoid scanning a growing rollout log.
        with (run_dir / "logs" / "actors" / "rollout_executor.log").open() as file:
            match = _RESOLVED_PATH.search(file.read(1024 * 1024))
        return match[1] if match else None
    except OSError:
        return None


def training_strategy(run_dir: Path, args: dict) -> dict:
    train = _config_train(run_dir)
    settings = {key: args.get(key, train.get(key)) for key in _KEYS}
    # AvaTrain's run_miles imports train_async.train. Native miles runs publish their entrypoint.
    entrypoint = args.get("dashboard_training_entrypoint")
    if entrypoint is None and (run_dir / "dashboard_run.json").is_file():
        entrypoint = "train_async"
    if entrypoint not in ("train", "train_async"):
        entrypoint = None
    resolved = args.get("rollout_function_path") or _resolved_path(run_dir)
    path = resolved or settings["rollout_function_path"]
    if not path and settings["fully_async"] is True:
        path = _FULLY_ASYNC
    # A known async driver plus an explicit false selects the standard batch rollout resolver.
    if not path and entrypoint == "train_async" and settings["fully_async"] is False:
        path = "miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn"
    kind = "unknown"
    if path == _FULLY_ASYNC:
        kind = "fully_async"
    elif path in _BATCH and entrypoint is not None:
        kind = "one_step_off_policy" if entrypoint == "train_async" else "synchronous"
    elif path and path not in _BATCH:
        kind = "custom"
    return dict(
        kind=kind,
        entrypoint=entrypoint,
        rollout_function_path=path,
        source="resolved_metadata" if resolved else "launch_config" if train else "unknown",
        settings={
            key: value for key, value in settings.items() if key != "rollout_function_path" and value is not None
        },
    )
