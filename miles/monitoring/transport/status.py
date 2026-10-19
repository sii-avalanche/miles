"""Read run configuration, scheduling mode and current runtime stages."""

from __future__ import annotations

import re
from pathlib import Path

import tomllib

from miles.monitoring.aggregation.runtime import _number
from miles.monitoring.aggregation.runtime import runtime_status as runtime_status


def _read_config(path: Path) -> dict:
    try:
        with path.open("rb") as file:
            return tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def reward_info(config_path: Path) -> dict:
    """Expose only judge identity and limits; never return endpoints, keys or env."""
    reward = _read_config(config_path).get("reward", {})
    if not isinstance(reward, dict):
        return {}
    result = {"kind": reward.get("kind")} if isinstance(reward.get("kind"), str) else {}
    if result.get("kind") != "lmaaj":
        return result
    model = reward.get("model", {})
    if not isinstance(model, dict):
        return result
    name = model.get("name")
    if isinstance(name, str) and "${" not in name:
        result["model_name"] = name
    for key in ("concurrency", "timeout", "max_retries"):
        value = _number(model.get(key))
        if value is not None:
            result[key] = value
    return result


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


def _resolved_path(run_dir: Path) -> str | None:
    try:
        # The import is logged at actor startup. Avoid scanning a growing rollout log.
        with (run_dir / "logs" / "actors" / "rollout_executor.log").open() as file:
            match = _RESOLVED_PATH.search(file.read(1024 * 1024))
        return match[1] if match else None
    except OSError:
        return None


def training_strategy(run_dir: Path, args: dict) -> dict:
    train = _read_config(run_dir / "config.toml").get("train", {})
    train = train if isinstance(train, dict) else {}
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
