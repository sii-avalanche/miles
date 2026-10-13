import pytest

from miles.dashboard.training_strategy import training_strategy

BATCH = "miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn"
FULLY_ASYNC = "miles.rollout.fully_async_rollout.FullyAsyncRolloutFn"


def test_existing_avatrain_run_uses_resolved_actor_path_and_config(tmp_path):
    (tmp_path / "dashboard_run.json").write_text("{}")
    (tmp_path / "config.toml").write_text(
        "[train]\nfully_async=false\nupdate_weights_interval=1\n"
        'num_steps_per_rollout=1\nuse_tis=true\n[env]\nSECRET="hidden"\n'
    )
    logs = tmp_path / "logs" / "actors"
    logs.mkdir(parents=True)
    (logs / "rollout_executor.log").write_text(f"[log] import {BATCH} as generate_rollout function.\n")
    result = training_strategy(tmp_path, {})
    assert result["kind"] == "one_step_off_policy"
    assert result["entrypoint"] == "train_async"
    assert result["source"] == "resolved_metadata"
    assert result["rollout_function_path"] == BATCH
    assert result["settings"]["use_tis"] is True
    assert "hidden" not in str(result)


@pytest.mark.parametrize(
    ("entrypoint", "path", "kind"),
    [
        ("train_async", BATCH, "one_step_off_policy"),
        ("train", BATCH, "synchronous"),
        ("train_async", FULLY_ASYNC, "fully_async"),
        ("train_async", "plugin.CustomRollout", "custom"),
        (None, BATCH, "unknown"),
    ],
)
def test_native_run_modes(tmp_path, entrypoint, path, kind):
    assert (
        training_strategy(tmp_path, {"dashboard_training_entrypoint": entrypoint, "rollout_function_path": path})[
            "kind"
        ]
        == kind
    )


def test_effective_metadata_wins_over_config_and_run_name_does_not_select_mode(tmp_path):
    (tmp_path / "config.toml").write_text("[train]\nfully_async=false\nnum_steps_per_rollout=1\n")
    result = training_strategy(
        tmp_path,
        {"dashboard_training_entrypoint": "train_async", "rollout_function_path": FULLY_ASYNC, "fully_async": True},
    )
    assert result["kind"] == "fully_async"
    assert result["settings"]["fully_async"] is True
    assert training_strategy(tmp_path, {"wandb_group": "one-step-off-policy"})["kind"] == "unknown"


def test_old_avatrain_config_fallback_and_missing_snapshot(tmp_path):
    (tmp_path / "dashboard_run.json").write_text("{}")
    (tmp_path / "config.toml").write_text("[train]\nfully_async=true\n")
    assert training_strategy(tmp_path, {})["kind"] == "fully_async"
    (tmp_path / "config.toml").write_text("[train]\nfully_async=false\n")
    assert training_strategy(tmp_path, {})["kind"] == "one_step_off_policy"
    (tmp_path / "config.toml").unlink()
    assert training_strategy(tmp_path, {})["kind"] == "unknown"
