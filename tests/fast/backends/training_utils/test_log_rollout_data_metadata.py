from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.data import context_parallel
from miles.backends.training_utils.metrics import log_utils


@pytest.fixture
def rollout_logger(monkeypatch):
    state = SimpleNamespace(tp=SimpleNamespace(rank=0), cp=SimpleNamespace(size=1), is_pp_last_stage=True)
    monkeypatch.setattr(log_utils, "get_parallel_state", lambda: state)
    monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: state)
    captured = []

    def capture(metric_name, args, rollout_id, log_dict):
        captured.append((metric_name, rollout_id, log_dict))
        return log_dict

    monkeypatch.setattr(log_utils, "gather_log_data", capture)
    args = Namespace(
        ci_test=False, qkv_format="thd", log_multi_turn=False, log_passrate=False, log_correct_samples=False
    )
    return args, captured


def rollout_data():
    return {
        "total_lengths": [3, 2, 2],
        "response_lengths": [2, 1, 1],
        "loss_masks": [torch.tensor([1, 1]), torch.tensor([1]), torch.tensor([0])],
        "log_probs": [torch.tensor([-0.2, -0.4]), torch.tensor([-0.6]), torch.tensor([-999.0])],
        "sample_weights": [0.5, 1.5, 0.0],
        "dynamic_global_batch_size": 3,
    }


@pytest.mark.parametrize("counts", [{}, {"model_input_tokens": 5, "supervised_tokens": 3, "dummy_tokens": 2}])
def test_training_token_counts_are_preserved_without_changing_logged_metrics(rollout_logger, counts):
    args, captured = rollout_logger
    data = rollout_data()
    weights = data["sample_weights"]
    original_counts = dict(counts)
    original_log_probs = [value.clone() for value in data["log_probs"]]
    log_utils.log_rollout_data(7, args, data)

    data["training_token_counts"] = counts
    log_utils.log_rollout_data(7, args, data)

    assert captured[0] == captured[1]
    metric_name, rollout_id, metrics = captured[1]
    assert (metric_name, rollout_id) == ("rollout", 7)
    assert "training_token_counts" not in metrics
    assert metrics["sample_weights"] == (2.0, 3)
    assert metrics["log_probs"] == pytest.approx((-0.9, 3))
    assert data["training_token_counts"] is counts
    assert counts == original_counts
    assert data["sample_weights"] is weights
    assert weights == [0.5, 1.5, 0.0]
    for actual, expected in zip(data["log_probs"], original_log_probs, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("key", ["unknown_accounting_metadata", "sample_weights"])
def test_unrecognized_dictionary_fields_still_raise(rollout_logger, key):
    args, captured = rollout_logger
    data = rollout_data()
    data["training_token_counts"] = {"model_input_tokens": 5, "supervised_tokens": 3, "dummy_tokens": 2}
    unexpected = {"value": 1}
    data[key] = unexpected

    with pytest.raises(ValueError, match=f"Unsupported type: <class 'dict'> for key: {key}"):
        log_utils.log_rollout_data(7, args, data)

    assert not captured
    assert data[key] is unexpected
    assert unexpected == {"value": 1}
