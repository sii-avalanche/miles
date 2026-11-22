import time

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from miles.backends.training_utils.metrics.update import summarize_rank_updates


def _record(seconds, **counts):
    return {
        "seconds": seconds,
        "token_counts": {
            "model_input_tokens": 1000,
            "supervised_tokens": 100,
            "dummy_tokens": 4,
            **counts,
        },
    }


def test_real_tokens_counted_once_and_slowest_rank_used():
    result = summarize_rank_updates([_record(2.0), _record(4.0)])
    assert result["perf/actor_train_tgs_per_gpu"] == 125
    assert result["perf/actor_train_model_input_tokens"] == 1000
    assert result["perf/actor_train_dummy_tokens"] == 4
    assert result["perf/actor_train_max_rank_time"] == 4
    assert result["perf/actor_train_mean_rank_time"] == 3


def test_unknown_token_basis_does_not_emit_real_tgs():
    result = summarize_rank_updates([{"seconds": 2, "token_counts": None}])
    assert "perf/actor_train_tgs_per_gpu" not in result


def test_mismatched_global_counts_rejected():
    with pytest.raises(ValueError, match="differ between ranks"):
        summarize_rank_updates([_record(2), _record(2, model_input_tokens=999)])


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf"), True])
def test_invalid_rank_time_rejected(seconds):
    with pytest.raises(ValueError):
        summarize_rank_updates([_record(seconds)])


@pytest.mark.parametrize("counts", [{"dummy_tokens": -1}, {"model_input_tokens": True}, {"supervised_tokens": 1001}])
def test_invalid_counts_rejected(counts):
    with pytest.raises(ValueError):
        summarize_rank_updates([_record(2, **counts)])


def test_every_rank_collects_before_metrics_are_reduced(monkeypatch):
    from miles.backends.training_utils.metrics import update

    group = object()
    monkeypatch.setattr(update.dist, "get_world_size", lambda actual: 2 if actual is group else 0)

    def gather(output, record, *, group):
        output[:] = [record, _record(4)]

    monkeypatch.setattr(update.dist, "all_gather_object", gather)
    result = update.collect_update_metrics(2, _record(2)["token_counts"], group=group)
    assert result["perf/actor_train_tgs_per_gpu"] == 125


def _gloo_worker(rank, rendezvous):
    from miles.backends.training_utils.metrics.update import collect_update_metrics

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        result = collect_update_metrics(2.0 + rank * 2, _record(1)["token_counts"], group=dist.group.WORLD)
        assert result["perf/actor_train_tgs_per_gpu"] == 125
        assert result["perf/actor_train_model_input_tokens"] == 1000
    finally:
        dist.destroy_process_group()


def test_real_two_rank_gloo_gathers_global_counts_once(tmp_path):
    processes = mp.start_processes(
        _gloo_worker, args=(tmp_path / "rendezvous",), nprocs=2, join=False, start_method="spawn"
    )
    deadline = time.monotonic() + 60
    try:
        while not processes.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("training metric workers timed out")
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
