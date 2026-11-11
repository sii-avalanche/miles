"""Exercise stream completion with real CPU Gloo collectives and skewed ranks."""

import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from miles.backends.training_utils.weight_update import updater
from miles.backends.training_utils.weight_update.protocols.broadcast import WeightSyncResponseError


def _run_rank(rank, rendezvous, scenario):
    group_timeout = 0.4 if scenario == "stalled" else 5.0
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
        timeout=timedelta(seconds=group_timeout),
    )
    updater.get_gloo_group = lambda: dist.group.WORLD
    # Scale the old 150s control budget down without changing production code.
    updater._collective_call.__kwdefaults__["timeout"] = timedelta(seconds=0.1)
    sends = []
    visited = []

    def buckets(*args, **kwargs):
        if rank == 0:
            for index in range(32):
                visited.append(index)
                yield index

    def send(bucket):
        sends.append(bucket)
        time.sleep(0.015)
        if scenario == "lost_reply" and bucket == 16:
            raise WeightSyncResponseError("injected lost response")

    protocol = SimpleNamespace(is_sender=rank == 0, send_bucket=send, after_base_weights=lambda: None)
    driver = SimpleNamespace(
        protocol=protocol, weights_getter=lambda: None,
        _hf_weight_iterator=SimpleNamespace(iter_hf_weights=buckets),
    )
    try:
        if scenario == "control":
            # Short control phases must still fail promptly on a tardy peer.
            if rank == 0:
                updater._collective_call(lambda: time.sleep(0.5))
            else:
                with pytest.raises(RuntimeError, match="timed out"):
                    updater._collective_call(lambda: None)
        elif scenario == "stalled":
            if rank == 0:
                time.sleep(1.0)
            with pytest.raises(RuntimeError) as caught:
                updater.WeightUpdater._stream_weights_for_retry(driver)
            assert not isinstance(caught.value, WeightSyncResponseError)
        elif scenario == "lost_reply":
            with pytest.raises(WeightSyncResponseError):
                updater.WeightUpdater._stream_weights_for_retry(driver)
            if rank == 0:
                assert sends == list(range(17))
                assert visited == list(range(32))
        else:
            updater.WeightUpdater._stream_weights_for_retry(driver)
            if rank == 0:
                assert sends == visited == list(range(32))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("scenario", ["slow_stream", "lost_reply", "stalled", "control"])
def test_stream_coordination(tmp_path, scenario):
    mp.start_processes(
        _run_rank, args=(str(tmp_path / "rendezvous"), scenario), nprocs=2, start_method="fork",
    )
