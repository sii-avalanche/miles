"""Retry complete, quiesced broadcasts after a lost response, never a bucket RPC."""

import logging

import torch
import torch.distributed as dist

from miles.backends.training_utils.weight_update.protocols.broadcast import WeightSyncResponseError
from miles.backends.training_utils.weight_update.session import (
    begin_weight_update,
    end_weight_update,
    pause_engines,
    resume_engines,
    set_weight_version,
)
from miles.utils import async_utils
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.http_utils import GeneralHttpClientProvider

logger = logging.getLogger(__name__)


def _collective_call(action):
    """Agree on failure before any rank enters another iterator collective."""
    error = None
    try:
        action()
    except Exception as caught:
        error = caught
    status = torch.tensor(
        0 if error is None else 1 if isinstance(error, WeightSyncResponseError) else 2,
        dtype=torch.int32,
        device="cpu",
    )
    dist.all_reduce(status, op=dist.ReduceOp.MAX, group=get_gloo_group())
    if status.item() == 2:
        raise RuntimeError("Weight synchronization failed; recovery is unsafe") from error
    if status.item() == 1:
        raise WeightSyncResponseError("Lost weight-sync response on a training rank") from error


def _abort_session(engines):
    results = async_utils.wait_futures([async_utils.submit(engine.abort_weight_update()) for engine in engines])
    for result in results:
        if not isinstance(result, dict) or result.get("success") is not True:
            raise RuntimeError(f"Cannot confirm weight session abort: {result}")


def _verify_versions(engines, target_version):
    versions = async_utils.wait_futures([async_utils.submit(engine.get_weight_version()) for engine in engines])
    if any(str(version) != str(target_version) for version in versions):
        raise RuntimeError(f"Weight version verification failed: expected {target_version}, got {versions}")


def _recover(updater, *, driver):
    # send_bucket settled all requests; no in-flight request can reuse the old client.
    _collective_call(lambda: async_utils.run(GeneralHttpClientProvider.reset()))
    _collective_call(lambda: _abort_session(updater.protocol.rollout_engines) if driver else None)
    # Abort acknowledges receiver-side work before the completed group is destroyed.
    _collective_call(updater.protocol.recover_connection)


def _stream(updater):
    protocol = updater.protocol
    error = None
    # Non-senders join iterator gathers but yield no buckets. After a failed send,
    # drain those gathers without further engine writes before agreeing to retry.
    for bucket in updater._hf_weight_iterator.iter_hf_weights(
        updater.weights_getter(), include_base=True, adapters=[], materialize=protocol.is_sender
    ):
        if protocol.is_sender and error is None:
            try:
                protocol.send_bucket(bucket)
            except Exception as caught:
                error = caught

    def finish():
        if error is not None:
            raise error
        protocol.after_base_weights()

    _collective_call(finish)


def update_weights_with_retry(updater, *, retries):
    """Keep one target version and replay the entire iterator after reconnecting."""
    args, protocol = updater.args, updater.protocol
    if retries < 1:
        raise ValueError("Weight sync recovery requires a positive retry budget")
    if (
        updater.is_lora
        or args.colocate
        or args.update_weight_transfer_mode not in ("broadcast", "broadcast_packed")
        or updater.parallel_state.pp.size != 1
    ):
        raise ValueError("Weight sync recovery requires non-colocated base-model broadcast with PP=1")
    driver = dist.get_rank() == 0
    target_version = updater.weight_version + 1
    _collective_call(lambda: pause_engines(args, protocol.rollout_engines) if driver else None)
    for attempt in range(retries + 1):
        _collective_call(
            lambda: (
                begin_weight_update(
                    protocol.rollout_engines, updater._hf_weight_iterator.weight_update_selector, strict=True
                )
                if driver
                else None
            )
        )
        try:
            _stream(updater)
        except WeightSyncResponseError:
            if attempt == retries:
                raise
            logger.warning(
                "Weight sync version=%s attempt=%s failed; reconnecting and replaying all buckets",
                target_version,
                attempt + 1,
                exc_info=True,
            )
            _recover(updater, driver=driver)
        else:
            break
    # No retry after commit/resume: that would require coordinating partially resumed engines.
    _collective_call(lambda: end_weight_update(protocol.rollout_engines) if driver else None)
    _collective_call(lambda: set_weight_version(protocol.rollout_engines, target_version) if driver else None)
    _collective_call(lambda: _verify_versions(protocol.rollout_engines, target_version) if driver else None)
    _collective_call(lambda: resume_engines(protocol.rollout_engines) if driver else None)
    updater.weight_version = target_version
    protocol.after_engines_resumed()
