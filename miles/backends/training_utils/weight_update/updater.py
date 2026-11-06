"""Backend-neutral weight-update driver.

The updater owns the lifecycle: it builds the HF weight iterator against the
protocol's required placement, runs the engine session frame, streams base
buckets (senders transmit, other ranks join the gathers), and orchestrates
LoRA adapter pushes.
"""

import asyncio
import logging
import random
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import wait
from datetime import timedelta
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from tqdm import tqdm

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.conn_status import ConnStatusManager
from miles.backends.training_utils.weight_update.protocol import get_weight_transfer_protocol
from miles.backends.training_utils.weight_update.session import (
    begin_weight_update,
    end_weight_update,
    pause_engines,
    register_lora_adapter,
    resume_engines,
    set_weight_version,
)
from miles.backends.training_utils.weight_update.utils import record_lora_checksums
from miles.utils import async_utils
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.http_utils import GeneralHttpClientProvider
from miles.utils.lora.utils import LORA_ADAPTER_NAME
from miles.utils.timer import timer

if TYPE_CHECKING:
    from miles.ray.rollout.inference_controller import UpdatableEngines

logger = logging.getLogger(__name__)


class WeightUpdater:
    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        *,
        weights_getter: Callable[[], Mapping[str, torch.Tensor] | None],
        model_name: str,
        quantization_config: dict | None,
        iterator_factory: Callable,
        parallel_state: ParallelState,
        is_lora: bool,
        lora_sync_config: dict | None = None,
    ) -> None:
        self.args = args
        self.parallel_state = parallel_state
        self.protocol = get_weight_transfer_protocol(args)
        self.conn_status = ConnStatusManager()
        assert (
            not is_lora or self.protocol.supports_lora
        ), f"LoRA weight sync is not supported for {args.update_weight_transfer_mode!r} weight transfer."
        self._hf_weight_iterator = iterator_factory(
            args,
            model,
            required_placement=self.protocol.required_placement,
            model_name=model_name,
            quantization_config=quantization_config,
        )
        self.weights_getter = weights_getter
        self.weight_version = 0
        self.is_lora = is_lora
        if is_lora:
            assert lora_sync_config is not None
        self._lora_sync_config = lora_sync_config
        self._registered_adapters: set[str] = set()

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        self.protocol.connect(
            rollout_engines,
            engine_gpu_counts,
            engine_gpu_offsets,
            self.parallel_state,
            self._hf_weight_iterator.placement,
            self._hf_weight_iterator.weight_update_selector,
        )
        assert self.protocol.is_sender is not None, "connect() must set is_sender"
        self._registered_adapters.clear()

    def reconnect_if_needed(self, info: "UpdatableEngines") -> bool:
        if not self.conn_status.needs_reconnect(info.snapshot_cell_id_to_hashes):
            return False
        self.reconnect(info)
        return True

    def reconnect(self, info: "UpdatableEngines") -> None:
        self.connect_rollout_engines(
            info.rollout_engines,
            engine_gpu_counts=info.engine_gpu_counts,
            engine_gpu_offsets=info.engine_gpu_offsets,
        )
        self.conn_status.mark_reconnected(info.snapshot_cell_id_to_hashes)
        dist.barrier(group=get_gloo_group())

    def verify_engine_version(self, rollout_engines: Sequence[SGLangApiClient]) -> None:
        if not rollout_engines or self.weight_version == 0:
            return
        engine = random.choice(list(rollout_engines))
        engine_version = async_utils.run(engine.get_weight_version())
        if str(engine_version) != str(self.weight_version):
            raise RuntimeError(f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_version}")

    def pop_metrics(self) -> dict[str, float]:
        """Return and clear the protocol's metrics; the actor drains them onto the step log."""
        return self.protocol.pop_metrics()

    @torch.no_grad()
    def update_weights(self) -> None:
        """Run one weight sync: session frame + base-bucket stream + adapter pushes for LoRA."""
        retries = getattr(self.args, "update_weight_sync_retries", 0)
        if retries:
            return self._update_weights_with_retry(retries=retries)
        protocol = self.protocol
        if not protocol.begin_sync(self.weight_version + 1, self._iter_base_buckets):
            return
        self.weight_version += 1

        sync_base = not self.is_lora or protocol.needs_base_resync_for_lora
        adapters = self._get_updated_adapters()

        driver = dist.get_rank() == 0
        if protocol.use_weight_update_session and driver:
            pause_engines(self.args, protocol.rollout_engines)
            self._register_new_lora_adapters(protocol.rollout_engines, adapters)
            begin_weight_update(
                protocol.rollout_engines, self._hf_weight_iterator.weight_update_selector, sync_base=sync_base
            )
        dist.barrier(group=get_gloo_group())

        checksums = {name: {} for name, _ in adapters} if self.is_lora and self.args.check_lora_weight_equal else None
        if checksums is not None:
            assert (
                self._hf_weight_iterator.placement.gather_pp
            ), "the LoRA checksum manifest is recorded on one rank, which must hold the full adapter"
        with timer("update_weights_implementation"):
            pbar = tqdm(desc=f"[{protocol.group_name}] Update weights", total=0) if protocol.is_sender else None
            for bucket in self._hf_weight_iterator.iter_hf_weights(
                self.weights_getter(),
                include_base=sync_base,
                adapters=adapters,
                materialize=protocol.is_sender,
            ):
                if protocol.is_sender:
                    if driver and checksums is not None:
                        record_lora_checksums(bucket, checksums)
                    protocol.send_bucket(bucket)
                    pbar.update(1)
            protocol.after_base_weights()
            dist.barrier(group=get_gloo_group())

        with timer("finalize_and_resume_engines"):
            protocol.finalize(self.weight_version)
            if protocol.use_weight_update_session and driver:
                end_weight_update(protocol.rollout_engines, expected_lora_checksums=checksums)
                set_weight_version(protocol.rollout_engines, self.weight_version)
                resume_engines(protocol.rollout_engines)
            dist.barrier(group=get_gloo_group())
        protocol.after_engines_resumed()

    def _update_weights_with_retry(self, *, retries):
        """Keep one target version and replay the entire iterator after reconnecting."""
        # Keep optional broadcast dependencies out of other transfer modes.
        from miles.backends.training_utils.weight_update.protocols.broadcast import (
            WeightSyncResponseError,
            WeightSyncTransportError,
        )

        args, protocol = self.args, self.protocol
        if retries < 1:
            raise ValueError("Weight sync recovery requires a positive retry budget")
        if (
            self.is_lora
            or args.colocate
            or args.update_weight_transfer_mode not in ("broadcast", "broadcast_packed")
            or self.parallel_state.pp.size != 1
        ):
            raise ValueError("Weight sync recovery requires non-colocated base-model broadcast with PP=1")
        driver = dist.get_rank() == 0
        target_version = self.weight_version + 1
        _collective_call(lambda: pause_engines(args, protocol.rollout_engines) if driver else None)
        for attempt in range(retries + 1):
            _collective_call(
                lambda: (
                    begin_weight_update(
                        protocol.rollout_engines, self._hf_weight_iterator.weight_update_selector, strict=True
                    )
                    if driver
                    else None
                )
            )
            try:
                with timer("update_weights_implementation"):
                    self._stream_weights_for_retry()
            except WeightSyncResponseError as error:
                if attempt == retries:
                    raise
                logger.warning(
                    "Weight sync version=%s attempt=%s failed; reconnecting and replaying all buckets",
                    target_version,
                    attempt + 1,
                    exc_info=True,
                )
                # Lost HTTP replies retain the healthy group; confirmed NCCL aborts
                # rebuild it only after all ranks have drained their iterator.
                _collective_call(lambda: async_utils.run(asyncio.wait_for(GeneralHttpClientProvider.reset(), 60.0)))
                _collective_call(lambda: _abort_session(protocol.rollout_engines) if driver else None)
                if isinstance(error, WeightSyncTransportError):
                    _collective_call(protocol.recover_connection)
            else:
                break
        # No retry after commit/resume: that would require coordinating partially resumed engines.
        _collective_call(lambda: end_weight_update(protocol.rollout_engines) if driver else None)
        _collective_call(lambda: set_weight_version(protocol.rollout_engines, target_version) if driver else None)
        _collective_call(lambda: _verify_versions(protocol.rollout_engines, target_version) if driver else None)
        _collective_call(lambda: resume_engines(protocol.rollout_engines) if driver else None)
        self.weight_version = target_version
        protocol.after_engines_resumed()

    def _stream_weights_for_retry(self):
        from miles.backends.training_utils.weight_update.protocols.broadcast import WeightSyncResponseError

        protocol = self.protocol
        error = None
        # Non-senders join iterator gathers but yield no buckets. After a failed send,
        # drain those gathers without further engine writes before agreeing to retry.
        for bucket in self._hf_weight_iterator.iter_hf_weights(
            self.weights_getter(), include_base=True, adapters=[], materialize=protocol.is_sender
        ):
            if protocol.is_sender and error is None:
                try:
                    protocol.send_bucket(bucket)
                except WeightSyncResponseError as caught:
                    # Completed or confirmed-aborted transfers may drain subsequent CUDA gathers.
                    error = caught

        def finish():
            if error is not None:
                raise error
            protocol.after_base_weights()

        _collective_call(finish)

    def _iter_base_buckets(self, *, materialize: bool):
        return self._hf_weight_iterator.iter_hf_weights(self.weights_getter(), materialize=materialize)

    def _get_updated_adapters(self) -> list[tuple[str, object]]:
        """``(lora_name, adapter_or_None)`` pairs for this sync; the push set is
        identical on every rank so the iterator's collectives align."""
        if not self.is_lora:
            return []
        return [(LORA_ADAPTER_NAME, None)]

    def _register_new_lora_adapters(self, rollout_engines, adapters: list[tuple[str, object]]) -> None:
        """Register adapters the current engine set has not seen, with their
        per-adapter config; eager so the engine validates rank before any bytes move."""
        for lora_name, adapter in adapters:
            if lora_name in self._registered_adapters:
                continue
            config = self._lora_sync_config
            if adapter is not None:
                config = config | {"r": adapter.rank, "lora_alpha": adapter.alpha}
            register_lora_adapter(rollout_engines, lora_name=lora_name, lora_config=config)
            self._registered_adapters.add(lora_name)


def _collective_call(action):
    """Agree on failure before any rank enters another iterator collective."""
    # The error type belongs to the optional broadcast protocol.
    from miles.backends.training_utils.weight_update.protocols.broadcast import (
        WeightSyncResponseError,
        WeightSyncTransportError,
    )

    error = None
    try:
        action()
    except Exception as caught:
        error = caught
    status = torch.tensor(
        (
            0
            if error is None
            else (
                3
                if isinstance(error, WeightSyncTransportError)
                else 1 if isinstance(error, WeightSyncResponseError) else 4
            )
        ),
        dtype=torch.int32,
        device="cpu",
    )
    work = dist.all_reduce(status, op=dist.ReduceOp.MAX, group=get_gloo_group(), async_op=True)
    # Peers may reach agreement before the sender finishes its 65s bucket/abort budget.
    work.wait(timeout=timedelta(seconds=150))
    if status.item() == 4:
        raise RuntimeError("Weight synchronization failed; recovery is unsafe") from error
    if status.item() == 3:
        raise WeightSyncTransportError("Aborted weight-transfer group on a training rank") from error
    if status.item() == 1:
        raise WeightSyncResponseError("Lost weight-sync response on a training rank") from error


def _abort_session(engines):
    futures = [async_utils.submit(engine.abort_weight_update()) for engine in engines]
    _, pending = wait(futures, timeout=65.0)
    if pending:
        raise RuntimeError("Weight session abort did not settle; engines must remain paused")
    results = async_utils.wait_futures(futures)
    for result in results:
        if not isinstance(result, dict) or result.get("success") is not True:
            raise RuntimeError(f"Cannot confirm weight session abort: {result}")


def _verify_versions(engines, target_version):
    versions = async_utils.wait_futures([async_utils.submit(engine.get_weight_version()) for engine in engines])
    if any(str(version) != str(target_version) for version in versions):
        raise RuntimeError(f"Weight version verification failed: expected {target_version}, got {versions}")
