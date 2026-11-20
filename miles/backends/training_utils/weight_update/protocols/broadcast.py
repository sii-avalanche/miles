import os
import socket
import time
from argparse import Namespace
from collections.abc import Sequence
from concurrent.futures import Future
from contextlib import AbstractContextManager, contextmanager, nullcontext
from datetime import timedelta

import httpx
import ray
import torch
import torch.distributed as dist

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState, get_parallel_state
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.utils import get_data_replica_rank_and_size
from miles.utils import async_utils
from miles.utils.distributed_lock import create_world_ticket_lock
from miles.utils.distributed_utils import init_process_group

_WEIGHT_SYNC_TIMEOUT_SECONDS = 60.0
_NCCL_WAIT_SECONDS = 30.0


class WeightSyncResponseError(RuntimeError):
    """All NCCL sends completed, but one or more HTTP replies were lost."""


class WeightSyncTransportError(WeightSyncResponseError):
    """The transfer communicator was aborted and all local GPU work settled."""


class UpdateWeightFromDistributed(WeightTransferProtocol):
    """
    Update distributed engines via NCCL. Each PP rank: group "miles-pp_{pp_rank}",
    only DP=TP=0 broadcasts. Non-expert (TP) and expert (EP) params separate.
    """

    supports_lora = True

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self._model_update_groups = None
        parallel_state = get_parallel_state()
        self._engine_lock: AbstractContextManager = (
            create_world_ticket_lock(
                prefix="miles/weight_update",
                participates=parallel_state.intra_dp_cp.rank == 0 and parallel_state.tp.rank == 0,
            )
            if parallel_state.pp.size > 1
            else nullcontext()
        )

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        """
        Create NCCL "miles-pp_{pp_rank}" if PP source (DP=TP=0). Lock prevents concurrent broadcasts.
        """
        self.rollout_engines = rollout_engines
        self._selector = selector
        self._engine_gpu_counts = engine_gpu_counts

        # One sender per replica set; one NCCL group (sender + all engines) per shard.
        replica_rank, _ = get_data_replica_rank_and_size(parallel_state, placement)
        self.is_sender = replica_rank == 0
        shard = 0 if placement.gather_pp else parallel_state.pp.rank
        if self.is_sender:
            self.group_name = f"miles-pp_{shard}"
            disconnect_rollout_engines_from_distributed(
                self.args, self.group_name, self._model_update_groups, self.rollout_engines
            )
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.args,
                self.group_name,
                rollout_engines,
                engine_gpu_counts=engine_gpu_counts,
                **(
                    {"timeout": _WEIGHT_SYNC_TIMEOUT_SECONDS}
                    if getattr(self.args, "update_weight_sync_retries", 0)
                    else {}
                ),
            )

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        """Lock → broadcast → clear → unlock. Lock prevents NCCL deadlock."""
        retry_enabled = bool(getattr(self.args, "update_weight_sync_retries", 0))
        recover_nccl = getattr(self.args, "update_weight_sync_recover_nccl", False)
        with self._engine_lock:
            try:
                futures = update_weights_from_distributed(
                    self.group_name,
                    self._model_update_groups,
                    self.rollout_engines,
                    bucket,
                    selector=self._selector,
                    transfer_mode=getattr(self.args, "update_weight_transfer_mode", "broadcast"),
                    **({"timeout": _WEIGHT_SYNC_TIMEOUT_SECONDS} if retry_enabled else {}),
                    **({"recoverable_timeout": _NCCL_WAIT_SECONDS} if recover_nccl else {}),
                )
                if retry_enabled:
                    _wait_for_bucket(futures, timeout=_WEIGHT_SYNC_TIMEOUT_SECONDS + 5.0, recover_nccl=recover_nccl)
                else:
                    async_utils.wait_futures(futures)
            except WeightSyncTransportError:
                # A receiver can time out even when the sender completed. Drop
                # our corresponding group too; never reuse a partially aborted group.
                _abort_transfer_group(self._model_update_groups)
                raise
            bucket.clear()

    def recover_connection(self):
        if not self.is_sender:
            return
        results = disconnect_rollout_engines_from_distributed(
            self.args,
            self.group_name,
            self._model_update_groups,
            self.rollout_engines,
            timeout=_WEIGHT_SYNC_TIMEOUT_SECONDS,
        )
        self._model_update_groups = None
        if any(not isinstance(result, dict) or result.get("success") is not True for result in results):
            raise RuntimeError(f"Cannot confirm weight-transfer group teardown: {results}")
        self._model_update_groups = connect_rollout_engines_from_distributed(
            self.args,
            self.group_name,
            self.rollout_engines,
            engine_gpu_counts=self._engine_gpu_counts,
            timeout=_WEIGHT_SYNC_TIMEOUT_SECONDS,
        )


@contextmanager
def _recoverable_nccl_settings():
    settings = {"TORCH_NCCL_BLOCKING_WAIT": "1", "TORCH_NCCL_ASYNC_ERROR_HANDLING": "2"}
    previous = {key: os.environ.get(key) for key in settings}
    try:
        os.environ.update(settings)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _abort_transfer_group(group):
    group._get_backend(torch.device("cuda", torch.cuda.current_device())).abort()
    # query() also surfaces a poisoned CUDA context. Do not drain trainer gathers
    # unless a new event behind the aborted transfer can actually complete.
    completed = torch.cuda.Event()
    completed.record()
    deadline = time.monotonic() + 5.0
    while not completed.query():
        if time.monotonic() >= deadline:
            raise RuntimeError("Aborted weight transfer did not release CUDA; recovery is unsafe")
        time.sleep(0.001)


def _wait_for_bucket(futures: Sequence[Future], *, timeout: float, recover_nccl=False) -> None:
    """A lost reply is recoverable only after GPU completion and settled RPCs."""
    # Work.wait() orders this stream after NCCL; synchronize() could block forever.
    completed = torch.cuda.Event()
    completed.record(torch.cuda.current_stream())
    deadline = time.monotonic() + timeout
    while not (completed.query() and all(future.done() for future in futures)):
        if time.monotonic() >= deadline:
            pending = [index for index, future in enumerate(futures) if not future.done()]
            raise RuntimeError(
                f"Weight transfer did not settle within {timeout}s (pending HTTP receivers={pending}); "
                "do not retry or drain CUDA gathers; restart the affected workers"
            )
        time.sleep(0.001)
    errors = []
    for future in futures:
        try:
            result = future.result()
            if not isinstance(result, dict) or result.get("success") is not True:
                raise RuntimeError(f"Cannot confirm weight bucket receipt: {result}")
        except Exception as error:
            errors.append(error)
    if errors:
        lost_reply = (httpx.ReadError, httpx.ReadTimeout, httpx.RemoteProtocolError)

        def receiver_aborted(error):
            return (
                isinstance(error, httpx.HTTPStatusError)
                and error.response.status_code == 400
                and "WEIGHT_SYNC_NCCL_ABORTED:" in error.response.text
            )

        if (
            recover_nccl
            and any(receiver_aborted(error) for error in errors)
            and all(receiver_aborted(error) or isinstance(error, lost_reply) for error in errors)
        ):
            raise WeightSyncTransportError("Receiver aborted NCCL weight transfer") from errors[0]
        if all(isinstance(error, lost_reply) for error in errors):
            raise WeightSyncResponseError("Lost reply after completed NCCL transfer") from errors[0]
        raise next(error for error in errors if not isinstance(error, lost_reply))


def connect_rollout_engines_from_distributed(
    args: Namespace,
    group_name: str,
    rollout_engines: Sequence[SGLangApiClient],
    engine_gpu_counts: Sequence[int] | None = None,
    timeout: float | None = None,
) -> dist.ProcessGroup:
    """
    Create NCCL group: training rank 0 + all engine GPUs. Blocks until joined.

    ``engine_gpu_counts`` gives the number of GPUs per engine.  When engines
    have heterogeneous TP sizes (e.g. prefill TP=2, decode TP=4), each engine
    occupies a different number of ranks in the NCCL group.
    """
    recover_nccl = getattr(args, "update_weight_sync_recover_nccl", False)
    if engine_gpu_counts is None:
        engine_gpu_counts = [args.rollout_num_gpus_per_engine] * len(rollout_engines)
    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    world_size = sum(engine_gpu_counts) + 1

    futures = []
    rank_cursor = 1
    for i, api_client in enumerate(rollout_engines):
        futures.append(
            async_utils.submit(
                api_client.init_weights_update_group(
                    master_address,
                    master_port,
                    rank_cursor,
                    world_size,
                    group_name,
                    backend="nccl",
                    **({"recoverable_timeout": _NCCL_WAIT_SECONDS} if recover_nccl else {}),
                    **({"timeout": timeout} if timeout is not None else {}),
                )
            )
        )
        rank_cursor += engine_gpu_counts[i]
    context = _recoverable_nccl_settings() if recover_nccl else nullcontext()
    with context:
        model_update_groups = init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=0,
            group_name=group_name,
            **(
                {"timeout": timedelta(seconds=4 * _NCCL_WAIT_SECONDS if recover_nccl else timeout)}
                if timeout is not None
                else {}
            ),
        )
    results = async_utils.wait_futures(futures)
    if getattr(args, "update_weight_sync_retries", 0) and any(
        not isinstance(result, dict) or result.get("success") is not True for result in results
    ):
        raise RuntimeError(f"Cannot confirm transfer group initialization: {results}")
    if recover_nccl and any(result.get("recoverable_timeout") != _NCCL_WAIT_SECONDS for result in results):
        raise RuntimeError("SGLang did not confirm recoverable NCCL support; update the receivers first")
    return model_update_groups


def disconnect_rollout_engines_from_distributed(
    args, group_name, model_update_groups, rollout_engines, *, timeout=None
):
    """
    Destroy NCCL on training and engines, returning the engine acknowledgements.
    """
    request_kwargs = {"timeout": timeout} if timeout is not None else {}
    futures = [
        async_utils.submit(client.destroy_weights_update_group(group_name, **request_kwargs))
        for client in rollout_engines
    ]
    try:
        if model_update_groups is not None:
            dist.destroy_process_group(model_update_groups)
    finally:
        results = async_utils.wait_futures(futures)
    return results


def update_weights_from_distributed(
    group_name: str,
    group: dist.ProcessGroup,
    rollout_engines: Sequence[SGLangApiClient],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
    selector: str = "all",
    transfer_mode: str = "broadcast",
    *,
    timeout: float | None = None,
    recoverable_timeout: float | None = None,
) -> list[Future]:
    """
    Send metadata (HTTP), broadcast tensors (NCCL rank 0 → engines).
    """
    if transfer_mode not in ("broadcast", "broadcast_packed"):
        raise ValueError(f"Expected a broadcast transfer mode, got {transfer_mode!r}")
    # Pack before asking receivers to enter their collective, so allocation or
    # format validation failures cannot leave them waiting for a broadcast.
    if transfer_mode == "broadcast_packed":
        tensors = [_flatten_weight_bucket(converted_named_tensors)]
        format_kwargs = {"load_format": "flattened_bucket"}
    else:
        tensors = [param.data.contiguous() for _, param in converted_named_tensors]
        format_kwargs = {}

    futures = [
        async_utils.submit(
            client.update_weights_from_distributed(
                names=[name for name, _ in converted_named_tensors],
                dtypes=[param.dtype for _, param in converted_named_tensors],
                shapes=[param.shape for _, param in converted_named_tensors],
                selector=selector,
                group_name=group_name,
                **format_kwargs,
                **({"timeout": timeout} if timeout is not None else {}),
            )
        )
        for client in rollout_engines
    ]

    handles = []
    try:
        for tensor in tensors:
            handles.append(dist.broadcast(tensor, 0, group=group, async_op=True))
        for handle in handles:
            handle.wait(
                **({"timeout": timedelta(seconds=recoverable_timeout)} if recoverable_timeout is not None else {})
            )
    except dist.DistBackendError as error:
        if recoverable_timeout is None:
            raise
        _abort_transfer_group(group)
        try:
            _wait_for_bucket(futures, timeout=_WEIGHT_SYNC_TIMEOUT_SECONDS + 5.0, recover_nccl=True)
        except WeightSyncResponseError:
            pass
        raise WeightSyncTransportError("Sender aborted NCCL weight transfer") from error

    return futures


def _flatten_weight_bucket(named_tensors: Sequence[tuple[str, torch.Tensor]]) -> torch.Tensor:
    # Keep SGLang optional for callers using the existing per-tensor protocol.
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

    if not getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
        raise RuntimeError("Flattened broadcasts require SGLang's mixed-dtype FlattenedTensorBucket")

    # SGLang concatenates raw bytes without padding. Each reconstructed dtype
    # view requires an aligned offset; preserve names/order and reject unsafe
    # layouts before allocating or sending metadata instead of casting values.
    offset = 0
    for name, tensor in named_tensors:
        if offset % tensor.element_size():
            raise ValueError(f"Unaligned flattened weight {name}: byte offset {offset}, dtype {tensor.dtype}")
        offset += tensor.numel() * tensor.element_size()
    return FlattenedTensorBucket(
        named_tensors=[(name, tensor.contiguous()) for name, tensor in named_tensors]
    ).get_flattened_tensor()
