import logging
import os
import socket
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
from dmodel.utils.manual_gc import ManualGarbageCollector

from miles.backends.dmodel_utils import checkpoint
from miles.backends.dmodel_utils.arguments import validate_dmodel_args
from miles.backends.dmodel_utils.engine import DModelEngine
from miles.backends.dmodel_utils.parallel import create_dmodel_parallel_state
from miles.backends.dmodel_utils.step_runner import DModelStepRunner
from miles.backends.training_utils.metrics.update import collect_update_metrics
from miles.backends.training_utils.parallel import get_parallel_state, set_parallel_state
from miles.backends.training_utils.replay import routing_replay
from miles.backends.training_utils.torch_native.actor import TorchNativeTrainRayActor
from miles.backends.training_utils.weight_update.version import acknowledge_versions
from miles.utils import async_utils
from miles.utils.context_utils import with_defer
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.memory_utils import clear_memory
from miles.utils.profile_utils import TrainProfiler
from miles.utils.timer import Timer
from miles.utils.tracking_utils.tracking import init_tracking
from miles.utils.workers.rpc.common.wire_types import Pickled

logger = logging.getLogger(__name__)


def configure_worker_cache(cache_dir: str | None, rank: int) -> dict[str, str]:
    if not cache_dir:
        return {}
    root = Path(cache_dir) / f"{socket.gethostname()}-rank-{rank:03d}"
    values = {
        "TILELANG_CACHE_DIR": str(root / "tilelang"),
        "TILELANG_TMP_DIR": str(root / "tilelang/tmp"),
        "TRITON_CACHE_DIR": str(root / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(root / "inductor"),
        "TORCH_EXTENSIONS_DIR": str(root / "extensions"),
    }
    for value in values.values():
        Path(value).mkdir(parents=True, exist_ok=True)
    os.environ.update(values)
    return values


class DModelTrainRayActor(TorchNativeTrainRayActor):
    backend_name = "dmodel"

    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Pickled,
        role: str,
        *,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
        recv_ckpt_src_rank: int | None = None,
        indep_dp_info: IndepDPInfo,
        indep_dp_store_addr: str | None,
    ) -> int | None:  # type: ignore[override]
        validate_dmodel_args(args)
        if with_ref or with_opd_teacher:
            raise ValueError("the initial DModel backend does not build reference or OPD models")
        if recv_ckpt_src_rank is not None:
            raise ValueError("the initial DModel backend does not support checkpoint healing")
        if indep_dp_info.quorum_id != 0 or indep_dp_store_addr is not None:
            raise ValueError("the initial DModel backend does not support independent DP")

        super()._init_common(args, role, with_ref, with_opd_teacher=with_opd_teacher)

        self.prof = TrainProfiler(args)
        self.load_hf_assets()
        routing_replay.enable(args)
        configure_worker_cache(args.dmodel_cache_dir, dist.get_rank())

        self.engine = DModelEngine(args)
        self.model = self.engine.model
        self.optimizer = self.engine.optimizer
        self.lr_scheduler = self.engine.scheduler
        self.model_parts = [self.model]
        self.optimizers = [self.optimizer]
        set_parallel_state(create_dmodel_parallel_state(self.engine.parallelism))

        if get_parallel_state().is_metrics_rank:
            init_tracking(args, primary=False)

        self.global_step = 0
        self.micro_step = 0
        self._update_metrics = {}
        checkpoint_payload = checkpoint.load(self)
        checkpoint.finalize_load(self, checkpoint_payload)
        self.dmodel_gc = ManualGarbageCollector(
            enabled=self.engine.config.training.manual_gc,
            interval=self.engine.config.training.manual_gc_interval,
        )
        self._compile_warmup_step = self.global_step + 1 if self.engine.config.compile.enable else None
        if not args.debug_train_only:
            from miles.backends.dmodel_utils.hf_weight_iterator import DModelHfWeightIterator

            self.weight_updater = self._build_weight_updater(self.model, DModelHfWeightIterator.build)

        self.prof.on_init_end()
        return int(args.start_rollout_id or 0)

    def _step_runner(self) -> DModelStepRunner:
        return DModelStepRunner(
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.engine.output_head,
            max_grad_norm=self.engine.config.training.max_grad_norm,
            sync_every_microbatch=self.engine.config.training.gradient_accumulation_sync_grad,
        )

    def _before_optimizer_steps(self, rollout_id: int, rollout_data: dict) -> None:
        del rollout_id, rollout_data
        clear_memory()

    @contextmanager
    def _actor_train_context(self, rollout_data: dict):
        self._update_metrics = {}
        with self.dmodel_gc.training():
            torch.cuda.synchronize()
            with Timer().context("actor_train"):
                started = perf_counter()
                yield
                torch.cuda.synchronize()
                seconds = perf_counter() - started
        self._update_metrics = collect_update_metrics(
            seconds, rollout_data.get("training_token_counts"), group=get_gloo_group()
        )

    def _extra_perf_metrics(self) -> dict:
        return self._update_metrics

    def _after_optimizer_step(self, microbatches: int) -> None:
        if self.args.dmodel_gpu_update_offload and not self.args.fsdp_cpu_offload:
            self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        self.micro_step += microbatches
        self.dmodel_gc.collect_between_steps(
            self.global_step, after_compile_warmup=self.global_step == self._compile_warmup_step
        )

    def _verify_weight_version(self, engines):
        version = self.weight_updater.weight_version
        receipt = [None]
        if dist.get_rank() == 0:
            try:
                receipt[0] = dict(passed=True, **async_utils.run(acknowledge_versions(engines, version)))
            except Exception as error:
                receipt[0] = dict(passed=False, error_type=type(error).__name__, version=version)
        dist.broadcast_object_list(receipt, src=0, group=get_gloo_group())
        if not receipt[0]["passed"]:
            raise RuntimeError(f"DModel policy publication was not acknowledged: {receipt[0]}")
        logger.info("DModel policy publication acknowledged: %s", receipt[0])

    def _save_checkpoint(self, rollout_id: int) -> None:
        checkpoint.save(self, rollout_id)

    @property
    def train_parallel_config(self) -> dict:
        size, _rank = self.engine.parallelism.get_batch_dp_info()
        return {"dp_size": size}
