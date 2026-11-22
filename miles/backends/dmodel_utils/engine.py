import logging

import torch

from miles.backends.dmodel_utils.config import load_dmodel_config

logger = logging.getLogger(__name__)


class DModelEngine:
    """Construct DModel through its native model/parallel/optimizer assembly."""

    def __init__(self, args):
        import dmodel.operators as op_glue
        from dmodel.models.layers.moe import freeze_moe_routing
        from dmodel.parallelism.compile import configure_compile, maybe_compile_muon
        from dmodel.train import (
            apply_config_overrides,
            apply_parallelism,
            build_meta_model,
            materialise_and_init_weights,
            resolve_model_config,
        )
        from dmodel.training.chunked_loss import get_chunked_cross_entropy_head
        from dmodel.training.optimizer import create_optimizer_and_scheduler
        from dmodel.utils.train_utils import configure_determinism, set_seed

        self.config = load_dmodel_config(args)
        configure_determinism(self.config.system.deterministic)
        set_seed(self.config.training.seed)
        configure_compile(self.config.compile)

        device = f"cuda:{torch.cuda.current_device()}"
        op_glue.load_dispatch(
            self.config.get("operators", None),
            context={
                "device": device,
                "backend": self.config.compile.backend,
                "tp": self.config.parallelism.tp,
            },
            logger=logger if torch.distributed.get_rank() == 0 else None,
        )

        self.family, self.model_config = resolve_model_config(self.config)
        self.model_config = apply_config_overrides(self.config, self.model_config)
        model = build_meta_model(self.config, self.family, self.model_config)
        op_glue.refresh_report()
        if torch.distributed.get_rank() == 0:
            logger.info("DModel operator binding report:\n%s", op_glue.report().describe())

        self.model, self.parallelism = apply_parallelism(
            model,
            self.config,
            torch.distributed.get_world_size(),
        )
        materialise_and_init_weights(self.model, self.config, device, self.family, self.model_config)
        freeze_moe_routing(
            self.model,
            freeze_gate=args.dmodel_freeze_router_gate,
            freeze_bias=args.dmodel_freeze_expert_bias,
        )
        self.model.train()
        self.output_head = get_chunked_cross_entropy_head(self.model)
        if self.output_head is None:
            raise RuntimeError("DModel chunked output head was not installed")
        self.optimizer, self.scheduler = create_optimizer_and_scheduler(
            model=self.model,
            config=self.config.optimizer,
            foreach=False,
            parallelism_config=self.parallelism,
            total_steps=None,
            cpu_state_bucket_bytes=(args.dmodel_gpu_update_bucket_mb << 20) if getattr(args, "dmodel_gpu_update_offload", False) else None,
            muon_prefetch=getattr(args, "dmodel_muon_prefetch", False),
        )
        maybe_compile_muon(self.optimizer, self.config.compile)
