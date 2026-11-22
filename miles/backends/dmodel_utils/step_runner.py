import math
from collections.abc import Callable, Iterable

import torch
import torch.distributed as dist

from miles.backends.dmodel_utils.batch import dmodel_forward_kwargs
from miles.backends.dmodel_utils.policy_scores import DModelPolicyScores
from miles.backends.training_utils.torch_native.step_runner import StepMetrics
from miles.utils.distributed_utils import get_gloo_group

class DModelStepRunner:
    """Drive RL forwards/backwards while preserving DModel FSDP sync semantics."""

    def __init__(
        self, model, optimizer, scheduler, output_head, *, max_grad_norm: float, sync_every_microbatch: bool = False
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.output_head = output_head
        self.max_grad_norm = max_grad_norm
        self.sync_every_microbatch = sync_every_microbatch

    def _forward(self, batch: dict) -> DModelPolicyScores:
        hidden = self.model(**dmodel_forward_kwargs(batch))
        if not isinstance(hidden, torch.Tensor):
            raise TypeError(f"DModel policy forward must return hidden states, got {type(hidden).__name__}")
        return DModelPolicyScores(hidden, self.output_head)

    def forward_only_step(self, batches: Iterable[dict], compute: Callable) -> list:
        was_training = self.model.training
        self.model.eval()
        try:
            return [compute(self._forward(batch), batch) for batch in batches]
        finally:
            self.model.train(was_training)

    def forward_backward_step(self, batches: Iterable[dict], loss_closure: Callable) -> list[dict]:
        from dmodel.training.rl import finish_gradient_accumulation, prepare_gradient_accumulation

        batches = list(batches)
        logs = []
        for index, batch in enumerate(batches):
            prepare_gradient_accumulation(
                self.model, index, len(batches), sync_every_microbatch=self.sync_every_microbatch
            )
            scores = self._forward(batch)
            loss, log_dict = loss_closure(scores, batch)
            loss.backward()
            logs.append(log_dict)
        finish_gradient_accumulation(self.model)
        return logs

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def apply_step(self) -> StepMetrics:
        from dmodel.utils.grad_clip import clip_grad_norm_mixed_

        grad_norm = clip_grad_norm_mixed_(self.model.parameters(), self.max_grad_norm)
        value = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        finite = torch.tensor(int(math.isfinite(value)), dtype=torch.int32)
        if dist.is_initialized():
            dist.all_reduce(finite, op=dist.ReduceOp.MIN, group=get_gloo_group())
        if not finite.item():
            raise FloatingPointError(
                "DModel gradient norm is non-finite; optimizer and scheduler were not stepped"
            )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        lrs = self.scheduler.get_last_lr() if self.scheduler is not None else []
        return StepMetrics(
            grad_norm=value,
            extra_metrics={f"lr/group_{index}": lr for index, lr in enumerate(lrs)},
        )
