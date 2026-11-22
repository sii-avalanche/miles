from argparse import Namespace
from collections.abc import Sequence

import torch

from miles.backends.training_utils.loss.hub.logit_processors import iter_response_rows, prepare_policy_logits
from miles.backends.training_utils.loss.hub.math_utils import calculate_log_probs_and_entropy
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.sampling_mask import RolloutSamplingMask


class DModelPolicyScores:
    """Response-only scores projected from DModel hidden states in bounded chunks."""

    def __init__(self, hidden: torch.Tensor, output_head, *, chunk_tokens: int = 4096):
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError(f"DModel hidden state must be [1, T, H], got {tuple(hidden.shape)}")
        self.hidden = hidden
        self.output_head = output_head
        self.chunk_tokens = chunk_tokens
        self._score_anchor: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        return self.hidden.device

    def sum(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        source = self._score_anchor if self._score_anchor is not None else self.hidden
        return source.sum(dtype=dtype) * 0

    def get_log_probs_and_entropy(
        self,
        *,
        args: Namespace,
        unconcat_tokens: list[torch.Tensor],
        total_lengths: list[int],
        response_lengths: list[int],
        with_entropy: bool,
        entropy_requires_grad: bool,
        non_loss_data: bool,
        max_seq_lens: list[int] | None,
        rollout_sampling_mask: Sequence[RolloutSamplingMask] | None,
    ) -> dict[str, list[torch.Tensor]]:
        if not non_loss_data:
            raise ValueError("DModel policy scores require the common non-loss-data path")
        if args.qkv_format != "thd":
            raise ValueError("DModel policy scores require THD")
        if max_seq_lens is not None:
            raise ValueError("DModel policy scores do not consume BSHD max sequence lengths")
        if rollout_sampling_mask is not None:
            raise ValueError("DModel policy scores do not yet support sampling-support replay")

        for tokens, total, response in zip(unconcat_tokens, total_lengths, response_lengths, strict=True):
            if total != tokens.numel() or not 0 <= response < total:
                raise ValueError("a response cannot begin at the first packed token of any sample")
        chunks = list(iter_response_rows(
            self.hidden.squeeze(0), args=args, unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths, response_lengths=response_lengths,
        ))
        selected_hidden = torch.cat([hidden for hidden, _, _ in chunks])
        label_tensor = torch.cat([labels for _, labels, _ in chunks]).to(device=self.device, dtype=torch.long)
        split_sizes = [labels.numel() for _, labels, _ in chunks]

        def project(_input_index: int, token_offset: int, logits: torch.Tensor):
            # An empty CP slice still returns a differentiable head output so
            # FSDP installs its pre-backward hook and reduces a zero gradient.
            if logits.shape[0] == 0:
                empty = logits.sum(dim=-1)
                entropy = (empty if entropy_requires_grad else empty.detach()) if with_entropy else None
                return empty, entropy
            selected_tokens = label_tensor[token_offset : token_offset + logits.shape[0]]
            logits = prepare_policy_logits(args, logits)
            log_prob, entropy = calculate_log_probs_and_entropy(
                logits,
                selected_tokens,
                get_parallel_state().tp.group,
                with_entropy=with_entropy,
                entropy_requires_grad=entropy_requires_grad,
                chunk_size=args.log_probs_chunk_size,
                true_on_policy=args.true_on_policy_mode,
                vocab_size=getattr(args, "vocab_size", None),
                temperature=1.0 if args.true_on_policy_mode else args.rollout_temperature,
                debug_unified_grad_fused_logprob=args.debug_unified_grad_fused_logprob,
            )
            return log_prob.reshape(-1), entropy.reshape(-1) if with_entropy else None

        # Consume the actual Module.__call__ outputs, not tensors captured by a
        # closure: FSDP may attach hooks or replace these returned tensors.
        (projected,) = self.output_head.chunked_projection((selected_hidden,), project, chunk_tokens=self.chunk_tokens)
        log_probs = torch.cat([log_prob for log_prob, _ in projected])
        self._score_anchor = log_probs
        if log_probs.numel() != sum(split_sizes):
            raise RuntimeError(f"projected {log_probs.numel()} response rows, expected {sum(split_sizes)}")
        result = {"log_probs": list(log_probs.split(split_sizes))}
        if with_entropy:
            entropy = torch.cat([entropy for _, entropy in projected])
            result["entropy"] = list(entropy.split(split_sizes))
        return result
