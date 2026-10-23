"""Apply span policies after baseline normalization and before the unchanged PPO loss."""

import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from miles.backends.training_utils.data.context_parallel import get_local_response_loss_masks, slice_log_prob_with_cp
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.credit_assignment import CreditConfig, checked_coefficients, make_policy, span_scores


def shape_token_advantages(args, rollout_data, advantages):
    config = CreditConfig.from_args(args)
    payloads = rollout_data.get("credit_assignment")
    if payloads is None and config.mode == "none":
        return advantages
    payloads = payloads or [None] * len(advantages)
    state = get_parallel_state()
    masks = get_local_response_loss_masks(
        rollout_data["total_lengths"],
        rollout_data["response_lengths"],
        rollout_data["loss_masks"],
        args.qkv_format,
        rollout_data.get("max_seq_lens"),
    )
    # All CP ranks must see the same scalar, including ranks owning only prompt tokens.
    stats = torch.stack(
        [
            torch.stack((a.double()[m.to(a.device).bool()].sum(), m.to(a.device, dtype=torch.float64).sum()))
            for a, m in zip(advantages, masks, strict=True)
        ]
    )
    if state.cp.size > 1:
        dist.all_reduce(stats, group=state.cp.group)
    scalars = (stats[:, 0] / stats[:, 1].clamp_min(1)).tolist()
    policy = make_policy(config)
    shaped, audit = [], []
    max_lengths = rollout_data.get("max_seq_lens") or [None] * len(advantages)
    for i, (original, payload, scalar, max_length) in enumerate(
        zip(advantages, payloads, scalars, max_lengths, strict=True)
    ):
        full_mask = rollout_data["loss_masks"][i].detach().cpu()
        scores, errors = span_scores(
            payload,
            rollout_data["tokens"][i],
            full_mask,
            min_confidence=config.min_confidence,
        )
        # An all-equal reward group must never acquire local signal through whitening.
        if rollout_data["rewards"][i] == 0:
            scores = torch.zeros_like(scores)
        weights = checked_coefficients(policy, config, scalar, scores, full_mask)
        local_weights = slice_log_prob_with_cp(
            weights,
            rollout_data["total_lengths"][i],
            rollout_data["response_lengths"][i],
            args.qkv_format,
            max_length,
        ).to(original.device, dtype=original.dtype)
        final = original if config.mode == "none" or not torch.any(scores) else original * local_weights
        shaped.append(final)
        audit.append(
            {
                "version": 1,
                "time_ns": time.time_ns(),
                "sample_index": rollout_data.get("sample_indices", [None] * len(advantages))[i],
                "mode": config.mode,
                "strength": config.strength,
                "min_confidence": config.min_confidence,
                "original_group_advantage": float(rollout_data["rewards"][i]),
                "baseline_sequence_advantage": scalar,
                "feedback": payload,
                "errors": errors,
                "cp_rank": state.cp.rank,
                "cp_size": state.cp.size,
                "total_length": rollout_data["total_lengths"][i],
                "response_length": rollout_data["response_lengths"][i],
                "qkv_format": args.qkv_format,
                "max_seq_len": max_length,
                "loss_mask": masks[i].tolist(),
                "original_token_advantages": original.tolist(),
                "coefficients": local_weights.tolist(),
                "final_token_advantages": final.tolist(),
            }
        )
    rollout_data["credit_audit"] = audit
    _write_audit(args, audit)
    return shaped


def _write_audit(args, records):
    root = getattr(args, "credit_audit_dir", None)
    if not root:
        root = str(Path(os.environ.get("RUN_DIR") or getattr(args, "save", None) or ".") / "credit_assignment")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank() if dist.is_initialized() else 0
    # Each process owns its file; CP shards carry reconstruction coordinates and masks.
    with (path / f"rank-{rank}-pid-{os.getpid()}.jsonl").open("a") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
