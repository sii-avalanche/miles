"""Manual small projection A/B check; no Ray, model loading or checkpoint writes.

Run with torchrun on user-selected idle GPUs. This retains the dense reference
and is a numerical check, not a large-sequence memory benchmark.
"""

import argparse
import json
import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from miles.backends.megatron_utils.chunked_lm_head import _project
from miles.backends.training_utils.loss.hub.math_utils import compute_dppo_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--sequence-parallel", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--vocab", type=int, default=256)
    parser.add_argument("--chunk", type=int, default=17)
    args = parser.parse_args()
    if min(args.tokens, args.hidden, args.vocab, args.chunk) <= 0:
        parser.error("Dimensions must be positive")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if args.tokens % world or args.vocab % world:
            raise ValueError("Tokens and vocabulary must be divisible by TP size")
        torch.manual_seed(42)
        full_h = torch.randn(args.tokens, 1, args.hidden, device=device, requires_grad=True)
        full_w = (torch.randn(args.vocab, args.hidden, device=device) * 0.1).requires_grad_()
        ids = torch.randint(args.vocab, (1, args.tokens), device=device)
        local_h = full_h.detach().chunk(world)[rank] if args.sequence_parallel else full_h.detach()
        h = local_h.clone().requires_grad_()
        w = full_w.detach().chunk(world)[rank].clone().requires_grad_()
        layer = SimpleNamespace(
            bias=None, weight=w, tp_group=dist.group.WORLD, sequence_parallel=args.sequence_parallel
        )
        # CPU fixtures redirect the CUDA allocation selector in Megatron SP mappings only.
        mapping_device = (
            patch("torch.cuda.current_device", return_value=device) if device.type == "cpu" else nullcontext()
        )
        with mapping_device:
            actual = _project(h, layer, None, ids, chunk_size=args.chunk, temperature=args.temperature).flatten()
            logits = (full_h[:, 0] @ full_w.t()) / args.temperature
            # Pinned fused vocabulary CE rounds at this point, before temperature backward.
            logits.register_hook(lambda g: g.to(torch.bfloat16).float())
            reference = logits.log_softmax(-1).gather(1, ids.flatten().roll(-1)[:, None]).flatten()
            torch.testing.assert_close(actual, reference, atol=3e-5, rtol=3e-5)
            old = reference.detach() + torch.linspace(-1, 1, args.tokens, device=device)
            advantages = torch.linspace(-1, 1, args.tokens, device=device)
            mask = (torch.arange(args.tokens, device=device) % 3 != 0).float()
            loss, blocked, _ = compute_dppo_loss(old, actual, advantages, 0.1, "tv")
            expected, expected_blocked, _ = compute_dppo_loss(old, reference, advantages, 0.1, "tv")
            torch.testing.assert_close(blocked, expected_blocked, atol=0, rtol=0)
            loss = (loss * mask).sum() / mask.sum()
            expected = (expected * mask).sum() / mask.sum()
            torch.testing.assert_close(loss, expected, atol=3e-5, rtol=3e-5)
            loss.backward()
            expected.backward()
        expected_h_grad = full_h.grad.chunk(world)[rank] if args.sequence_parallel else full_h.grad
        torch.testing.assert_close(h.grad, expected_h_grad, atol=3e-5, rtol=3e-3)
        torch.testing.assert_close(w.grad, full_w.grad.chunk(world)[rank], atol=3e-5, rtol=3e-3)
        torch.optim.AdamW([w], lr=1e-4).step()
        torch.optim.AdamW([full_w], lr=1e-4).step()
        torch.testing.assert_close(w, full_w.chunk(world)[rank], atol=3e-6, rtol=3e-5)
        print(
            json.dumps(
                {
                    "result": "PASS",
                    "scope": "small_projection_ab",
                    "rank": rank,
                    "tp": world,
                    "sp": args.sequence_parallel,
                    "dtype": str(w.dtype),
                    "temperature": args.temperature,
                    "tokens": args.tokens,
                    "chunk": args.chunk,
                    "torch": torch.__version__,
                    "loss": loss.item(),
                    "max_logprob_error": (actual - reference).abs().max().item(),
                }
            )
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
