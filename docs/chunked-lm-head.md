# Experimental chunked FP32 LM head

`--chunked-lm-head-size 1024` chunks the vocabulary projection along tokens and
recomputes each chunk in backward. The default, `0`, preserves the normal logits
path. Unlike `--log-probs-chunk-size`, this avoids materializing the full
`[tokens, local_vocab]` output before computing sampled-token logprobs.

The existing response slicer and policy/DPPO objective are reused. Temperature
and next-token shifting are applied once; terminal and padding predictions do
not become response loss terms. The primitive saves hidden states, weights and
targets, not full vocabulary activations. Weight gradients and hidden states
still occupy memory; this does not make all training memory constant in length.

## Requirements and scope

- Megatron's public `GPTModel.forward(output_processor=...)` interface. Validated
  against `sii-avalanche/Megatron-LM` commit
  `d3c4640c5e7b34c73526224eab60569bcd5305e1`; no Megatron changes are required.
- An already configured **FP32, untied, bias-free output weight**, BF16 or FP32
  hidden states, packed THD with one packed stream `[1, T]`, PP=CP=1.
  This option does not silently change model/parameter precision.
- Policy loss only; TP and SP are supported. No critic, entropy (including
  observation), MTP, true-on-policy, sampling-support replay, OPD, witness,
  LoRA/multi-LoRA, FP8/FP4/FP16, MuP, CUDA graphs, deferred weight gradients, or
  autocast. Unsupported combinations raise rather than changing run settings.
- First-order autograd only. The adapter preserves the BF16 logit-gradient
  rounding in the pinned Megatron fused vocabulary CE, including the position of
  rounding relative to temperature scaling. This is not a promise of bitwise
  equality across GPU GEMM kernels or other Megatron versions.

The pinned Megatron also has fused linear cross entropy, but its implementation
selects only compute capability 10 (Blackwell); it is not a drop-in H200 path.
The normal Miles policy forward uses `labels=None` and sampled logprobs, so it
still materializes logits without this adapter. No Liger/Triton dependency is
added by this patch.

## CPU regression

Use an explicit supported Megatron checkout for standalone Miles CI. The tests
fail if the dependency is absent; core integration checks are not silently
skipped. In an AvaTrain checkout the sibling `Megatron-LM` is the default.

```bash
MEGATRON_ROOT=/path/to/Megatron-LM \
FLASHINFER_WORKSPACE_BASE=/tmp/chunked-lm-head-cache \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 GLOO_SOCKET_IFNAME=lo \
python3 -m pytest -q -o addopts='' -p no:cacheprovider \
  --confcutdir=tests/fast/backends/training_utils \
  tests/fast/backends/training_utils/test_chunked_lm_head.py
```

The CPU tests compare sampled logprobs, DPPO masks, gradients and one AdamW
update, run gradcheck, check saved tensors and packed boundaries, and exercise
two-rank Gloo TP/SP mappings. They also execute the pinned Megatron output-hook
and DDP accumulation methods in isolated CPU fixtures. Those fixtures test the
method contracts, not a complete GPU model or distributed optimizer.

## Validation limits

A previous prototype's H200 TP=2 projection-only benchmark at 65,536 tokens and
chunk size 1,024 reported peak allocated memory 94.43 -> 6.05 GiB, with
forward/backward time 4.27 -> 5.73 seconds. A historical full learner update and
checkpoint reload also passed in a different, modified development tree.
These are context, **not validation of this clean branch**.

This branch still needs a fixed-batch full-learner GPU comparison, distributed
optimizer integration, checkpoint save/reload, and long multi-step training.
It makes no claim that all OOMs are fixed or that end-to-end training is faster.
No parameter names or checkpoint state are introduced by the runtime callback.

## Manual small GPU A/B

Run only on GPUs selected by the operator, from the Miles checkout, with the
supported Megatron checkout on `PYTHONPATH`:

```bash
PYTHONPATH=.:/path/to/Megatron-LM \
  torchrun --standalone --nproc-per-node=2 tools/validate_chunked_lm_head.py \
  --device cuda --sequence-parallel --temperature 0.7
```

Repeat without `--sequence-parallel` to compare TP without SP. The check uses a
fixed seed and mask and compares every sampled logprob, DPPO TV mask, loss,
hidden/head gradient and one head-weight AdamW update. The FP32 dense reference
rounds scaled-logit gradients to BF16, matching the pinned CE contract. Reported
assertion tolerances are explicit in the script. This intentionally keeps a
small dense reference and is not a peak-memory benchmark or full learner test.
The CPU `--device cpu` variant also runs via two-rank Gloo; its SP fixture only
redirects Megatron's CUDA allocation selector to CPU.
