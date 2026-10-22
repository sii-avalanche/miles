# GRPO / DAPO / Dr.GRPO presets

Select `--rl-algorithm grpo_original`, `dapo`, or `dr_grpo` with the Megatron
backend. AvaTrain exposes the same setting as `rl_algorithm` under `[train]`.
Omitting it retains the existing Miles defaults. Explicit component flags win
over preset defaults.

| Preset | Group reward normalization | PPO clip | Policy loss reduction | Raw reward filter | Overlong reward |
| --- | --- | --- | --- | --- | --- |
| `grpo_original` | Center / sample std | 0.8–1.2 | Response mean | Off | None |
| `dapo` | Center / sample std | 0.8–1.28 | Global token mean | Nonzero group std | Soft penalty |
| `dr_grpo` | Center only | 0.8–1.2 | Sum / (responses × fixed C) | Off | None |

The existing sample-standard-deviation convention is preserved. Centering is
performed before DP partitioning using prompt group identities, with one shared
advantage for all fragments of a rollout. Zero-variance groups have zero advantage.

For example, in an existing AvaTrain TOML:

```toml
[train]
rl_algorithm = "dapo"
rollout_max_response_len = 250000
overlong_buffer_length = 50000
overlong_penalty_strength = 1.0
dynamic_sampling_max_groups = 160 # target batch: 16 accepted groups
```

This subtracts `clip((generated_length - 200000) / 50000, 0, 1)` from the
original reward. Filtering uses the original reward first, retaining complete
groups and continuing generation until the target is met. The sampling budget
counts completed groups per target batch, including rejections, and raises a
visible error on exhaustion. In fully async mode it bounds producer acceptance
windows; buffered groups may subsequently be rejected for staleness. It is not
a wall-clock timeout for hung generation requests.

For Dr.GRPO, change `rl_algorithm = "dr_grpo"`; C defaults to the fixed
`rollout_max_response_len`, or set `policy_loss_fixed_normalizer` explicitly.
The fixed divisor applies to policy loss; entropy and KL retain their existing
separate reducers and coefficients. The legacy custom reducer entrypoint still
works with its original C=1000, now using the same CP-aware core reducer.

Independent overrides reuse `eps_clip`, `eps_clip_high`,
`disable_grpo_std_normalization` / `grpo_std_normalization`, and
`dynamic_sampling_filter_path` (empty string disables it). New reduction choices
are `response_mean`, `token_mean`, `group_token_mean`, and `fixed_token_sum`.
`group_token_mean` averages per-prompt token means and currently requires one
optimizer step over the complete rollout batch. Denominators are computed before
DP/CP/microbatch splitting; importance-sampling rejection masks change numerators
without changing those denominators. `token_mean` uses Megatron's global token
normalization across gradient accumulation, DP and CP, including zero-token masks.

`overlong_reward_mode` can independently be `none`, `soft_penalty`, or
`mask_truncated`. Only the last mode zeros loss masks for length-truncated
rollouts (including sibling fragments); normal stop/EOS responses stay enabled.
Do not turn the two strategies on implicitly together. Custom reward/data
conversion hooks and the legacy rollout API are rejected for named recipes,
rather than silently bypassing their semantics.

Monitoring reuses existing reward, entropy, gradient-norm, length and truncation
metrics, adding group acceptance, centered-advantage statistics, positive/negative
clip fractions, and length-penalty / shaped-reward statistics. Benchmark accuracy
still requires an evaluation dataset; EOS versus explicit stop sequence is not
inferred from the shared COMPLETED status.

The preset supplies algorithm components, not an exact paper reproduction:
existing reward encoding, TIS, entropy/KL and MTP remain independently configured.
In particular, GLM 0/1 judging and fully asynchronous rollout differ from the
[DAPO experiment recipe](https://dapo-sia.github.io/). Dr.GRPO follows the
[fixed-denominator, unscaled-advantage formulation](https://arxiv.org/abs/2503.20783).

Validation: `python -m pytest tests/fast/test_rl_algorithms.py` covers recipe
precedence, group identity, clipping gradients, masked reducers under virtual
DP/zigzag CP partitions and uneven accumulation, group-preserving filtering,
sampling exhaustion, and overlong shaping. CPU numerical tests do not establish
GPU kernel, NCCL, optimizer or full-training readiness.
