"""Post-GRPO credit policies. Scores express desirability, never ready-made advantages."""

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Protocol

import torch

from miles.utils.function_registry import load_function


@dataclass(frozen=True)
class CreditConfig:
    mode: str = "none"
    strength: float = 0.5
    min_confidence: float = 0.5
    policy_path: str | None = None

    def __post_init__(self):
        if self.mode not in ("none", "reweight"):
            raise ValueError("Credit shaping supports none/reweight; signed is a separate, deferred experiment")
        for name in ("strength", "min_confidence"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"credit {name} must be in [0, 1]")
        if self.mode == "none" and self.policy_path:
            raise ValueError("A custom credit policy requires reweight mode")

    @classmethod
    def from_args(cls, args):
        return cls(
            **{
                name: getattr(args, f"credit_{name}", field.default)
                for name, field in cls.__dataclass_fields__.items()
            }
        )


class AdvantageShapingPolicy(Protocol):
    def shape(self, sequence_advantage: float, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return full-response token advantages; called AFTER GRPO normalization."""
        ...


@dataclass(frozen=True)
class ReweightPolicy:
    strength: float = 0.5

    def shape(self, sequence_advantage, scores, mask):
        weights = torch.ones_like(scores, dtype=torch.float64)
        valid = mask.bool()
        if sequence_advantage == 0 or not valid.any() or self.strength == 0:
            return weights * sequence_advantage
        centered = scores[valid].double() - scores[valid].double().mean()
        centered -= centered.mean()
        centered /= centered.abs().max().clamp_min(1.0)
        # Positive desirability strengthens positive A and weakens negative A.
        direction = 1.0 if sequence_advantage > 0 else -1.0
        weights[valid] += self.strength * direction * centered
        return weights * sequence_advantage


@dataclass(frozen=True)
class NoShapingPolicy:
    def shape(self, sequence_advantage, scores, mask):
        return torch.full_like(scores, sequence_advantage, dtype=torch.float64)


def make_policy(config: CreditConfig) -> AdvantageShapingPolicy:
    if config.policy_path:
        return load_function(config.policy_path)(config)
    return NoShapingPolicy() if config.mode == "none" else ReweightPolicy(config.strength)


def checked_coefficients(policy, config, advantage, scores, mask):
    """Validate plugins as well as builtins. Clipping weights afterward would break mass."""
    shaped = policy.shape(advantage, scores, mask)
    if shaped.shape != scores.shape or not torch.isfinite(shaped).all():
        raise ValueError("Credit policy returned invalid token advantages")
    if advantage == 0 and torch.any(shaped):
        raise ValueError("Credit policy must preserve zero-advantage behavior")
    weights = shaped / advantage if advantage != 0 else torch.ones_like(shaped)
    valid = mask.bool()
    if weights.shape != scores.shape or not torch.isfinite(weights).all():
        raise ValueError("Credit policy returned invalid coefficients")
    bound = config.strength if config.mode != "none" else 0.0
    if ((weights < 1 - bound - 1e-10) | (weights > 1 + bound + 1e-10)).any():
        raise ValueError("Credit policy coefficients exceed the configured bounds")
    if not torch.equal(weights[~valid], torch.ones_like(weights[~valid])):
        raise ValueError("Credit policy changed non-trainable tokens")
    if valid.any() and not torch.allclose(weights[valid].double().sum(), valid.double().sum(), atol=1e-9, rtol=1e-10):
        raise ValueError("Credit policy does not preserve per-sequence coefficient mass")
    if advantage == 0 and not torch.equal(weights, torch.ones_like(weights)):
        raise ValueError("Credit policy must preserve zero-advantage behavior")
    if not torch.any(scores) and not torch.equal(weights, torch.ones_like(weights)):
        raise ValueError("Credit policy must preserve no-feedback behavior")
    return weights


def token_fingerprint(tokens) -> str:
    """Wire-contract hash, shared with AvaCore; never tokenize text to recover coordinates."""
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    return hashlib.sha256(json.dumps(list(tokens), separators=(",", ":")).encode()).hexdigest()


def span_scores(payload, tokens, mask, *, min_confidence):
    """Mean confidence-scaled desirability on overlaps; identical spans count only once."""
    scores = torch.zeros(len(mask), dtype=torch.float64)
    if payload is None:
        return scores, []
    if not isinstance(payload, dict) or payload.get("version") != 1:
        return scores, ["unsupported credit payload"]
    if payload.get("token_hash") != token_fingerprint(tokens) or payload.get("response_length") != len(mask):
        return scores, ["tokens changed since resolution; ignored stale credit coordinates"]
    spans = payload.get("spans", [])
    if not isinstance(spans, list) or len(spans) > 16384:
        return scores, ["invalid resolved span list"]
    sums, counts = torch.zeros(len(mask) + 1, dtype=torch.float64), torch.zeros(len(mask) + 1, dtype=torch.float64)
    errors, seen = [], set()
    for i, span in enumerate(spans):
        try:
            start, end, score, confidence = (span[key] for key in ("start", "end", "score", "confidence"))
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(mask):
                raise ValueError("invalid offsets")
            for name, value, lower in (("score", score, -1), ("confidence", confidence, 0)):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not lower <= value <= 1
                ):
                    raise ValueError(f"invalid {name}")
            if confidence < min_confidence or confidence == 0:
                continue
            key = (start, end, score, confidence)
            if key in seen:
                continue
            seen.add(key)
            sums[start] += score * confidence
            sums[end] -= score * confidence
            counts[start] += 1
            counts[end] -= 1
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"span[{i}]: {error}")
    coverage = counts.cumsum(0)[:-1]
    scores = (sums.cumsum(0)[:-1] / coverage.clamp_min(1)).clamp(-1, 1)
    scores[coverage == 0] = 0
    scores[~mask.bool()] = 0
    return scores, errors


def add_credit_arguments(parser):
    group = parser.add_argument_group("Post-rollout span credit assignment")
    group.add_argument("--credit-mode", choices=["none", "reweight"], default="none")
    group.add_argument("--credit-strength", type=float, default=0.5, help="Coefficient bounds are 1 +/- strength.")
    group.add_argument("--credit-min-confidence", type=float, default=0.5)
    group.add_argument(
        "--credit-policy-path", default=None, help="Optional factory(config) implementing AdvantageShapingPolicy."
    )
    group.add_argument(
        "--credit-audit-dir", default=None, help="JSONL records; defaults to RUN_DIR/credit_assignment."
    )
    return parser


def validate_credit_args(args):
    config = CreditConfig.from_args(args)
    if config.mode == "none":
        return
    if args.advantage_estimator != "grpo" or not args.rewards_normalization or args.loss_type != "policy_loss":
        raise ValueError("Span reweight requires GRPO group advantages and policy_loss")
    if args.use_opd or getattr(args, "multi_lora", False):
        raise ValueError("Span reweight currently excludes OPD and multi-LoRA objectives")
