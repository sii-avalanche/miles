"""Continuous SEC state. Completion costs O(grid_size), independent of question count.

The lock protects completion, dispatch, telemetry and checkpoint access from the
rollout loop and RPC/monitor threads. No network or disk operation holds it.
"""

import copy
import hashlib
import json
import math
import random
import threading
import uuid
from collections import Counter, deque
from dataclasses import asdict

import numpy as np
import torch

from miles.rollout.curriculum_config import SECConfig
from miles.utils.group_advantages import normalize_group_rewards
from miles.utils.types import Sample

METADATA_KEY = "continuous_sec"
GROUP_SIZE = 16


def curriculum_records(args, records):
    """Retain curriculum fields through the existing Dataset/pre-filter pipeline."""
    for record in records:
        metadata = dict(record.get(args.metadata_key) or {})
        identity = record.get(args.curriculum_id_field, metadata.get(args.curriculum_id_field))
        value = record.get(
            args.curriculum_initial_probability_field, metadata.get(args.curriculum_initial_probability_field)
        )
        if identity is None or not isinstance(identity, (str, int)):
            raise ValueError(f"SEC requires a stable {args.curriculum_id_field!r} question ID")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Question {identity}: missing/nonnumeric initial probability")
        probability = value / args.curriculum_initial_probability_scale
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"Question {identity}: initial probability is outside the configured scale")
        content = [record[args.input_key], record.get(args.label_key)]
        metadata[METADATA_KEY] = {
            "question_id": str(identity),
            "initial_probability": probability,
            "content_hash": hashlib.sha256(
                json.dumps(content, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest(),
        }
        yield {**record, args.metadata_key: metadata}


def binary_group_rewards(group, reward_key):
    """One reward per episode, including generators returning several fragments."""
    if len(group) != GROUP_SIZE:
        return None
    rewards = []
    for episode in group:
        fragments = episode if isinstance(episode, list) else [episode]
        if not fragments or any(
            s.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED) or s.metadata.get("grader_invalid")
            for s in fragments
        ):
            return None
        values = [s.reward.get(reward_key) if isinstance(s.reward, dict) else s.reward for s in fragments]
        if any(value not in (0, 1) for value in values) or any(value != values[0] for value in values):
            return None
        rewards.append(float(values[0]))
    return rewards


class ContinuousSEC:
    def __init__(self, questions, *, config: SECConfig, seed: int, standardize: bool, quantile=None):
        self.config = config
        self.standardize = standardize
        self.quantile = quantile
        self._lock = threading.RLock()
        self._rng = random.Random(seed)
        self.grid = np.linspace(0.0, 1.0, config.grid_size)
        self.q = np.zeros(config.grid_size)
        self.questions = {}
        for row in questions:
            key, initial = row["question_id"], row["initial_probability"]
            if key in self.questions:
                raise ValueError(f"Duplicate SEC question ID: {key}")
            if not math.isfinite(initial) or not 0 <= initial <= 1:
                raise ValueError(f"Invalid SEC initial probability for {key}")
            self.questions[key] = {
                "initial_probability": initial,
                "p": initial,
                "n": 1,
                "observed_successes": 0,
                "observed_samples": 0,
                "content_hash": row.get("content_hash"),
            }
        if not self.questions:
            raise ValueError("SEC needs at least one eligible question")
        self._keys = tuple(self.questions)
        self._reserved = set()
        self._leases = {}
        self._lease_counts = Counter()
        self._inflight = {}
        self._seen = set()
        self.version = 0
        self.policy_version = None
        self._counts = Counter()
        self._windows = {
            name: deque(maxlen=4096) for name in ("sampled", "completed", "pass_rate", "signal")
        }
        self._distribution = {}

    def probabilities(self, keys=None):
        with self._lock:
            keys = list(keys) if keys is not None else [key for key in self._keys if key not in self._reserved]
            if not keys:
                return keys, np.array([], dtype=float)
            p = np.array([self.questions[key]["p"] for key in keys])
            values = np.interp(p, self.grid, self.q)
            # Center before dividing as well, so very small temperatures cannot overflow.
            scores = (values - values.max()) / self.config.temperature
            weights = np.exp(scores)
            weights /= weights.sum()
            weights = (1 - self.config.exploration) * weights + self.config.exploration / len(keys)
            return keys, weights

    def select(self, count):
        """Weighted draws without replacement among currently eligible questions."""
        selected = []
        with self._lock:
            for _ in range(count):
                keys, weights = self.probabilities()
                if not keys:
                    break
                key = self._rng.choices(keys, weights=weights, k=1)[0]
                selected.append(key)
                self._reserved.add(key)
                p = np.array([self.questions[k]["p"] for k in keys])
                self._distribution = {
                    "eligible_questions": len(keys),
                    "probability_min": float(weights.min()),
                    "probability_max": float(weights.max()),
                    "entropy": float(-np.sum(weights * np.log(weights.clip(1e-300)))),
                    "expected_p_correct": float(weights @ p),
                    "effective_questions": float(1 / np.sum(weights**2)),
                }
        return selected

    def reserve(self, key):
        with self._lock:
            self._reserved.add(key)

    def dispatch(self, prompt_group, policy_version):
        if policy_version is None:
            raise ValueError("SEC dispatch requires a published policy version")
        with self._lock:
            key = prompt_group[0].metadata[METADATA_KEY]["question_id"]
            event_id = uuid.uuid4().hex
            row = self.questions[key]
            stamp = {
                "question_id": key,
                "group_id": event_id,
                "difficulty_at_dispatch": row["p"],
                "initial_probability": row["initial_probability"],
                "difficulty_update_count": row["n"] - 1,
                "policy_version": policy_version,
                "curriculum_version": self.version,
            }
            for sample in prompt_group:
                sample.metadata = {**sample.metadata, METADATA_KEY: stamp.copy()}
            self._reserved.add(key)
            self._leases[event_id] = key
            self._lease_counts[key] += 1
            self._inflight[event_id] = GROUP_SIZE
            self.policy_version = policy_version
            self._counts["dispatched"] += 1
            self._windows["sampled"].append(row["p"])

    def complete(self, prompt_group, group, *, reward_key=None):
        stamp = prompt_group[0].metadata[METADATA_KEY]
        rewards = binary_group_rewards(group, reward_key)
        if len(prompt_group) != GROUP_SIZE or len({sample.index for sample in prompt_group}) != GROUP_SIZE:
            rewards = None
        fragments = [s for episode in group for s in (episode if isinstance(episode, list) else [episode])]
        versions = {str(span.version) for s in fragments for span in s.all_weight_version_spans}
        expected = stamp["policy_version"]
        if len(versions) > 1 or (versions and expected is not None and versions != {str(expected)}):
            self.failed(prompt_group)
            raise ValueError(f"SEC group {stamp['group_id']} crossed policy versions: {versions}, expected {expected}")
        with self._lock:
            event_id = stamp["group_id"]
            if event_id in self._seen:
                return False
            self._seen.add(event_id)
            self._inflight.pop(event_id, None)
            if rewards is None:
                self._counts["invalid"] += 1
                return False
            advantages = normalize_group_rewards(
                torch.tensor(rewards, dtype=torch.float32), standardize=self.standardize, quantile=self.quantile
            )
            signal = float(advantages.abs().mean())
            observed_p = sum(rewards) / GROUP_SIZE
            row = self.questions[stamp["question_id"]]
            row["p"] = (row["n"] * row["p"] + observed_p) / (row["n"] + 1)
            row["n"] += 1
            row["observed_successes"] += int(sum(rewards))
            row["observed_samples"] += GROUP_SIZE
            x = stamp["difficulty_at_dispatch"]
            kernel = np.exp(-0.5 * ((self.grid - x) / self.config.bandwidth) ** 2)
            self.q += self.config.alpha * kernel * (signal - self.q)
            self.version += 1
            self._counts["complete"] += 1
            self._windows["completed"].append(x)
            self._windows["pass_rate"].append(observed_p)
            self._windows["signal"].append(signal)
            for sample in fragments:
                sample.metadata = {
                    **sample.metadata,
                    METADATA_KEY: {
                        **stamp,
                        "observed_p": observed_p,
                        "signal": signal,
                        "p_after_completion": row["p"],
                        "difficulty_update_count": row["n"] - 1,
                        "curriculum_version_after_completion": self.version,
                    },
                }
            return True

    def failed(self, prompt_group):
        with self._lock:
            event_id = prompt_group[0].metadata[METADATA_KEY]["group_id"]
            if event_id not in self._seen:
                self._seen.add(event_id)
                self._counts["invalid"] += 1
            self._inflight.pop(event_id, None)
            self.release(prompt_group)

    def release(self, prompt_group):
        with self._lock:
            event_id = prompt_group[0].metadata[METADATA_KEY]["group_id"]
            if (key := self._leases.pop(event_id, None)) is not None:
                self._lease_counts[key] -= 1
                if not self._lease_counts[key]:
                    self._reserved.discard(key)
                    del self._lease_counts[key]

    def metrics(self):
        with self._lock:
            result = {
                "version": self.version,
                "update_count": self.version,
                "policy_version": self.policy_version if self.policy_version is not None else -1,
                "in_flight_groups": len(self._inflight),
                "in_flight_rollouts": sum(self._inflight.values()),
                "reserved_questions": len(self._reserved),
                "q/mean": float(self.q.mean()),
                "q/min": float(self.q.min()),
                "q/max": float(self.q.max()),
                **{
                    f"groups/{key}_total": self._counts[key]
                    for key in ("dispatched", "complete", "invalid")
                },
                **{f"distribution/{key}": value for key, value in self._distribution.items()},
            }
            for x in np.linspace(0, 1, 11):
                result[f"q/x_{x:.1f}"] = float(np.interp(x, self.grid, self.q))
            counts = [row["n"] - 1 for row in self.questions.values()]
            result.update(
                {
                    "question_updates/mean": float(np.mean(counts)),
                    "question_updates/min": min(counts),
                    "question_updates/max": max(counts),
                }
            )
            for name, values in self._windows.items():
                if values:
                    result[f"{name}/mean"] = float(np.mean(values))
                    for quantile in (10, 50, 90):
                        result[f"{name}/p{quantile}"] = float(np.percentile(values, quantile))
            return result

    def state_dict(self):
        with self._lock:
            return copy.deepcopy(
                {
                    "schema": 1,
                    "config": asdict(self.config),
                    "standardize": self.standardize,
                    "quantile": self.quantile,
                    "questions": self.questions,
                    "q": self.q.tolist(),
                    "version": self.version,
                    "seen": sorted(self._seen),
                    "rng": self._rng.getstate(),
                    "counts": dict(self._counts),
                    "windows": {key: list(value) for key, value in self._windows.items()},
                }
            )

    def load_state_dict(self, state):
        with self._lock:
            if state["schema"] != 1 or state["config"] != asdict(self.config):
                raise ValueError("SEC checkpoint configuration/schema mismatch")
            if state["standardize"] != self.standardize or state["quantile"] != self.quantile:
                raise ValueError("SEC checkpoint GRPO normalization mismatch")

            def identity(rows):
                return {key: (row["initial_probability"], row["content_hash"]) for key, row in rows.items()}

            if identity(state["questions"]) != identity(self.questions):
                raise ValueError("SEC checkpoint question IDs/content/initial probabilities do not match the dataset")
            self.questions = copy.deepcopy(state["questions"])
            self.q = np.array(state["q"], dtype=float)
            if self.q.shape != self.grid.shape or not np.all(np.isfinite(self.q)):
                raise ValueError("Invalid continuous Q grid in SEC checkpoint")
            self.version = state["version"]
            self._seen = set(state["seen"])
            self._rng.setstate(state["rng"])
            self._counts = Counter(state["counts"])
            self._windows = {key: deque(state["windows"].get(key, []), maxlen=4096) for key in self._windows}
            # Network requests / finished-group buffers do not survive a process restart.
            # Keep feedback already checkpointed; release their reservations for fresh generation.
            self._reserved.clear()
            self._leases.clear()
            self._lease_counts.clear()
            self._inflight.clear()
