"""Compact implementations of DARE and DOTS online curriculum algorithms."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
import threading
import uuid
from collections import Counter, deque
from dataclasses import asdict, dataclass

import numpy as np

from miles.utils.types import Sample

_TOKEN_RE = re.compile(r"\w+")
_EPS = 1e-4


def _embed_text(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    toks = _TOKEN_RE.findall(text.lower().strip())
    feats = [f"w:{t}" for t in toks] + [f"b:{toks[i]}_{toks[i+1]}" for i in range(len(toks) - 1)]
    s = " ".join(toks)[:2048]
    feats.extend(f"c:{s[i:i+3]}" for i in range(max(0, len(s) - 2)))
    for f in feats:
        h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
        vec[h % dim] += 1.0 if ((h >> 63) & 1) == 0 else -1.0
    n = float(np.linalg.norm(vec))
    return (vec / n) if n > 0 else np.eye(1, dim, dtype=np.float32)[0]


def prepare_records(args, records, strategy: str):
    """Attach question ID, optional prior probability, and (for DOTS) prompt embedding."""
    meta_key = f"{strategy}_curriculum"
    emb_dim = int(getattr(args, "curriculum_dots_embedding_dim", 256))
    for idx, rec in enumerate(records):
        meta = dict(rec.get(args.metadata_key) or {})
        qid = rec.get(args.curriculum_id_field, meta.get(args.curriculum_id_field))
        qid = f"row_{idx}" if qid is None else str(qid)
        raw_p = rec.get(args.curriculum_initial_probability_field, meta.get(args.curriculum_initial_probability_field))
        init_p = 0.5 if raw_p is None else float(raw_p) / args.curriculum_initial_probability_scale
        if not (math.isfinite(init_p) and 0 <= init_p <= 1):
            raise ValueError(f"Question {qid}: initial probability {init_p} out of [0, 1]")
        entry = {
            "question_id": qid,
            "initial_probability": init_p,
            "has_prior": raw_p is not None,
            "content_hash": hashlib.sha256(
                json.dumps([rec[args.input_key], rec.get(args.label_key)], sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest(),
        }
        if strategy == "dots":
            prompt = rec.get(args.input_key)
            text = "\n".join(str(x.get("content", x) if isinstance(x, dict) else x) for x in prompt) if isinstance(prompt, list) else str(prompt or "")
            entry["embedding"] = _embed_text(text, emb_dim).tolist()
        meta[meta_key] = entry
        yield {**rec, args.metadata_key: meta}


def _extract_episodes(group, reward_key: str | None) -> list[dict] | None:
    if not group:
        return None
    out = []
    for ep in group:
        frags = ep if isinstance(ep, list) else [ep]
        if not frags or any(s.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED) or s.metadata.get("grader_invalid") for s in frags):
            return None
        vals = [s.reward.get(reward_key) if isinstance(s.reward, dict) else s.reward for s in frags]
        if any(v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in vals):
            return None
        lps = [float(x) for s in frags for x in (getattr(s, "rollout_log_probs", None) or ()) if math.isfinite(float(x))]
        out.append({
            "reward": max(0.0, min(1.0, float(vals[0]))),
            "behavior_logprob": (sum(lps) / len(lps)) if lps else None,
            "response_len": sum(int(getattr(s, "response_length", 0) or 0) for s in frags),
        })
    return out


class _BaseCurriculum:
    METADATA_KEY: str = ""
    STRATEGY: str = ""

    def __init__(self, seed: int):
        self._lock = threading.RLock()
        self._rng = random.Random(seed)
        self.questions: dict[str, dict] = {}
        self._keys: tuple[str, ...] = ()
        self._reserved: set[str] = set()
        self._leases: dict[str, str] = {}
        self._lease_counts: Counter[str] = Counter()
        self._seen: set[str] = set()
        self._counts: Counter[str] = Counter()
        self.version = 0
        self.policy_version: int | None = None

    def reserve(self, key: str):
        with self._lock:
            self._reserved.add(key)

    def _begin_dispatch(self, prompt_group, policy_version, extra: dict | None = None):
        if policy_version is None:
            raise ValueError(f"{self.STRATEGY} dispatch requires a published policy version")
        key = prompt_group[0].metadata[self.METADATA_KEY]["question_id"]
        eid = uuid.uuid4().hex
        stamp = {"question_id": key, "group_id": eid, "policy_version": policy_version, **(extra or {})}
        for s in prompt_group:
            s.metadata = {**s.metadata, self.METADATA_KEY: stamp.copy()}
        self._reserved.add(key)
        self._leases[eid] = key
        self._lease_counts[key] += 1
        self.policy_version = policy_version
        self._counts["dispatched"] += 1
        return key, stamp

    def _check_group(self, prompt_group, group, reward_key):
        stamp = prompt_group[0].metadata[self.METADATA_KEY]
        frags = [s for ep in group for s in (ep if isinstance(ep, list) else [ep])]
        versions = {str(sp.version) for s in frags for sp in s.all_weight_version_spans}
        expected = stamp["policy_version"]
        if len(versions) > 1 or (versions and expected is not None and versions != {str(expected)}):
            self.failed(prompt_group)
            raise ValueError(f"{self.STRATEGY} group crossed policy versions: {versions}, expected {expected}")
        stats = _extract_episodes(group, reward_key) if (prompt_group and len(group) == len(prompt_group)) else None
        return stamp, stats

    def failed(self, prompt_group):
        with self._lock:
            eid = prompt_group[0].metadata[self.METADATA_KEY]["group_id"]
            if eid not in self._seen:
                self._seen.add(eid)
                self._counts["invalid"] += 1
            self.release(prompt_group)

    def release(self, prompt_group):
        with self._lock:
            eid = prompt_group[0].metadata[self.METADATA_KEY]["group_id"]
            if (key := self._leases.pop(eid, None)) is not None:
                self._lease_counts[key] -= 1
                if not self._lease_counts[key]:
                    self._reserved.discard(key)
                    del self._lease_counts[key]

    def _base_state(self) -> dict:
        return copy.deepcopy({
            "schema": 1, "strategy": self.STRATEGY, "config": asdict(self.config),
            "questions": self.questions, "version": self.version, "policy_version": self.policy_version,
            "seen": sorted(self._seen), "rng": self._rng.getstate(), "counts": dict(self._counts),
        })

    def _load_base_state(self, state: dict) -> None:
        if state.get("schema") != 1 or state.get("strategy") != self.STRATEGY or state.get("config") != asdict(self.config):
            raise ValueError(f"{self.STRATEGY} checkpoint config/schema mismatch")
        if {k: r["content_hash"] for k, r in state["questions"].items()} != {k: r["content_hash"] for k, r in self.questions.items()}:
            raise ValueError(f"{self.STRATEGY} checkpoint questions do not match dataset")
        self.questions = copy.deepcopy(state["questions"])
        self.version, self.policy_version = state["version"], state.get("policy_version")
        self._seen, self._counts = set(state["seen"]), Counter(state["counts"])
        self._rng.setstate(state["rng"])
        self._reserved.clear()
        self._leases.clear()
        self._lease_counts.clear()


@dataclass(frozen=True)
class DAREConfig:
    buffer_size: int = 64
    beta_alpha: float = 2.0
    snis_clip: float = 5.0
    drift_decay: float = 0.15
    exploration: float = 0.10
    easy_length_penalty: float = 0.0

    def __post_init__(self):
        if self.buffer_size < 1 or self.beta_alpha <= 0 or self.snis_clip <= 0 or self.drift_decay < 0 or self.easy_length_penalty < 0 or not 0 <= self.exploration <= 1:
            raise ValueError("Invalid DAREConfig parameters")

    @classmethod
    def from_args(cls, args) -> "DAREConfig":
        return cls(
            buffer_size=args.curriculum_dare_buffer_size,
            beta_alpha=args.curriculum_dare_beta_alpha,
            snis_clip=args.curriculum_dare_snis_clip,
            drift_decay=args.curriculum_dare_drift_decay,
            exploration=args.curriculum_exploration,
            easy_length_penalty=args.curriculum_dare_easy_length_penalty,
        )


class DARECurriculum(_BaseCurriculum):
    """DARE: Prompt-wise FIFO replay buffer + clipped SNIS difficulty + symmetric Beta selection."""

    METADATA_KEY = "dare_curriculum"
    STRATEGY = "dare"

    def __init__(self, questions, *, config: DAREConfig, seed: int):
        super().__init__(seed)
        self.config = config
        for row in questions:
            k = row["question_id"]
            if k in self.questions:
                raise ValueError(f"Duplicate DARE question ID: {k}")
            d0 = 1.0 - float(row.get("initial_probability", 0.5))
            self.questions[k] = {
                "prior_d": d0, "has_prior": bool(row.get("has_prior", True)),
                "difficulty": d0, "buffer": [], "content_hash": row.get("content_hash"),
            }
        if not self.questions:
            raise ValueError("DARE needs at least one question")
        self._keys = tuple(self.questions)

    def _estimate_snis(self, row: dict, cur_ver: int | None) -> float:
        buf = row["buffer"]
        pw = 1.0 if row["has_prior"] else 0.5
        if not buf:
            return row["prior_d"]
        lws, fails = [], []
        for e in buf:
            beh, cur = e.get("behavior_logprob"), e.get("current_logprob")
            ratio = (cur - beh) if (cur is not None and beh is not None) else 0.0
            lag = max(0, int(cur_ver) - int(e["policy_version"])) if (cur_ver is not None and e.get("policy_version") is not None) else 0
            lws.append(max(-self.config.snis_clip, min(self.config.snis_clip, ratio - self.config.drift_decay * lag)))
            fails.append(1.0 - float(e["reward"]))
        lw = np.asarray(lws, dtype=float)
        w = np.exp(lw - lw.max())
        d = (pw * row["prior_d"] + float(w @ np.asarray(fails, dtype=float))) / (pw + float(w.sum()))
        return max(0.0, min(1.0, float(d)))

    def update_policy_logprobs(self, qid: str, logprobs: list[float], policy_version: int | None = None) -> float:
        with self._lock:
            row = self.questions[qid]
            for e, lp in zip(row["buffer"], logprobs, strict=True):
                e["current_logprob"] = float(lp)
            row["difficulty"] = self._estimate_snis(row, self.policy_version if policy_version is None else policy_version)
            return row["difficulty"]

    def probabilities(self, keys=None):
        with self._lock:
            keys = list(keys) if keys is not None else [k for k in self._keys if k not in self._reserved]
            if not keys:
                return keys, np.array([], dtype=float)
            diffs = [self._estimate_snis(self.questions[k], self.policy_version) for k in keys]
            for k, d in zip(keys, diffs, strict=True):
                self.questions[k]["difficulty"] = d
            x = np.clip(np.asarray(diffs, dtype=float), _EPS, 1.0 - _EPS)
            lw = (self.config.beta_alpha - 1.0) * (np.log(x) + np.log(1.0 - x))
            w = np.exp(lw - lw.max())
            w /= w.sum()
            return keys, (1.0 - self.config.exploration) * w + self.config.exploration / len(keys)

    def select(self, count: int) -> list[str]:
        out = []
        with self._lock:
            for _ in range(count):
                keys, w = self.probabilities()
                if not keys:
                    break
                k = self._rng.choices(keys, weights=w, k=1)[0]
                out.append(k)
                self._reserved.add(k)
        return out

    def dispatch(self, prompt_group, policy_version):
        with self._lock:
            key = prompt_group[0].metadata[self.METADATA_KEY]["question_id"]
            d = self._estimate_snis(self.questions[key], policy_version)
            self.questions[key]["difficulty"] = d
            self._begin_dispatch(prompt_group, policy_version, {"difficulty_at_dispatch": d})

    def complete(self, prompt_group, group, *, reward_key=None) -> bool:
        stamp, stats = self._check_group(prompt_group, group, reward_key)
        with self._lock:
            eid = stamp["group_id"]
            if eid in self._seen:
                return False
            self._seen.add(eid)
            if stats is None:
                self._counts["invalid"] += 1
                return False
            row = self.questions[stamp["question_id"]]
            row["buffer"].extend({**e, "current_logprob": e["behavior_logprob"], "policy_version": stamp["policy_version"]} for e in stats)
            if len(row["buffer"]) > self.config.buffer_size:
                del row["buffer"][: len(row["buffer"]) - self.config.buffer_size]
            row["difficulty"] = self._estimate_snis(row, stamp["policy_version"])
            if self.config.easy_length_penalty > 0 and stamp["difficulty_at_dispatch"] <= 0.25:
                eps = [ep if isinstance(ep, list) else [ep] for ep in group]
                lens = [sum(int(getattr(s, "response_length", 0) or 0) for s in fs) for fs in eps]
                if (mx := max(lens, default=0)) > 0:
                    for fs, ln in zip(eps, lens, strict=True):
                        pen = self.config.easy_length_penalty * (ln / mx)
                        for s in fs:
                            if isinstance(s.reward, dict) and reward_key in s.reward and float(s.reward[reward_key]) > 0:
                                s.reward[reward_key] = max(0.0, float(s.reward[reward_key]) - pen)
                            elif isinstance(s.reward, (int, float)) and float(s.reward) > 0:
                                s.reward = max(0.0, float(s.reward) - pen)
            self.version += 1
            self._counts["complete"] += 1
            return True

    def metrics(self) -> dict[str, float]:
        with self._lock:
            diffs = [r["difficulty"] for r in self.questions.values()]
            return {
                "version": self.version,
                "difficulty/mean": float(np.mean(diffs)),
                "buffer/mean_entries": float(np.mean([len(r["buffer"]) for r in self.questions.values()])),
                **{f"groups/{k}_total": self._counts[k] for k in ("dispatched", "complete", "invalid")},
            }

    state_dict = _BaseCurriculum._base_state
    load_state_dict = _BaseCurriculum._load_base_state


@dataclass(frozen=True)
class DOTSConfig:
    num_references: int = 64
    embedding_dim: int = 256
    attention_temp: float = 0.15
    target_passrate: float = 0.5
    sigma: float = 0.20
    ema_decay: float = 0.5
    refresh_interval: int = 5
    exploration: float = 0.10
    replay_capacity: int = 256
    replay_ratio: float = 0.0
    max_replay_age: int = 2

    def __post_init__(self):
        if self.num_references < 1 or self.embedding_dim < 4 or self.attention_temp <= 0 or self.sigma <= 0 or self.refresh_interval < 1:
            raise ValueError("Invalid DOTSConfig parameters")
        if not (0 <= self.target_passrate <= 1 and 0 < self.ema_decay <= 1 and 0 <= self.exploration <= 1 and 0 <= self.replay_ratio < 1):
            raise ValueError("Invalid DOTSConfig range parameters")

    @classmethod
    def from_args(cls, args) -> "DOTSConfig":
        return cls(
            num_references=args.curriculum_dots_num_references,
            embedding_dim=args.curriculum_dots_embedding_dim,
            attention_temp=args.curriculum_dots_attention_temp,
            target_passrate=args.curriculum_dots_target_passrate,
            sigma=args.curriculum_dots_sigma,
            ema_decay=args.curriculum_dots_ema_decay,
            refresh_interval=args.curriculum_dots_refresh_interval,
            exploration=args.curriculum_exploration,
            replay_capacity=args.curriculum_dots_replay_capacity,
            replay_ratio=args.curriculum_dots_replay_ratio,
            max_replay_age=args.curriculum_dots_max_replay_age,
        )


class DOTSCurriculum(_BaseCurriculum):
    """DOTS: Reference prompts + Attention difficulty prediction + FIFO Rollout Replay."""

    METADATA_KEY = "dots_curriculum"
    STRATEGY = "dots"

    def __init__(self, questions, *, config: DOTSConfig, seed: int):
        super().__init__(seed)
        self.config = config
        embs = []
        for idx, row in enumerate(questions):
            k = row["question_id"]
            if k in self.questions:
                raise ValueError(f"Duplicate DOTS question ID: {k}")
            emb = np.asarray(row["embedding"], dtype=np.float32) if row.get("embedding") is not None else _embed_text(str(k), config.embedding_dim)
            n = float(np.linalg.norm(emb))
            embs.append((emb / n) if n > 0 else emb)
            p0 = float(row.get("initial_probability", 0.5))
            self.questions[k] = {"idx": idx, "initial_p": p0, "predicted_p": p0, "observed_p": None, "last_ver": None, "content_hash": row.get("content_hash")}
        if not self.questions:
            raise ValueError("DOTS needs at least one question")
        self._keys = tuple(self.questions)
        self._embs = np.stack(embs, axis=0)
        ordered = sorted(self._keys, key=lambda k: (self.questions[k]["initial_p"], k))
        n_ref = min(config.num_references, len(ordered))
        self._refs = [ordered[i] for i in dict.fromkeys(int(x) for x in np.linspace(0, len(ordered) - 1, n_ref).round().astype(int))]
        self._ref_set = set(self._refs)
        self._attn = self._compute_attn()
        self._recompute_preds()
        self._replay: deque[tuple[int, list]] = deque(maxlen=max(1, config.replay_capacity))

    def _compute_attn(self) -> np.ndarray:
        ref_e = self._embs[[self.questions[k]["idx"] for k in self._refs]]
        sim = (self._embs @ ref_e.T).astype(np.float64) / self.config.attention_temp
        w = np.exp(sim - sim.max(axis=1, keepdims=True))
        return w / w.sum(axis=1, keepdims=True)

    def _is_fresh(self, r: dict) -> bool:
        return r["last_ver"] is not None and (self.policy_version is None or int(self.policy_version) - int(r["last_ver"]) < self.config.refresh_interval)

    def _recompute_preds(self) -> None:
        ref_p = np.array([self.questions[k]["observed_p"] if self.questions[k]["observed_p"] is not None else self.questions[k]["initial_p"] for k in self._refs], dtype=np.float64)
        preds = np.clip(self._attn @ ref_p, 0.0, 1.0)
        for i, k in enumerate(self._keys):
            r = self.questions[k]
            r["predicted_p"] = float(r["observed_p"]) if (k in self._ref_set and r["observed_p"] is not None) else (0.5 * float(r["observed_p"]) + 0.5 * float(preds[i]) if (r["observed_p"] is not None and self._is_fresh(r)) else float(preds[i]))

    def probabilities(self, keys=None):
        with self._lock:
            keys = list(keys) if keys is not None else [k for k in self._keys if k not in self._reserved]
            if not keys:
                return keys, np.array([], dtype=float)
            preds = np.array([self.questions[k]["predicted_p"] for k in keys], dtype=np.float64)
            z = -0.5 * ((preds - self.config.target_passrate) / self.config.sigma) ** 2
            w = np.exp(z - z.max())
            w /= w.sum()
            return keys, (1.0 - self.config.exploration) * w + self.config.exploration / len(keys)

    def select(self, count: int) -> list[str]:
        out: list[str] = []
        with self._lock:
            stale = sorted((k for k in self._refs if k not in self._reserved and not self._is_fresh(self.questions[k])), key=lambda k: -1 if self.questions[k]["last_ver"] is None else self.questions[k]["last_ver"])
            for k in stale[: min(max(1, count // 4) if count > 0 else 0, count)]:
                out.append(k)
                self._reserved.add(k)
            for _ in range(count - len(out)):
                keys, w = self.probabilities()
                if not keys:
                    break
                k = self._rng.choices(keys, weights=w, k=1)[0]
                out.append(k)
                self._reserved.add(k)
        return out

    def dispatch(self, prompt_group, policy_version):
        with self._lock:
            self._begin_dispatch(prompt_group, policy_version)

    def complete(self, prompt_group, group, *, reward_key=None) -> bool:
        stamp, stats = self._check_group(prompt_group, group, reward_key)
        with self._lock:
            eid = stamp["group_id"]
            if eid in self._seen:
                return False
            self._seen.add(eid)
            if stats is None:
                self._counts["invalid"] += 1
                return False
            obs_p = sum(e["reward"] for e in stats) / len(stats)
            r = self.questions[stamp["question_id"]]
            r["observed_p"] = obs_p if r["observed_p"] is None else (1.0 - self.config.ema_decay) * r["observed_p"] + self.config.ema_decay * obs_p
            r["last_ver"] = stamp["policy_version"]
            self._recompute_preds()
            if self.config.replay_capacity > 0 and self.config.replay_ratio > 0 and 0.0 < obs_p < 1.0:
                self._replay.append((int(stamp["policy_version"]), copy.deepcopy(group)))
            self.version += 1
            self._counts["complete"] += 1
            return True

    def target_replay_count(self, batch_size: int) -> int:
        return 0 if (self.config.replay_ratio <= 0 or self.config.replay_capacity <= 0) else min(batch_size - 1, int(batch_size * self.config.replay_ratio))

    def pop_replay_groups(self, max_count: int, current_policy_version: int | None) -> list[list]:
        out = []
        with self._lock:
            while self._replay and len(out) < max_count:
                ver, grp = self._replay.popleft()
                if current_policy_version is not None and int(current_policy_version) - ver > self.config.max_replay_age:
                    continue
                out.append(copy.deepcopy(grp))
                self._counts["replay_served"] += 1
        return out

    def metrics(self) -> dict[str, float]:
        with self._lock:
            return {
                "version": self.version,
                "predicted_p/mean": float(np.mean([r["predicted_p"] for r in self.questions.values()])),
                "replay/buffer_size": len(self._replay),
                "replay/served_total": self._counts["replay_served"],
                **{f"groups/{k}_total": self._counts[k] for k in ("dispatched", "complete", "invalid")},
            }

    def state_dict(self) -> dict:
        with self._lock:
            return {**self._base_state(), "refs": list(self._refs)}

    def load_state_dict(self, state: dict) -> None:
        with self._lock:
            self._load_base_state(state)
            self._refs = list(state["refs"])
            self._ref_set = set(self._refs)
            self._attn = self._compute_attn()
            self._recompute_preds()
            self._replay.clear()
