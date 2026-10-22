"""Durable local batches, in the existing rollout dump format."""

import hashlib
import json
import logging
import math
import os
import tempfile
from pathlib import Path

import torch

from miles.ray.rollout.debug_data import _sample_to_debug_dict
from miles.ray.rollout.rollout_data_conversion import _compute_dynamic_global_batch_size
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def tokenizer_identity(checkpoint):
    """Hash tokenizer assets rather than the changing actor checkpoint weights."""
    root = Path(checkpoint)
    names = (
        "tokenizer.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    )
    assets = {}
    for name in names:
        path = root / name
        if path.is_file():
            assets[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not assets:
        # Hub revisions and remote tokenizers cannot be verified by local content.
        # Require exactly the same identifier in that case.
        assets["checkpoint"] = str(checkpoint)
    return hashlib.sha256(json.dumps(assets, sort_keys=True).encode()).hexdigest()


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


class LocalRolloutData:
    def __init__(self, args):
        self.args = args
        self.save_path = getattr(args, "save_rollout_data", None)
        self.load_path = getattr(args, "load_rollout_data", None)
        self.tokenizer = tokenizer_identity(args.hf_checkpoint) if self.save_path or self.load_path else None

    def save(self, samples, metadata, *, rollout_id, trainer_model_id=None):
        if self.save_path is None:
            return
        stem = rollout_id if trainer_model_id is None else f"{trainer_model_id}_{rollout_id}"
        path = Path(self.save_path.format(rollout_id=stem))
        payload = {
            "rollout_id": rollout_id,
            "metadata": metadata,
            "recording": {
                "version": 1,
                "tokenizer": self.tokenizer,
                "n_samples_per_prompt": self.args.n_samples_per_prompt,
                "trainer_model_id": trainer_model_id,
            },
            "samples": [_sample_to_debug_dict(self.args, sample) for sample in samples],
        }
        atomic_save(payload, path)
        logger.info("Saved %d rollout samples before training to %s", len(samples), path)

    def load(self, rollout_id, *, train_parallel_config):
        path = Path(self.load_path.format(rollout_id=rollout_id))
        if not path.is_file():
            raise FileNotFoundError(f"Recorded rollout batch is missing: {path}; offline replay never regenerates it")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        recording = payload.get("recording", {})
        if recording.get("version") != 1:
            raise ValueError(
                f"{path} is not a versioned local rollout recording; use --load-debug-rollout-data for legacy debug files"
            )
        if recording["tokenizer"] != self.tokenizer:
            raise ValueError(f"Tokenizer differs from the rollout recording: {path}")
        if recording["n_samples_per_prompt"] != self.args.n_samples_per_prompt:
            raise ValueError(f"n_samples_per_prompt differs from the rollout recording: {path}")
        if recording.get("trainer_model_id") is not None:
            raise ValueError("Offline rollout loading currently supports single-policy recordings only")
        if payload["rollout_id"] != rollout_id:
            raise ValueError(f"Recorded rollout_id does not match requested step {rollout_id}: {path}")
        samples = [Sample.from_dict(data) for data in payload["samples"]]
        if not samples:
            raise ValueError(f"Empty rollout recording: {path}")
        for sample in samples:
            self._validate_sample(sample, path)
        metadata = dict(payload.get("metadata", {}))
        # Stored rollout content wins; batch scheduling belongs to the new trainer.
        metadata.pop("dynamic_global_batch_size", None)
        if self.args.use_dynamic_global_batch_size:
            metadata["dynamic_global_batch_size"] = _compute_dynamic_global_batch_size(
                self.args, train_parallel_config=train_parallel_config, num_samples=len(samples)
            )
        elif not any(sample.rollout_id is not None for sample in samples):
            if len(samples) % self.args.global_batch_size:
                raise ValueError(
                    f"Recorded batch size {len(samples)} is not divisible by global_batch_size={self.args.global_batch_size}: {path}"
                )
        logger.info("Reusing %d recorded rollout samples from %s; no generation or reward calls", len(samples), path)
        return samples, metadata

    @staticmethod
    def _validate_sample(sample, path):
        size = sample.response_length
        probs = sample.rollout_log_probs
        if not 0 < size <= len(sample.tokens) or sample.reward is None:
            raise ValueError(f"Incomplete rollout sample {sample.index} in {path}")
        if sample.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED):
            raise ValueError(f"Unsuccessful rollout sample {sample.index} in {path}")
        if (
            probs is None
            or len(probs) != size
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in probs)
        ):
            raise ValueError(f"Missing or invalid behavior logprobs for sample {sample.index} in {path}")
        if sample.loss_mask is not None and len(sample.loss_mask) != size:
            raise ValueError(f"Invalid loss mask for sample {sample.index} in {path}")
