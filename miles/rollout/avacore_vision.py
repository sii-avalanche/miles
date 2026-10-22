"""Reconstruct image tensors from AvaCore's immutable observation segments."""

from functools import lru_cache

import torch
from ava_core.core.segment import ChatSegment
from ava_core.core.vision import process_images
from transformers import AutoProcessor


@lru_cache(maxsize=4)
def _processor(path: str):
    return AutoProcessor.from_pretrained(path)


def training_images(segments: list[ChatSegment]) -> dict[str, torch.Tensor] | None:
    blocks = []
    for segment in segments:
        if (vision := segment.metadata.get("vision")) is None:
            continue
        if segment.is_generated:
            raise ValueError("Image inputs cannot belong to generated segments")
        result = process_images(_processor(vision["processor"]), vision["text"], vision["images"])
        if tuple(result["input_ids"][0]) != segment.tokens:
            raise ValueError("Training processor changed the rollout image tokens")
        block = {
            key: value
            for key, value in result.items()
            if key not in {"input_ids", "attention_mask", "mm_token_type_ids"}
        }
        if not block or any(not isinstance(value, torch.Tensor) for value in block.values()):
            raise ValueError("Vision processor must return tensor training inputs")
        blocks.append(block)
    if not blocks:
        return None
    if any(block.keys() != blocks[0].keys() for block in blocks):
        raise ValueError("Inconsistent vision processor fields across turns")
    return {key: torch.cat([block[key] for block in blocks], dim=0) for key in blocks[0]}
