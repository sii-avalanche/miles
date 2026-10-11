"""Small identities for group decisions, without retaining rollout tensors or text."""

from collections import defaultdict
from collections.abc import Iterable

from miles.utils.types import Sample


def group_identity(samples: list[Sample]) -> dict:
    first = samples[0]
    query_id = (first.metadata or {}).get("_index")
    return {
        "group_index": first.group_index,
        "query_id": str(query_id) if query_id is not None else None,
        # Sample stores a zero-based retry counter; the dashboard labels rounds from 1.
        "generation_attempt": first.generation_attempt + 1,
        "sample_indices": [sample.index for sample in samples],
        "prompt_preview": str(first.prompt)[:160],
    }


def batch_group_identities(samples: Iterable[Sample]) -> list[dict]:
    groups = defaultdict(list)
    for sample in samples:
        groups[sample.group_index].append(sample)
    return [group_identity(group) for group in groups.values()]
