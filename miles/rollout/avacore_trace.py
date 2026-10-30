"""Lossless durable snapshots of AvaCore message DAGs and legacy rollout records."""

from dataclasses import dataclass, replace
from typing import Any, cast

from ava_core.core import ChatMessage, Trace, Tool
from ava_core.core.segment import ChatSegment, Tokenizer
from ava_core.core.trace import (
    decode_traces as restore_traces,
    encode_traces as snapshot_traces,
    iter_trace_nodes,
)
from ava_core.generate.core import Sample as Row
from ava_core.rewards.core import Reward
from ava_core.store.core import STORED

__all__ = ["conversation_tips", "graph_nodes", "restore_traces", "snapshot_traces", "write_record"]


def graph_nodes(traces: tuple[Trace, ...]) -> list[Trace]:
    return list(iter_trace_nodes(traces))


@dataclass(frozen=True)
class _StoredTokenizer:
    """Keep legacy segment text available when dataclasses.replace clears caches."""

    template: str

    def decode(self, _tokens, **_kwargs):
        return self.template


def _segment(data: dict[str, Any]) -> ChatSegment:
    """Restore a segment from a legacy flat rollout without retokenizing it."""
    tokens, probs = data["tokens"], data["log_probs"]
    if len(tokens) != len(probs):
        raise ValueError("Stored token and log-probability lengths differ")
    if any(p is None for p in probs) and any(p is not None for p in probs):
        raise ValueError("A segment mixes prompt and generated log-probabilities")
    segment = ChatSegment(
        tokens=tuple(tokens),
        _log_probs=tuple(probs) if probs and probs[0] is not None else None,
        is_generated=data["is_generated"],
        tokenizer=cast(Tokenizer, _StoredTokenizer(data["template"])),
        metadata=data.get("metadata", {}),
    )
    object.__setattr__(segment, "template", data["template"])
    object.__setattr__(segment, "metadata", data.get("metadata", {}))
    return segment


def conversation_tips(traces: tuple[Trace, ...]) -> list[Trace]:
    nodes = graph_nodes(traces)
    parents = {node.prev for node in nodes}
    # Match the store's ownership order: listed traces first, source-only tips next.
    return list(dict.fromkeys((*traces, *(node for node in nodes if node not in parents and node not in traces))))


def _legacy_trace(data: dict[str, Any], tokenizer) -> Trace:
    messages = STORED.structure(data["messages"], list[ChatMessage])
    tokens, probs, mask = (data.get(key, []) for key in ("tokens", "log_probs", "loss_mask"))
    if not len(tokens) == len(probs) == len(mask):
        raise ValueError("Legacy token/log-probability/loss-mask lengths differ")
    # The old format flattens turns. Generated spans map in order to assistant completions.
    spans, start = [], 0
    for index in range(1, len(mask) + 1):
        if index == len(mask) or mask[index] != mask[start]:
            spans.append((start, index, bool(mask[start])))
            start = index
    generated = [span for span in spans if span[2]]
    completed = [i for i, message in enumerate(messages) if "finish_reason" in message.metadata]
    if tokens and len(generated) != len(completed):
        raise ValueError("Cannot align legacy generated spans with completion messages")
    attached, offset = {}, 0
    for message_index, (_, end, _) in zip(completed if tokens else [], generated, strict=True):
        parts = []
        for a, b, trained in spans:
            if offset <= a < end:
                text = tokenizer.decode(tokens[a:b], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                parts.append(
                    _segment(dict(tokens=tokens[a:b], log_probs=probs[a:b], template=text, is_generated=trained))
                )
        attached[message_index] = tuple(parts)
        offset = end
    if offset != len(tokens):
        raise ValueError("Legacy trace has unaligned trailing tokens")
    trace = None
    for index, message in enumerate(messages):
        trace = Trace(
            message=message,
            prev=trace,
            generated=index in completed,
            segments=attached.get(index, ()),
            _tools=tuple(STORED.structure(data.get("tools", []), list[Tool])) if trace is None else (),
        )
    if trace is None:
        raise ValueError("Legacy trace has no messages")
    trace = replace(trace, metadata=data.get("metadata", {}))
    if tokens and trace.template != data["template"]:
        raise ValueError("Legacy tokenizer does not reproduce the stored template")
    return trace


async def write_record(run, record: dict[str, Any], *, legacy_tokenizer=None) -> None:
    if "trace_graph" in record:
        traces = restore_traces(record["trace_graph"])
    else:
        if legacy_tokenizer is None:
            raise ValueError("Legacy snapshots require the original tokenizer for lossless replay")
        traces = tuple(
            _legacy_trace(data, legacy_tokenizer) for data in (record["trace"], *record.get("subtraces", []))
        )
    metadata = dict(record.get("metadata", {}))
    # PostgreSQL indexes weight_version as int32; raw-HF servers can return "default".
    version = metadata.get("weight_version")
    try:
        numeric = int(version) if version is not None else None
        if numeric is not None and (not -(2**31) <= numeric < 2**31 or isinstance(version, (bool, float))):
            raise ValueError("not an int32 version")
    except (ValueError, TypeError):
        metadata["source_weight_version"] = version
        metadata["weight_version"] = None
    await run.create_rollout(
        query_id=record["query_id"],
        trial_id=record["trial_id"],
        instance=Row(record["instance"], key=lambda _: record["query_id"]),
        traces=traces,
        reward=STORED.structure(record["reward"], Reward) if record.get("reward") is not None else None,
        status=record["status"],
        metadata=metadata,
    )
