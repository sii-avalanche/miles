"""Lossless durable snapshots of AvaCore message DAGs and legacy rollout records."""

from dataclasses import dataclass, replace
from typing import Any

from ava_core.core import ChatMessage, Trace, Tool
from ava_core.core.segment import ChatSegment
from ava_core.generate.core import Sample as Row
from ava_core.rewards.core import Reward
from ava_core.store.core import STORED


def graph_nodes(traces: tuple[Trace, ...]) -> list[Trace]:
    """Topological order, retaining identity across branches and source edges."""
    nodes, seen = [], set()
    pending = [(trace, False) for trace in reversed(traces)]
    while pending:
        node, ready = pending.pop()
        if node in seen:
            continue
        if ready:
            seen.add(node)
            nodes.append(node)
        else:
            pending.append((node, True))
            pending.extend((source, False) for source in reversed(node.sources))
            if node.prev is not None:
                pending.append((node.prev, False))
    return nodes


def conversation_tips(traces: tuple[Trace, ...]) -> list[Trace]:
    nodes = graph_nodes(traces)
    parents = {node.prev for node in nodes}
    # Match the store's ownership order: listed traces first, source-only tips next.
    return list(dict.fromkeys((*traces, *(node for node in nodes if node not in parents and node not in traces))))


def snapshot_traces(traces: tuple[Trace, ...]) -> dict[str, Any]:
    nodes = graph_nodes(traces)
    ids = {node: index for index, node in enumerate(nodes)}
    return {
        "version": 1,
        "tips": [ids[node] for node in traces],
        "nodes": [
            {
                "message": STORED.unstructure(node.message),
                "prev": ids.get(node.prev),
                "sources": [ids[source] for source in node.sources],
                "generated": node.generated,
                "tools": STORED.unstructure(node.tools) if node.prev is None else [],
                "metadata": {key: value for key, value in node.metadata.items() if key != "agent_patch_bytes"},
                "segments": [
                    {
                        "tokens": list(segment.tokens),
                        "log_probs": segment.log_probs,
                        "template": segment.template,
                        "is_generated": segment.is_generated,
                        "metadata": segment.metadata,
                    }
                    for segment in node.segments
                ],
            }
            for node in nodes
        ],
    }


@dataclass(frozen=True)
class _StoredTokenizer:
    """Snapshots already contain exact rendered text; no model/tokenizer load on writes."""

    template: str

    def decode(self, _tokens, **_kwargs):
        return self.template


def _segment(data: dict[str, Any]) -> ChatSegment:
    tokens, probs = data["tokens"], data["log_probs"]
    if len(tokens) != len(probs):
        raise ValueError("Stored token and log-probability lengths differ")
    if any(p is None for p in probs) and any(p is not None for p in probs):
        raise ValueError("A segment mixes prompt and generated log-probabilities")
    return ChatSegment(
        tokens=tuple(tokens),
        _log_probs=tuple(probs) if probs and probs[0] is not None else None,
        is_generated=data["is_generated"],
        tokenizer=_StoredTokenizer(data["template"]),
        metadata=data.get("metadata", {}),
    )


def restore_traces(data: dict[str, Any]) -> tuple[Trace, ...]:
    if data["version"] != 1:
        raise ValueError(f"Unsupported Trace snapshot version: {data['version']}")
    nodes = []
    for item in data["nodes"]:
        nodes.append(
            Trace(
                message=STORED.structure(item["message"], ChatMessage),
                prev=nodes[item["prev"]] if item["prev"] is not None else None,
                sources=tuple(nodes[index] for index in item["sources"]),
                generated=item["generated"],
                segments=tuple(_segment(segment) for segment in item["segments"]),
                metadata=item["metadata"],
                _tools=tuple(STORED.structure(item["tools"], list[Tool])),
            )
        )
    return tuple(nodes[index] for index in data["tips"])


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
