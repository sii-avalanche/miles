"""Pure timeline, fleet, engine-rate and downsampling transforms.

Inputs are observations/frames, independent of JSONL, Postgres and HTTP.
"""

import json
from dataclasses import replace

import numpy as np
import polars as pl

from miles.monitoring.collection.records import INITIALIZE_PHASE, Role, TrajectoryEvent, TrajectoryEventKind

HEATMAP_MAGNITUDE_FIELDS = ("util", "mem_mb", "power_w")


def trajectory_lanes(events, *, sample_indices=None):
    """Per-sample lifecycle assembly for the batch-anatomy swimlanes:
    gen/tool segments paired by (kind, turn), attempt windows, and the
    version span observed across segments. A start without an end (still
    running, or clipped by the window) yields t1=None and vice versa."""
    grouped: dict[int, list[TrajectoryEvent]] = {}
    for event in events:
        if sample_indices is not None and event.sample_index not in sample_indices:
            continue
        grouped.setdefault(event.sample_index, []).append(event)

    span_kinds = {
        TrajectoryEventKind.GEN_START: ("gen", True),
        TrajectoryEventKind.GEN_END: ("gen", False),
        TrajectoryEventKind.TOOL_START: ("tool", True),
        TrajectoryEventKind.TOOL_END: ("tool", False),
    }
    lanes = []
    for index, events in grouped.items():
        events.sort(key=lambda e: e.ts)
        segments: list[dict] = []
        open_spans: dict[tuple[str, int], dict] = {}
        attempts: list[dict] = []
        attempt_t0: float | None = None
        status = ""
        versions = sorted({int(e.weight_version) for e in events if e.weight_version.isdigit()})
        for event in events:
            if event.kind == TrajectoryEventKind.ATTEMPT_START:
                attempt_t0 = event.ts
            elif event.kind == TrajectoryEventKind.ATTEMPT_END:
                attempts.append(dict(t0=attempt_t0, t1=event.ts))
                attempt_t0 = None
                status = event.detail or status
                # an attempt ending closes whatever it left open: the
                # single-turn path emits no gen_end (generation ends WITH
                # the attempt), and an abort mid-turn stops right here —
                # dangling spans must not render to the consume line
                for segment in open_spans.values():
                    segment["t1"] = event.ts
                    # coarse gen spans open before the first turn finished,
                    # so their start event predates any weight_version; the
                    # attempt_end event carries the version that generated them
                    if not segment["weight_version"]:
                        segment["weight_version"] = event.weight_version
                open_spans.clear()
            else:
                base, is_start = span_kinds[TrajectoryEventKind(event.kind)]
                if is_start:
                    segment = dict(
                        kind=base, t0=event.ts, t1=None, turn=event.turn, weight_version=event.weight_version
                    )
                    open_spans[(base, event.turn)] = segment
                    segments.append(segment)
                else:
                    segment = open_spans.pop((base, event.turn), None)
                    if segment is None:
                        segments.append(
                            dict(
                                kind=base,
                                t0=None,
                                t1=event.ts,
                                turn=event.turn,
                                weight_version=event.weight_version,
                            )
                        )
                    else:
                        segment["t1"] = event.ts
                        segment["weight_version"] = event.weight_version or segment["weight_version"]
        if attempt_t0 is not None:
            attempts.append(dict(t0=attempt_t0, t1=None))  # attempt still running
        if any(segment["turn"] > 0 for segment in segments if segment["kind"] == "gen"):
            # per-turn spans (multi_turn / agentic) supersede the coarse
            # attempt-level gen span the generate_and_rm probe emits
            segments = [s for s in segments if not (s["kind"] == "gen" and s["turn"] < 0)]
        lanes.append(
            dict(
                sample_index=index,
                group_index=events[0].group_index,
                first_ts=events[0].ts,
                last_ts=events[-1].ts,
                segments=segments,
                attempts=attempts,
                status=status,
                versions=versions,
            )
        )
    lanes.sort(key=lambda lane: lane["first_ts"])
    return lanes


def phases_by_lane(events, *, edge, windows, known_lanes, meta, t0=None, t1=None, lanes=None):
    """Phase intervals resolved onto (node, gpu) lanes.

    Train-side events carry their own GPUs. ``rollout_manager`` events have
    none — the manager is a GPU-less driver process — so they are expanded
    onto every GPU of every rollout engine, clipped at topology-window
    boundaries (an engine restart mid-rollout splits the painted interval).
    """

    def overlaps(a0: float, a1: float) -> bool:
        return (t1 is None or a0 < t1) and (t0 is None or a1 > t0)

    # resolve OPEN intervals (Timer.start markers): a closing event with
    # the real t1 supersedes its open twin; a still-open one is clipped to
    # the newest data timestamp so it renders as a growing in-progress band
    closed = {(e.node, e.rank, e.name, e.t0) for e in events if not e.open}
    events = [
        replace(e, t1=max(edge, e.t0)) if e.open else e
        for e in events
        if not (e.open and (e.node, e.rank, e.name, e.t0) in closed)
    ]

    out = []
    for event in events:
        if not overlaps(event.t0, event.t1):
            continue
        if event.role == Role.ROLLOUT_MANAGER:
            for window in windows:
                clip0 = max(event.t0, window["t0"])
                if event.t1 < 0:
                    # still-open interval: extends to the window edge (the
                    # renderer draws open t1=-1 through to "now")
                    clip1 = -1.0 if window["t1"] is None else window["t1"]
                else:
                    clip1 = event.t1 if window["t1"] is None else min(event.t1, window["t1"])
                if clip1 >= 0 and clip0 >= clip1:
                    continue
                covered = {(node, gpu) for engine in window["engines"] for node, gpu in engine["gpus"]}
                if not covered:
                    # External engines register without GPU identity; fall
                    # back to every known lane on the engines' nodes.
                    engine_nodes = {engine["addr"].split("//")[-1].split(":")[0] for engine in window["engines"]}
                    covered = {(lane["node"], lane["gpu"]) for lane in known_lanes if lane["node"] in engine_nodes}
                if lanes is not None:
                    covered &= lanes
                for node, gpu in sorted(covered):
                    out.append(
                        dict(
                            name=event.name,
                            t0=clip0,
                            t1=clip1,
                            node=node,
                            gpu=gpu,
                            rank=event.rank,
                            role=event.role,
                        )
                    )
        else:
            for gpu in event.gpus:
                if lanes is not None and (event.node, gpu) not in lanes:
                    continue
                out.append(
                    dict(
                        name=event.name,
                        t0=event.t0,
                        t1=event.t1,
                        node=event.node,
                        gpu=gpu,
                        rank=event.rank,
                        role=event.role,
                    )
                )
    # synthesize the initialize band: from collector start (meta.start_ts)
    # to each lane's first observed event — model loading / engine startup
    # has GPU util but no Timer instrumentation. Only when the window
    # covers the run start: a later window's first event is not the
    # lane's first event ever
    if meta is not None and (t0 is None or t0 <= meta.start_ts):
        first_event: dict[tuple[str, int], float] = {}
        for event in out:
            key = (event["node"], event["gpu"])
            first_event[key] = min(first_event.get(key, float("inf")), event["t0"])
        for (node, gpu), first_t0 in first_event.items():
            if lanes is not None and (node, gpu) not in lanes:
                continue
            if first_t0 > meta.start_ts and overlaps(meta.start_ts, first_t0):
                out.append(
                    dict(
                        name=INITIALIZE_PHASE,
                        t0=meta.start_ts,
                        t1=first_t0,
                        node=node,
                        gpu=gpu,
                        rank=-1,
                        role=Role.DERIVED,
                    )
                )
    out.sort(key=lambda e: (e["node"], e["gpu"], e["t0"]))
    return out


def fold_dp_ranks(frame: pl.DataFrame, metric: str, mean_metrics) -> pl.DataFrame:
    """Engine-level view: collapse per-dp-rank samples — including
    same-tick duplicates from dumps written before ``dp_rank`` was kept —
    into one value per (addr, ts). Ranks share a scrape ts (one fetch per
    engine yields every rank's sample), so grouping on ts is exact."""
    if frame.is_empty():
        return frame
    # also drop nulls: chunk-level struct unification pads absent labels
    # with null, which would otherwise split one logical series
    stripped = {
        labels_json: json.dumps(
            {k: v for k, v in json.loads(labels_json).items() if k != "dp_rank" and v is not None},
            sort_keys=True,
        )
        for labels_json in frame["labels_json"].unique()
    }
    frame = frame.with_columns(pl.col("labels_json").replace_strict(stripped))
    agg = pl.col("value").mean() if metric in mean_metrics else pl.col("value").sum()
    return frame.group_by(["ts", "addr", "metric", "labels_json"]).agg(agg.alias("value"))


def derived_series(num_parts, den_parts, max_points: int) -> list[dict]:
    """Per-interval derivative: rate when ``den_parts`` is None (dt as the
    denominator), else delta(num)/delta(den) joined on scrape timestamps."""
    out = []
    for (addr, labels_json), num in num_parts.items():
        ts = num["ts"].to_numpy()
        values = num["value"].to_numpy()
        if den_parts is None:
            den_ts, den_values = ts, None
        else:
            den = den_parts.get((addr, labels_json))
            if den is None:
                continue
            den_ts, den_values = den["ts"].to_numpy(), den["value"].to_numpy()
            shared, num_idx, den_idx = np.intersect1d(ts, den_ts, return_indices=True)
            ts, values, den_values = shared, values[num_idx], den_values[den_idx]
        if len(ts) < 2:
            continue
        d_num = np.diff(values)
        d_den = np.diff(ts) if den_values is None else np.diff(den_values)
        keep = (d_num >= 0) & (d_den > 0)
        point_ts, point_values = ts[1:][keep], d_num[keep] / d_den[keep]
        if not len(point_ts):
            continue
        point_ts, point_values = stride_downsample(point_ts, point_values, max_points)
        out.append(
            dict(addr=addr, labels=_engine_labels(labels_json), ts=point_ts.tolist(), value=point_values.tolist())
        )
    return out


def fleet_summary(phase, frame, *, x_buckets):
    """Scale-invariant fleet overview: per-bucket phase composition
    (fraction of lanes in each phase) and the util distribution band
    across all lanes. Payload size is O(x_buckets), never O(lanes)."""
    palette = phase["palette"] or []
    n_rows = len(phase["rows"])
    composition = {}
    if n_rows and palette:
        matrix = np.frombuffer(phase["values"], dtype=np.uint8).reshape(n_rows, x_buckets)
        for pid, name in enumerate(palette):
            fractions = (matrix == pid).sum(axis=0) / n_rows
            composition[name or "none"] = [round(float(v), 4) for v in fractions]

    band = dict(p10=[None] * x_buckets, p50=[None] * x_buckets, p90=[None] * x_buckets, min=[None] * x_buckets)
    rt0, rt1 = phase["t0"], phase["t1"]
    if len(frame) and rt1 > rt0:
        # exact percentiles from the raw stream: carpet cells cannot
        # distinguish "no sample" from 0% util
        span = rt1 - rt0
        per_bucket = (
            frame.with_columns(
                bucket=((pl.col("ts") - rt0) / span * x_buckets).floor().clip(0, x_buckets - 1).cast(pl.Int32)
            )
            .group_by("bucket", "node", "gpu")
            .agg(pl.col("util").max().alias("u"))
            .group_by("bucket")
            .agg(
                p10=pl.col("u").quantile(0.1),
                p50=pl.col("u").quantile(0.5),
                p90=pl.col("u").quantile(0.9),
                min=pl.col("u").min(),
            )
        )
        for row in per_bucket.iter_rows(named=True):
            for key in ("p10", "p50", "p90", "min"):
                band[key][row["bucket"]] = float(row[key])

    return dict(t0=rt0, t1=rt1, x_buckets=x_buckets, lanes=n_rows, palette=palette, composition=composition, band=band)


def heatmap(metric, *, window, lane_index, frame, events, t0=None, t1=None, x_buckets=1200, lanes=None):
    """Rank-carpet matrix: one uint8 cell per (lane, time bucket).

    Magnitude metrics take the bucket MAX (activity must not average away);
    ``phase`` paints the covering phase id with non-idle winning over idle.
    Returns rows metadata + the raw bytes; the server frames it as binary.
    """
    assert x_buckets >= 2, f"{x_buckets=}"
    if window is None:
        return dict(rows=[], x_buckets=x_buckets, t0=0.0, t1=0.0, metric=metric, values=b"", scale=None, palette=None)
    t0 = window[0] if t0 is None else t0
    t1 = window[1] if t1 is None else t1
    assert t1 > t0, f"empty heatmap window [{t0}, {t1})"
    rows = [
        dict(node=entry["node"], gpu=entry["gpu"], index=entry["index"], roles=entry["roles"])
        for entry in lane_index
        if lanes is None or (entry["node"], entry["gpu"]) in lanes
    ]
    row_of = {(lane["node"], lane["gpu"]): i for i, lane in enumerate(rows)}
    matrix = np.zeros((len(rows), x_buckets), dtype=np.uint8)
    span = t1 - t0

    def bucket(ts: float) -> int:
        return min(x_buckets - 1, max(0, int((ts - t0) / span * x_buckets)))

    if metric in HEATMAP_MAGNITUDE_FIELDS:
        values = np.zeros((len(rows), x_buckets))
        filled = np.zeros((len(rows), x_buckets), dtype=bool)
        for (node, gpu), part in frame.partition_by(["node", "gpu"], as_dict=True).items():
            row = row_of.get((node, gpu))
            if row is None:
                continue
            buckets = np.clip(((part["ts"].to_numpy() - t0) / span * x_buckets).astype(np.int64), 0, x_buckets - 1)
            np.maximum.at(values[row], buckets, part[metric].to_numpy().astype(float))
            filled[row, buckets] = True
        peak = float(values[filled].max()) if filled.any() else 1.0
        scale = dict(max=100.0 if metric == "util" else peak)
        cell = np.minimum(255, (values[filled] / scale["max"] * 255).astype(np.int64))
        matrix[filled] = cell.astype(np.uint8)
        return dict(
            rows=rows,
            x_buckets=x_buckets,
            t0=t0,
            t1=t1,
            metric=metric,
            values=matrix.tobytes(),
            scale=scale,
            palette=None,
        )

    if metric != "phase":
        raise ValueError(f"unknown heatmap metric {metric!r}")
    idle = {INITIALIZE_PHASE, "train_wait", "sleep"}
    names = sorted({e["name"] for e in events})
    palette = [""] + names  # id 0 = no phase observed
    phase_id = {name: i + 1 for i, name in enumerate(names)}
    # idle painted first so any active phase in the same bucket wins
    for event in sorted(events, key=lambda e: e["name"] not in idle):
        row = row_of.get((event["node"], event["gpu"]))
        if row is None:
            continue
        b0 = bucket(max(event["t0"], t0))
        b1 = bucket(min(event["t1"], t1))
        matrix[row, b0 : b1 + 1] = phase_id[event["name"]]
    return dict(
        rows=rows,
        x_buckets=x_buckets,
        t0=t0,
        t1=t1,
        metric=metric,
        values=matrix.tobytes(),
        scale=None,
        palette=palette,
    )


def lifecycle_heatmap(
    all_lanes, t0, t1, x_buckets, *, max_rows=1024, palette=("", "queue", "generating", "tool_wait")
):
    """Run-wide trajectory carpet: y = samples ordered by first event,
    color = lifecycle state. Queue paints attempt windows first so gen and
    tool segments win the shared buckets. Rows cap at LIFECYCLE_MAX_ROWS
    (submit order, earliest kept); ``rows_total`` reports the uncapped
    count so the frontend can say "showing N/M"."""
    lanes = all_lanes[:max_rows]
    matrix = np.zeros((len(lanes), x_buckets), dtype=np.uint8)
    span = t1 - t0
    state_id = {name: i for i, name in enumerate(palette)}

    def bucket(ts: float) -> int:
        return min(x_buckets - 1, max(0, int((ts - t0) / span * x_buckets)))

    for row, lane in enumerate(lanes):
        for attempt in lane["attempts"]:
            matrix[row, bucket(attempt["t0"] or t0) : bucket(attempt["t1"] or t1) + 1] = state_id["queue"]
        for segment in lane["segments"]:
            cell = state_id["generating" if segment["kind"] == "gen" else "tool_wait"]
            matrix[row, bucket(segment["t0"] or t0) : bucket(segment["t1"] or t1) + 1] = cell
    rows = [dict(sample_index=lane["sample_index"], group_index=lane["group_index"]) for lane in lanes]
    return dict(
        rows=rows,
        rows_total=len(all_lanes),
        x_buckets=x_buckets,
        t0=t0,
        t1=t1,
        metric="lifecycle",
        values=matrix.tobytes(),
        scale=None,
        palette=list(palette),
    )


def _metric_by_suffix(metrics: dict[str, float], base_key: str) -> float | None:
    if (value := metrics.get(base_key)) is not None:
        return value
    for key, value in metrics.items():
        if key.endswith(f"/{base_key}"):
            return value
    return None


# ------------------------------ downsampling --------------------------------


def _engine_labels(labels_json: str) -> dict:
    return {k: v for k, v in json.loads(labels_json).items() if v is not None}


def stride_downsample(xs, ys, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = np.asarray(xs), np.asarray(ys)
    assert max_points >= 2, f"{max_points=}"
    if len(xs) <= max_points:
        return xs, ys
    idx = np.linspace(0, len(xs) - 1, max_points).astype(np.int64)
    return xs[idx], ys[idx]


def minmax_downsample(xs, ys, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Bucketed downsampling keeping each bucket's min and max, so spikes and
    dips survive. Output length is at most ``max_points``."""
    xs, ys = np.asarray(xs), np.asarray(ys)
    assert max_points >= 2, f"{max_points=}"
    if len(xs) <= max_points:
        return xs, ys
    num_buckets = max_points // 2
    edges = np.linspace(0, len(xs), num_buckets + 1).astype(np.int64)
    keep: set[int] = set()
    for b in range(num_buckets):
        lo, hi = int(edges[b]), int(edges[b + 1])
        if lo >= hi:
            continue
        segment = ys[lo:hi]
        keep.add(lo + int(np.argmin(segment)))
        keep.add(lo + int(np.argmax(segment)))
    idx = sorted(keep)
    return xs[idx], ys[idx]


def outliers(criterion, *, frame=None, events=(), top_k=16):
    """Candidate lanes for the quick-pick buttons; the user confirms the
    selection — the machine only proposes."""
    if criterion == "lowest_util":
        grouped = frame.group_by("node", "gpu").agg(pl.col("util").mean().alias("score"))
        scored = [(row["score"], row["node"], row["gpu"]) for row in grouped.iter_rows(named=True)]
        scored.sort()
    elif criterion.startswith("slowest_phase:"):
        phase_name = criterion.removeprefix("slowest_phase:")
        durations: dict[tuple[str, int], list[float]] = {}
        for event in events:
            if event["name"] != phase_name:
                continue
            durations.setdefault((event["node"], event["gpu"]), []).append(event["t1"] - event["t0"])
        scored = [(sum(v) / len(v), node, gpu) for (node, gpu), v in durations.items()]
        scored.sort(reverse=True)
    else:
        raise ValueError(f"unknown outlier criterion {criterion!r}")
    return [dict(node=node, gpu=gpu, score=score) for score, node, gpu in scored[:top_k]]
