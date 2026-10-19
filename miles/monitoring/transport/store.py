"""File-backed store for miles dashboard time series.

The live collector buffers records via ``append()`` and persists them with
``flush()`` as append-only JSONL streams under ``{dump_details}/dashboard/``.
The dashboard server reads the same directory with ``MetricStore.load()`` and
picks up appended lines incrementally with ``follow()`` — a plain byte-offset
tail, correct because every stream is append-only.

Each stream holds one record type (one JSON object per line); see the
``Record`` subclasses for the schemas.

The two high-rate streams (``gpu_util``, ``engine_series``) are held in memory
as columnar polars frames instead of dataclass lists (design doc §17):
~16 B/row instead of ~600 B, vectorized parsing and numpy queries.

High-rate streams (those two plus ``phases``) write hourly partition files
``{stream}/{YYYYMMDD_HH}.jsonl`` (phases keyed by END hour) and parse lazily:
open reads nothing from them; a windowed query parses only the hours it
touches, through an LRU block cache with per-file byte-offset tail refresh.
A v1 flat ``{stream}.jsonl`` still reads as one "legacy" partition. Lane
metadata for selection resolution is folded into ``lane_catalog.json`` at
flush time so it never needs a stream scan.
"""

from __future__ import annotations

import calendar
import io
import json
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import polars as pl

from miles.monitoring.aggregation.timeline import (
    _engine_labels,
    _metric_by_suffix,
    derived_series,
    fleet_summary,
    fold_dp_ranks,
    heatmap,
    lifecycle_heatmap,
    minmax_downsample,
    outliers,
    phases_by_lane,
    stride_downsample,
    trajectory_lanes,
)
from miles.monitoring.collection.records import (
    _RECORD_TYPE_OF_STREAM as _RECORD_TYPE_OF_STREAM,
)
from miles.monitoring.collection.records import (
    INITIALIZE_PHASE as INITIALIZE_PHASE,
)
from miles.monitoring.collection.records import (
    DataBufferSample as DataBufferSample,
)
from miles.monitoring.collection.records import (
    EngineInfo as EngineInfo,
)
from miles.monitoring.collection.records import (
    EngineSample as EngineSample,
)
from miles.monitoring.collection.records import (
    GpuProcessSample as GpuProcessSample,
)
from miles.monitoring.collection.records import (
    GpuSample as GpuSample,
)
from miles.monitoring.collection.records import (
    Meta as Meta,
)
from miles.monitoring.collection.records import (
    MetricsRecord as MetricsRecord,
)
from miles.monitoring.collection.records import (
    PhaseEvent as PhaseEvent,
)
from miles.monitoring.collection.records import (
    Record as Record,
)
from miles.monitoring.collection.records import (
    Role as Role,
)
from miles.monitoring.collection.records import (
    RuntimeEvent as RuntimeEvent,
)
from miles.monitoring.collection.records import (
    Stream as Stream,
)
from miles.monitoring.collection.records import (
    TopologySnapshot as TopologySnapshot,
)
from miles.monitoring.collection.records import (
    TrajectoryEvent as TrajectoryEvent,
)
from miles.monitoring.collection.records import (
    TrajectoryEventKind as TrajectoryEventKind,
)


def _hour_key(ts: float) -> str:
    return time.strftime("%Y%m%d_%H", time.gmtime(ts))


def _hour_bounds(key: str) -> tuple[float, float]:
    start = calendar.timegm(time.strptime(key, "%Y%m%d_%H"))
    return float(start), float(start + 3600)


class _PartitionReader:
    """One hourly-partitioned JSONL stream on the read side.

    Files parse lazily at first windowed read into an LRU block cache; a
    cached block picks up appended bytes through a per-file byte offset with
    the same complete-line rule as ``MetricStore.follow``. A v1 flat
    ``{stream}.jsonl`` participates as one "legacy" partition spanning all
    time, so every pre-partition dump dir keeps working.
    """

    LEGACY_KEY: ClassVar[str] = "legacy"

    def __init__(
        self,
        dir: Path,
        legacy_path: Path,
        *,
        parse: Callable[[bytes, Path, int], Any],
        concat: Callable[[list], Any],
        length: Callable[[Any], int],
        line_stamps: Callable[[dict], tuple],
        max_blocks: int,
    ):
        self.dir = dir
        self.legacy_path = legacy_path
        self._parse = parse
        self._concat = concat
        self._length = length
        self._line_stamps = line_stamps
        self._max_blocks = max_blocks
        self._blocks: OrderedDict[str, Any] = OrderedDict()
        self._offsets: dict[str, int] = {}
        self._lock = threading.Lock()

    def files(self) -> list[tuple[str, Path]]:
        out = [(self.LEGACY_KEY, self.legacy_path)] if self.legacy_path.exists() else []
        if self.dir.is_dir():
            out.extend(sorted((path.stem, path) for path in self.dir.glob("*.jsonl")))
        return out

    def has_data(self) -> bool:
        return any(path.stat().st_size > 0 for _, path in self.files())

    def window(self, t0: float | None, t1: float | None) -> Any:
        """Concatenated block over the partitions intersecting [t0, t1]."""
        blocks = []
        for key, path in self.files():
            if key != self.LEGACY_KEY:
                lo, hi = _hour_bounds(key)
                if (t0 is not None and hi <= t0) or (t1 is not None and lo > t1):
                    continue
            blocks.append(self._block(key, path))
        return self._concat([block for block in blocks if self._length(block)])

    def refresh_cached(self) -> int:
        """Tail-refresh every cached block; returns rows appended. Partitions
        never read remain lazy — they parse fully at their first read."""
        paths = dict(self.files())
        added = 0
        for key in list(self._blocks):
            if key in paths:
                before = self._length(self._blocks[key])
                self._block(key, paths[key])
                added += self._length(self._blocks[key]) - before
        return added

    def _block(self, key: str, path: Path) -> Any:
        # read+parse release the GIL; two fills from one stale offset append the bytes twice
        with self._lock:
            offset = self._offsets.get(key, 0)
            if key in self._blocks and path.stat().st_size <= offset:
                self._blocks.move_to_end(key)
                return self._blocks[key]
            with open(path, "rb") as f:
                f.seek(offset)
                chunk = f.read()
            end = chunk.rfind(b"\n")
            if end >= 0:
                new = self._parse(chunk[: end + 1], path, offset)
                pieces = [self._blocks[key], new] if key in self._blocks else [new]
                self._blocks[key] = self._concat([piece for piece in pieces if self._length(piece)])
                self._offsets[key] = offset + end + 1
            elif key not in self._blocks:
                self._blocks[key] = self._concat([])
                self._offsets[key] = offset
            self._blocks.move_to_end(key)
            while len(self._blocks) > self._max_blocks:
                evicted, _ = self._blocks.popitem(last=False)
                self._offsets.pop(evicted, None)
            return self._blocks[key]

    def edge_stamps(self) -> list[float]:
        """Global-range candidates from the first line of the oldest file and
        the last complete line of the newest — no full scan."""
        files = [(key, path) for key, path in self.files() if path.stat().st_size > 0]
        if not files:
            return []
        lines = []
        with open(files[0][1], "rb") as f:
            lines.append(f.readline())
        with open(files[-1][1], "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 65536))
            tail = f.read()
        complete = tail[: tail.rfind(b"\n") + 1]
        if complete:
            lines.append(complete.splitlines()[-1])
        stamps: list[float] = []
        for line in lines:
            try:
                stamps.extend(self._line_stamps(json.loads(line)))
            except json.JSONDecodeError:
                pass  # partial line mid-append: no complete record to stamp yet
        return stamps


class MetricStore:
    COLUMNAR_STREAMS: ClassVar[tuple[Stream, ...]] = (Stream.GPU_UTIL, Stream.ENGINE_SERIES)
    PARTITIONED_STREAMS: ClassVar[tuple[Stream, ...]] = (
        Stream.GPU_UTIL,
        Stream.ENGINE_SERIES,
        Stream.PHASES,
        Stream.TRAJECTORIES,
        Stream.GPU_PROCESSES,
    )
    MAX_WINDOW_S: ClassVar[float] = 4 * 3600.0
    PARTITION_CACHE_BLOCKS: ClassVar[int] = 24
    CATALOG_FILENAME: ClassVar[str] = "lane_catalog.json"
    FRAME_SCHEMAS: ClassVar[dict[Stream, dict[str, Any]]] = {
        Stream.GPU_UTIL: dict(
            ts=pl.Float64, node=pl.String, gpu=pl.Int64, util=pl.Int64, mem_mb=pl.Int64, power_w=pl.Int64
        ),
        # labels dicts are canonicalized to a JSON string column at parse time
        Stream.ENGINE_SERIES: dict(
            ts=pl.Float64, addr=pl.String, metric=pl.String, labels_json=pl.String, value=pl.Float64
        ),
    }

    def __init__(self, dir: Path | str):
        self.dir = Path(dir)
        self.meta: Meta | None = None
        self.records: dict[Stream, list[Record]] = {s: [] for s in Stream if s not in self.PARTITIONED_STREAMS}
        self._buffers: dict[Stream, list[Record]] = {s: [] for s in Stream}
        self._offsets: dict[Stream, int] = {s: 0 for s in Stream if s not in self.PARTITIONED_STREAMS}
        self._readers: dict[Stream, _PartitionReader] = {s: self._make_reader(s) for s in self.PARTITIONED_STREAMS}
        self._catalog: dict[str, dict] | None = None  # write-side lane catalog accumulator
        # full-window engine query cache, versioned by partition file sizes so
        # appended scrapes invalidate it; the lock is load-bearing because sync
        # endpoints serve concurrently from FastAPI's threadpool
        self._engine_cache_lock = threading.Lock()
        self._engine_cache_version: tuple | None = None
        self._engine_frame_cache: pl.DataFrame | None = None
        self._engine_parts_cache: dict[tuple[str, bool], dict[tuple[str, str], pl.DataFrame]] = {}

    def _make_reader(self, stream: Stream) -> _PartitionReader:
        if stream in self.COLUMNAR_STREAMS:
            empty = pl.DataFrame(schema=self.FRAME_SCHEMAS[stream])

            def concat_frames(blocks: list, empty: pl.DataFrame = empty) -> pl.DataFrame:
                if not blocks:
                    return empty
                merged = blocks[0] if len(blocks) == 1 else pl.concat(blocks, rechunk=False)
                # bound chunk fragmentation from long tail sessions
                return merged.rechunk() if merged.n_chunks() > 256 else merged

            return _PartitionReader(
                self.dir / stream.value,
                self.dir / stream.filename,
                parse=lambda raw, path, offset, stream=stream: self._parse_columnar(stream, raw, path, offset),
                concat=concat_frames,
                length=lambda frame: frame.height,
                line_stamps=lambda data: (data["ts"],),
                max_blocks=self.PARTITION_CACHE_BLOCKS,
            )
        assert stream in (Stream.PHASES, Stream.TRAJECTORIES, Stream.GPU_PROCESSES), stream
        record_type = _RECORD_TYPE_OF_STREAM[stream]
        return _PartitionReader(
            self.dir / stream.value,
            self.dir / stream.filename,
            parse=lambda raw, path, offset, record_type=record_type: self._parse_object_lines(
                record_type, raw, path, offset
            ),
            concat=lambda blocks: [event for block in blocks for event in block],
            length=len,
            # the record's own timestamps() skips the open-interval t1 sentinel
            line_stamps=lambda data, record_type=record_type: record_type.from_dict(data).timestamps(),
            max_blocks=self.PARTITION_CACHE_BLOCKS,
        )

    # ------------------------------ write side ------------------------------

    def append(self, record: Record) -> None:
        self._buffers[record.stream].append(record)

    def buffered_count(self, stream: Stream) -> int:
        return len(self._buffers[stream])

    def drop_oldest_buffered(self, stream: Stream, *, keep_ratio: float = 0.9) -> int:
        """Drop the oldest buffered (not yet flushed) records of one stream and
        return how many were dropped. Lets the collector bound memory when the
        disk stays unwritable instead of growing without limit."""
        buffer = self._buffers[stream]
        dropped = max(1, int(len(buffer) * (1 - keep_ratio)))
        del buffer[:dropped]
        return dropped

    def write_meta(self, meta: Meta) -> None:
        self.meta = meta
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / Meta.FILENAME).write_text(json.dumps(asdict(meta), indent=1))

    def flush(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._merge_catalog()
        for stream, buffer in self._buffers.items():
            if not buffer:
                continue
            if stream in self.PARTITIONED_STREAMS:
                subdir = self.dir / stream.value
                subdir.mkdir(exist_ok=True)
                groups: dict[str, list[Record]] = {}
                for record in buffer:
                    # phases key by END hour (the completion append is what
                    # lands on disk, keeping the window lower bound exact);
                    # OPEN markers have no end yet and key by their start
                    if stream is Stream.PHASES:
                        ts = record.t0 if record.open else record.t1
                    else:
                        ts = record.ts
                    groups.setdefault(_hour_key(ts), []).append(record)
                for key, group in sorted(groups.items()):
                    with open(subdir / f"{key}.jsonl", "a") as f:
                        f.write("".join(json.dumps(r.to_dict(), separators=(",", ":")) + "\n" for r in group))
            else:
                with open(self.dir / stream.filename, "a") as f:
                    f.write("".join(json.dumps(r.to_dict(), separators=(",", ":")) + "\n" for r in buffer))
            buffer.clear()

    def _merge_catalog(self) -> None:
        """Fold buffered records into ``lane_catalog.json``: per-lane train
        ranks, engine addrs and roles, so selection resolution and heatmap
        rows never need a stream scan. Atomic rewrite, only when changed."""
        if self._catalog is None:
            path = self.dir / self.CATALOG_FILENAME
            self._catalog = json.loads(path.read_text())["lanes"] if path.exists() else {}
        lanes = self._catalog
        dirty = False

        def entry(node: str, gpu: int) -> dict:
            nonlocal dirty
            key = f"{node}:{gpu}"
            if key not in lanes:
                lanes[key] = dict(ranks=[], engine_addrs=[], roles=[])
                dirty = True
            return lanes[key]

        def add(values: list, value) -> None:
            nonlocal dirty
            if value not in values:
                values.append(value)
                dirty = True

        for event in self._buffers[Stream.PHASES]:
            if event.role != Role.TRAIN:
                continue
            for gpu in event.gpus:
                lane = entry(event.node, gpu)
                add(lane["roles"], Role.TRAIN.value)
                if event.rank >= 0:
                    add(lane["ranks"], event.rank)
        for snapshot in self._buffers[Stream.TOPOLOGY]:
            for engine in snapshot.engines:
                for node, gpu in engine.gpus:
                    lane = entry(node, gpu)
                    add(lane["roles"], "rollout")
                    add(lane["engine_addrs"], engine.addr)
        for sample in self._buffers[Stream.GPU_UTIL]:
            entry(sample.node, sample.gpu)
        if dirty:
            for lane in lanes.values():
                lane["ranks"].sort()
                lane["engine_addrs"].sort()
                lane["roles"].sort()
            tmp = self.dir / (self.CATALOG_FILENAME + ".tmp")
            tmp.write_text(json.dumps(dict(version=1, lanes=lanes)))
            os.replace(tmp, self.dir / self.CATALOG_FILENAME)

    # ------------------------------ read side -------------------------------

    @classmethod
    def load(cls, dir: Path | str) -> MetricStore:
        store = cls(dir)
        meta_path = store.dir / Meta.FILENAME
        if meta_path.exists():
            store.meta = Meta(**json.loads(meta_path.read_text()))
        store.follow()
        return store

    def follow(self) -> int:
        """Read records appended since the last load()/follow(). Returns the count.

        Only complete lines (terminated by a newline) are consumed, so a
        crash-interrupted partial write at the tail is left for a later call
        once its writer completes it. A malformed *complete* line is real
        corruption and raises — for hour-partitioned streams that happens at
        the first windowed read of the bad partition, since those parse
        lazily; here only already-cached partition blocks refresh.
        """
        num_new = 0
        for stream in self.records:
            path = self.dir / stream.filename
            if not path.exists():
                continue
            with open(path, "rb") as f:
                f.seek(self._offsets[stream])
                chunk = f.read()
            end = chunk.rfind(b"\n")
            if end < 0:
                continue
            complete = chunk[: end + 1]
            record_type = _RECORD_TYPE_OF_STREAM[stream]
            for line in complete.splitlines():
                try:
                    self.records[stream].append(record_type.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    raise ValueError(
                        f"corrupt record in {path} near byte {self._offsets[stream]}: {line[:200]!r}"
                    ) from e
                num_new += 1
            self._offsets[stream] += len(complete)
        for reader in self._readers.values():
            num_new += reader.refresh_cached()
        return num_new

    def _parse_columnar(self, stream: Stream, raw: bytes, path: Path, offset: int) -> pl.DataFrame:
        schema = self.FRAME_SCHEMAS[stream]
        try:
            frame = pl.read_ndjson(io.BytesIO(raw))
            if stream is Stream.ENGINE_SERIES:
                # all-empty labels: read_ndjson drops the column entirely
                dtype = frame.schema.get("labels")
                empty = dtype is None or dtype == pl.Null or (isinstance(dtype, pl.Struct) and not dtype.fields)
                # struct schemas unify per chunk (same-writer key order is stable,
                # so equal label dicts encode to equal strings within a store)
                encoded = pl.lit("{}") if empty else pl.col("labels").struct.json_encode()
                frame = frame.with_columns(encoded.alias("labels_json")).drop("labels", strict=False)
            if set(frame.columns) != set(schema):
                raise ValueError(f"fields {sorted(frame.columns)} != schema {sorted(schema)}")
            return frame.select([pl.col(name).cast(dtype) for name, dtype in schema.items()])
        except (pl.exceptions.PolarsError, ValueError) as e:
            raise ValueError(f"corrupt record in {path} near byte {offset}: {e}") from e

    def _parse_object_lines(self, record_type: type[Record], raw: bytes, path: Path, offset: int) -> list[Record]:
        events = []
        for line in raw.splitlines():
            try:
                events.append(record_type.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                raise ValueError(f"corrupt record in {path} near byte {offset}: {line[:200]!r}") from e
        return events

    def trajectory_events(self, t0: float | None = None, t1: float | None = None) -> list[TrajectoryEvent]:
        """Lifecycle events in [t0, t1], keyed by emission ts (no slack here —
        the consuming-step join and its lookback window live in the reader)."""
        return self._window_records(self._readers[Stream.TRAJECTORIES].window(t0, t1), t0, t1)

    @staticmethod
    def _window_records(events: list[TrajectoryEvent], t0: float | None, t1: float | None) -> list[TrajectoryEvent]:
        return [e for e in events if (t0 is None or e.ts >= t0) and (t1 is None or e.ts <= t1)]

    def trajectory_lanes(self, *, t0=None, t1=None, sample_indices=None):
        return trajectory_lanes(self.trajectory_events(t0, t1), sample_indices=sample_indices)

    def phase_events(self, t0: float | None = None, t1: float | None = None) -> list[PhaseEvent]:
        """Raw phase events, open markers included — advisory heuristics need
        the unexpanded events rather than the lane-resolved view."""
        return self._phase_events(t0, t1)

    def _phase_events(self, t0: float | None, t1: float | None) -> list[PhaseEvent]:
        upper = None if t1 is None else t1 + self.MAX_WINDOW_S
        return self._readers[Stream.PHASES].window(None, upper)

    def has_stream(self, stream: Stream) -> bool:
        if stream in self.PARTITIONED_STREAMS:
            return self._readers[stream].has_data()
        return bool(self.records[stream])

    def iter_records(self, stream: Stream) -> list[Record]:
        """Records as dataclass objects — slow materialization for tests/tools."""
        if stream is Stream.GPU_UTIL:
            frame = self._readers[stream].window(None, None)
            return [GpuSample(**row) for row in frame.iter_rows(named=True)]
        if stream is Stream.ENGINE_SERIES:
            return [
                EngineSample(
                    ts=row["ts"],
                    addr=row["addr"],
                    metric=row["metric"],
                    labels={k: v for k, v in json.loads(row["labels_json"]).items() if v is not None},
                    value=row["value"],
                )
                for row in self._readers[stream].window(None, None).iter_rows(named=True)
            ]
        if stream is Stream.PHASES:
            return list(self._phase_events(None, None))
        if stream is Stream.TRAJECTORIES:
            return self.trajectory_events()
        if stream is Stream.GPU_PROCESSES:
            return self._readers[stream].window(None, None)
        return list(self.records[stream])

    @staticmethod
    def _window(frame: pl.DataFrame, t0: float | None, t1: float | None) -> pl.DataFrame:
        if t0 is not None:
            frame = frame.filter(pl.col("ts") >= t0)
        if t1 is not None:
            frame = frame.filter(pl.col("ts") <= t1)
        return frame

    # ------------------------------- queries --------------------------------

    def metric_keys(self) -> list[str]:
        keys: set[str] = set()
        for record in self.records[Stream.METRICS]:
            keys.update(record.metrics.keys())
        return sorted(keys)

    def step_keys(self) -> list[str]:
        return sorted({r.step_key for r in self.records[Stream.METRICS] if r.step_key is not None})

    def step_key_of_metric_key(self) -> dict[str, str]:
        ans: dict[str, str] = {}
        for record in self.records[Stream.METRICS]:
            if record.step_key is None:
                continue
            for key in record.metrics:
                ans[key] = record.step_key
        return ans

    def metric_series(
        self, keys: list[str], *, x_key: str, t0: float | None = None, t1: float | None = None
    ) -> dict[str, dict[str, list]]:
        """Series for each key, restricted to records logged against ``x_key``
        (e.g. "rollout/step") so values from different step axes never mix."""
        out: dict[str, dict[str, list]] = {k: {"x": [], "y": [], "ts": []} for k in keys}
        for record in self.records[Stream.METRICS]:
            if record.step_key != x_key:
                continue
            if (t0 is not None and record.ts < t0) or (t1 is not None and record.ts > t1):
                continue
            for k in keys:
                if k in record.metrics:
                    series = out[k]
                    series["x"].append(record.step)
                    series["y"].append(record.metrics[k])
                    series["ts"].append(record.ts)
        return out

    def time_range(self) -> tuple[float, float] | None:
        stamps = [ts for records in self.records.values() for record in records for ts in record.timestamps()]
        for reader in self._readers.values():
            stamps.extend(reader.edge_stamps())
        if not stamps:
            return None
        return min(stamps), max(stamps)

    # --------------------------- timeline queries ---------------------------

    def lanes(self) -> list[dict]:
        """Every (node, gpu) seen in any stream — one timeline lane each."""
        return [dict(node=entry["node"], gpu=entry["gpu"], index=entry["index"]) for entry in self.lane_index()]

    def topology_windows(self) -> list[dict]:
        """Engine topology snapshots with validity windows: snapshot N is valid
        [ts_N, ts_{N+1}); the last one is open-ended (t1=None)."""
        snapshots = sorted(self.records[Stream.TOPOLOGY], key=lambda s: s.ts)
        return [
            dict(
                t0=snapshot.ts,
                t1=snapshots[i + 1].ts if i + 1 < len(snapshots) else None,
                engines=[asdict(engine) for engine in snapshot.engines],
            )
            for i, snapshot in enumerate(snapshots)
        ]

    def latest_data_buffer_length(self) -> int | None:
        """Most recent ``RolloutDataSourceWithBuffer`` buffer length, or None
        when the run's data source never reported one (plain RolloutDataSource,
        or a dump predating this probe)."""
        records = self.records[Stream.DATA_BUFFER]
        return records[-1].length if records else None

    def phases_by_lane(self, *, t0=None, t1=None, lanes=None):
        return phases_by_lane(
            self._phase_events(t0, t1),
            edge=(self.time_range() or (0.0, 0.0))[1],
            windows=self.topology_windows(),
            known_lanes=self.lanes(),
            meta=self.meta,
            t0=t0,
            t1=t1,
            lanes=lanes,
        )

    def gpu_series(
        self,
        *,
        t0: float | None = None,
        t1: float | None = None,
        max_points: int = 2000,
        lanes: set[tuple[str, int]] | None = None,
    ) -> dict[str, dict]:
        """Per-lane NVML series, min/max-downsampled on util (the primary
        signal); mem/power take the same indices so all arrays stay aligned."""
        frame = self._window(self._readers[Stream.GPU_UTIL].window(t0, t1), t0, t1)
        out = {}
        for (node, gpu), part in sorted(frame.partition_by(["node", "gpu"], as_dict=True).items()):
            if lanes is not None and (node, gpu) not in lanes:
                continue
            part = part.sort("ts")
            util = part["util"].to_numpy()
            indices, _ = minmax_downsample(np.arange(part.height), util, max_points)
            out[f"{node}:{gpu}"] = dict(
                ts=part["ts"].to_numpy()[indices].tolist(),
                util=util[indices].tolist(),
                mem_mb=part["mem_mb"].to_numpy()[indices].tolist(),
                power_w=part["power_w"].to_numpy()[indices].tolist(),
            )
        return out

    def gpu_processes(
        self, *, t0: float | None = None, t1: float | None = None, lanes: set[tuple[str, int]] | None = None
    ) -> list[dict]:
        """Per-process memory snapshots in [t0, t1] (coarser cadence than
        ``gpu_series``): the timeline hover resolves the nearest snapshot per
        lane client-side, same as it already does for the util/mem series."""
        events = self._window_records(self._readers[Stream.GPU_PROCESSES].window(t0, t1), t0, t1)
        return [
            dict(ts=e.ts, node=e.node, gpu=e.gpu, pid=e.pid, name=e.name, mem_mb=e.mem_mb)
            for e in events
            if lanes is None or (e.node, e.gpu) in lanes
        ]

    # cumulative engine families are stored raw; the catalog exposes them as
    # per-interval derived series instead
    ENGINE_RATE_METRICS: ClassVar[dict[str, str]] = {
        "sglang_prompt_tokens_per_s": "sglang_prompt_tokens_total",
        "sglang_generation_tokens_per_s": "sglang_generation_tokens_total",
        "sglang_requests_per_s": "sglang_num_requests_total",
        "sglang_aborted_requests_per_s": "sglang_num_aborted_requests_total",
    }
    ENGINE_MEAN_METRICS: ClassVar[dict[str, str]] = {
        "sglang_ttft_mean_s": "sglang_time_to_first_token_seconds",
        "sglang_inter_token_latency_mean_s": "sglang_inter_token_latency_seconds",
        "sglang_time_per_output_token_mean_s": "sglang_time_per_output_token_seconds",
        "sglang_e2e_latency_mean_s": "sglang_e2e_request_latency_seconds",
    }
    _CUMULATIVE_SUFFIXES: ClassVar[tuple[str, ...]] = ("_total", "_sum", "_count")

    # engine-level aggregation across dp ranks: intensive metrics average,
    # everything else (queue depths, throughputs, cumulative counters) sums
    ENGINE_MEAN_AGGREGATED_METRICS: ClassVar[frozenset[str]] = frozenset(
        {"sglang_token_usage", "sglang_cache_hit_rate", "sglang_kv_transfer_latency_ms"}
    )

    def engine_metric_names(self) -> list[str]:
        """Distinct scraped engine metrics — the L0 sglang category catalog."""
        frame, _ = self._engine_cache()
        raw = set(frame.get_column("metric").unique())
        names = {name for name in raw if not name.endswith(self._CUMULATIVE_SUFFIXES)}
        names.update(derived for derived, base in self.ENGINE_RATE_METRICS.items() if base in raw)
        names.update(
            derived
            for derived, base in self.ENGINE_MEAN_METRICS.items()
            if base + "_sum" in raw and base + "_count" in raw
        )
        return sorted(names)

    def engine_series(
        self,
        metric: str,
        *,
        t0: float | None = None,
        t1: float | None = None,
        max_points: int = 2000,
        per_dp_rank: bool = False,
    ) -> list[dict]:
        """One series per (engine addr, label set) for the given metric.

        By default per-dp-rank samples (dp-attention engines export one per
        rank) are folded into one engine-level series — summed, or averaged
        for ``ENGINE_MEAN_AGGREGATED_METRICS``; ``per_dp_rank=True`` keeps
        them apart, with ``dp_rank`` in each series' labels.

        Derived names (``ENGINE_RATE_METRICS`` / ``ENGINE_MEAN_METRICS``) are
        computed from adjacent raw cumulative samples; a counter reset or an
        interval without completions leaves a gap, never a fabricated value."""
        if metric in self.ENGINE_RATE_METRICS:
            parts = self._engine_parts(self.ENGINE_RATE_METRICS[metric], t0, t1, per_dp_rank=per_dp_rank)
            return self._derived_series(parts, None, max_points)
        if metric in self.ENGINE_MEAN_METRICS:
            base = self.ENGINE_MEAN_METRICS[metric]
            return self._derived_series(
                self._engine_parts(base + "_sum", t0, t1, per_dp_rank=per_dp_rank),
                self._engine_parts(base + "_count", t0, t1, per_dp_rank=per_dp_rank),
                max_points,
            )
        out = []
        for (addr, labels_json), part in self._engine_parts(metric, t0, t1, per_dp_rank=per_dp_rank).items():
            ts, values = stride_downsample(part["ts"].to_numpy(), part["value"].to_numpy(), max_points)
            out.append(dict(addr=addr, labels=_engine_labels(labels_json), ts=ts.tolist(), value=values.tolist()))
        return out

    def _engine_parts(
        self, metric: str, t0: float | None, t1: float | None, *, per_dp_rank: bool = False
    ) -> dict[tuple[str, str], pl.DataFrame]:
        if t0 is None and t1 is None:
            return self._engine_parts_cached(metric, per_dp_rank)
        frame = self._window(self._readers[Stream.ENGINE_SERIES].window(t0, t1), t0, t1).filter(
            pl.col("metric") == metric
        )
        if not per_dp_rank:
            frame = self._fold_dp_ranks(frame, metric)
        return {
            key: part.sort("ts")
            for key, part in sorted(frame.partition_by(["addr", "labels_json"], as_dict=True).items())
        }

    def _fold_dp_ranks(self, frame, metric):
        return fold_dp_ranks(frame, metric, self.ENGINE_MEAN_AGGREGATED_METRICS)

    def _engine_cache(self) -> tuple[pl.DataFrame, dict[tuple[str, bool], dict[tuple[str, str], pl.DataFrame]]]:
        """The full-window frame and its partition cache, returned as a pair so a
        caller racing a rebuild files partitions under the frame they came from."""
        reader = self._readers[Stream.ENGINE_SERIES]
        with self._engine_cache_lock:
            version = tuple((key, path.stat().st_size) for key, path in reader.files())
            if version != self._engine_cache_version:
                self._engine_frame_cache = reader.window(None, None)
                self._engine_parts_cache = {}
                self._engine_cache_version = version  # last: a failed rebuild must not look cached
            return self._engine_frame_cache, self._engine_parts_cache

    def _engine_parts_cached(self, metric: str, per_dp_rank: bool) -> dict[tuple[str, str], pl.DataFrame]:
        frame, cache = self._engine_cache()
        cache_key = (metric, per_dp_rank)
        parts = cache.get(cache_key)
        if parts is None:
            frame = frame.filter(pl.col("metric") == metric)
            if not per_dp_rank:
                frame = self._fold_dp_ranks(frame, metric)
            parts = {
                key: part.sort("ts")
                for key, part in sorted(frame.partition_by(["addr", "labels_json"], as_dict=True).items())
            }
            cache[cache_key] = parts
        return parts

    _derived_series = staticmethod(derived_series)

    def lane_index(self) -> list[dict]:
        """Per-lane metadata for selection resolution: train ranks observed on
        the lane, engine addrs that ever covered it, and derived roles.

        Reads ``lane_catalog.json`` (merge-written at flush time) when
        present; a legacy dir without one falls back to a full-stream scan.
        """
        path = self.dir / self.CATALOG_FILENAME
        if path.exists():
            lanes = json.loads(path.read_text())["lanes"]
            out = []
            for key, entry in lanes.items():
                node, _, gpu = key.rpartition(":")
                out.append(
                    dict(
                        node=node,
                        gpu=int(gpu),
                        ranks=entry["ranks"],
                        engine_addrs=entry["engine_addrs"],
                        roles=entry["roles"],
                    )
                )
            out.sort(key=lambda entry: (entry["node"], entry["gpu"]))
            for i, entry in enumerate(out):
                entry["index"] = i
            return out

        seen: set[tuple[str, int]] = set(
            self._readers[Stream.GPU_UTIL].window(None, None).select("node", "gpu").unique().iter_rows()
        )
        events = self._phase_events(None, None)
        for event in events:
            for gpu in event.gpus:
                seen.add((event.node, gpu))
        for snapshot in self.records[Stream.TOPOLOGY]:
            for engine in snapshot.engines:
                for node, gpu in engine.gpus:
                    seen.add((node, gpu))
        info: dict[tuple[str, int], dict] = {
            (node, gpu): dict(node=node, gpu=gpu, ranks=set(), engine_addrs=set(), roles=set())
            for node, gpu in sorted(seen)
        }
        for event in events:
            if event.role != Role.TRAIN:
                continue
            for gpu in event.gpus:
                entry = info.get((event.node, gpu))
                if entry is not None:
                    entry["roles"].add(Role.TRAIN.value)
                    if event.rank >= 0:
                        entry["ranks"].add(event.rank)
        for snapshot in self.records[Stream.TOPOLOGY]:
            for engine in snapshot.engines:
                for node, gpu in engine.gpus:
                    entry = info.get((node, gpu))
                    if entry is not None:
                        entry["roles"].add("rollout")
                        entry["engine_addrs"].add(engine.addr)
        return [
            dict(
                node=entry["node"],
                gpu=entry["gpu"],
                index=i,
                ranks=sorted(entry["ranks"]),
                engine_addrs=sorted(entry["engine_addrs"]),
                roles=sorted(entry["roles"]),
            )
            for i, entry in enumerate(info.values())
        ]

    def resolve_lanes(self, grammar: str | None) -> set[tuple[str, int]] | None:
        """Parse the lane-selection grammar into a lane set (None = all lanes).

        Comma-separated selectors: ``rank:5`` / ``rank:0-7`` (train ranks),
        ``g:5`` / ``g:0-31`` (global lane numbers), ``node:<ip>``,
        ``gpu:<node>:<index>``, ``engine:<addr substring>``,
        ``role:train`` / ``role:rollout``, ``every:<stride>`` (equidistant
        global-index sampling), or ``all``. Unknown selector syntax
        raises; a valid selector matching nothing selects nothing.
        """
        if grammar is None or grammar.strip() in ("", "all"):
            return None
        index = self.lane_index()
        selected: set[tuple[str, int]] = set()
        for raw_token in grammar.split(","):
            token = raw_token.strip()
            if not token:
                continue
            kind, _, value = token.partition(":")
            if not value:
                raise ValueError(f"bad lane selector {token!r}: expected kind:value")
            if kind == "rank":
                lo, _, hi = value.partition("-")
                ranks = set(range(int(lo), int(hi if hi else lo) + 1))
                selected |= {(e["node"], e["gpu"]) for e in index if ranks & set(e["ranks"])}
            elif kind == "g":
                # cluster-global lane numbers (the g{index} labels on every view)
                lo, _, hi = value.partition("-")
                positions = set(range(int(lo), int(hi if hi else lo) + 1))
                selected |= {(e["node"], e["gpu"]) for e in index if e["index"] in positions}
            elif kind == "node":
                selected |= {(e["node"], e["gpu"]) for e in index if e["node"] == value}
            elif kind == "gpu":
                node, _, gpu = value.rpartition(":")
                if not node:
                    raise ValueError(f"bad lane selector {token!r}: expected gpu:<node>:<index>")
                selected.add((node, int(gpu)))
            elif kind == "engine":
                selected |= {(e["node"], e["gpu"]) for e in index if any(value in a for a in e["engine_addrs"])}
            elif kind == "every":
                stride = int(value)
                if stride <= 0:
                    raise ValueError(f"bad lane selector {token!r}: stride must be positive")
                selected |= {(e["node"], e["gpu"]) for e in index if e["index"] % stride == 0}
            elif kind == "role":
                if value not in ("train", "rollout"):
                    raise ValueError(f"bad lane selector {token!r}: role must be train or rollout")
                selected |= {(e["node"], e["gpu"]) for e in index if value in e["roles"]}
            else:
                raise ValueError(f"unknown lane selector kind {kind!r} in {token!r}")
        return selected

    HEATMAP_MAGNITUDE_FIELDS: ClassVar[tuple[str, ...]] = ("util", "mem_mb", "power_w")

    def heatmap(self, metric, *, t0=None, t1=None, x_buckets=1200, lanes=None):
        window = self.time_range()
        if window is not None:
            t0 = window[0] if t0 is None else t0
            t1 = window[1] if t1 is None else t1
            if metric == "lifecycle":
                assert x_buckets >= 2 and t1 > t0
                return self._lifecycle_heatmap(t0, t1, x_buckets)
        frame = (
            self._window(self._readers[Stream.GPU_UTIL].window(t0, t1), t0, t1)
            if window is not None and metric in self.HEATMAP_MAGNITUDE_FIELDS
            else None
        )
        events = self.phases_by_lane(t0=t0, t1=t1, lanes=lanes) if window is not None and metric == "phase" else []
        return heatmap(
            metric,
            window=window,
            lane_index=self.lane_index() if window else [],
            frame=frame,
            events=events,
            t0=t0,
            t1=t1,
            x_buckets=x_buckets,
            lanes=lanes,
        )

    LIFECYCLE_PALETTE: ClassVar[tuple[str, ...]] = ("", "queue", "generating", "tool_wait")
    LIFECYCLE_MAX_ROWS: ClassVar[int] = 1024  # y-cap: thousands of samples must not explode the canvas

    def _lifecycle_heatmap(self, t0, t1, x_buckets):
        return lifecycle_heatmap(
            self.trajectory_lanes(t0=t0, t1=t1),
            t0,
            t1,
            x_buckets,
            max_rows=self.LIFECYCLE_MAX_ROWS,
            palette=self.LIFECYCLE_PALETTE,
        )

    def fleet(self, *, t0=None, t1=None, x_buckets=600):
        phase = self.heatmap("phase", t0=t0, t1=t1, x_buckets=x_buckets)
        rt0, rt1 = phase["t0"], phase["t1"]
        frame = self._window(self._readers[Stream.GPU_UTIL].window(rt0, rt1), rt0, rt1)
        return fleet_summary(phase, frame, x_buckets=x_buckets)

    def outliers(self, criterion, *, t0=None, t1=None, top_k=16):
        frame = (
            self._window(self._readers[Stream.GPU_UTIL].window(t0, t1), t0, t1) if criterion == "lowest_util" else None
        )
        events = self.phases_by_lane(t0=t0, t1=t1) if criterion.startswith("slowest_phase:") else []
        return outliers(criterion, frame=frame, events=events, top_k=top_k)

    def bubbles(self) -> list[dict]:
        """Per-step bubble summary strip: step time and wait ratio from the
        perf metrics, with the wall-clock ts as the timeline zoom anchor."""
        out = []
        for record in self.records[Stream.METRICS]:
            if record.step_key is None or not record.step_key.endswith("rollout/step"):
                continue
            step_time = _metric_by_suffix(record.metrics, "perf/step_time")
            wait_ratio = _metric_by_suffix(record.metrics, "perf/wait_time_ratio")
            if step_time is None and wait_ratio is None:
                continue
            out.append(dict(step=record.step, ts=record.ts, step_time=step_time, wait_ratio=wait_ratio))
        return out
