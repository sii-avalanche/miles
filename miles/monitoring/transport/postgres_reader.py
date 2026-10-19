"""Use shared PostgreSQL telemetry with the existing dashboard calculations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from miles.monitoring.transport.postgres import DashboardDatabase
from miles.monitoring.transport.store import _RECORD_TYPE_OF_STREAM, Meta, MetricStore, Stream, _PartitionReader


@dataclass(frozen=True)
class _RemoteFile:
    path: str
    size: int
    first_ts: float | None
    last_ts: float | None

    def stat(self):
        return SimpleNamespace(st_size=self.size)


class _DatabasePartitionReader(_PartitionReader):
    def __init__(self, owner, stream, template):
        super().__init__(
            template.dir,
            template.legacy_path,
            parse=template._parse,
            concat=template._concat,
            length=template._length,
            line_stamps=template._line_stamps,
            max_blocks=template._max_blocks,
        )
        self.owner = owner
        self.stream = stream

    def files(self):
        files = []
        for path, item in self.owner.files.items():
            if path == self.stream.filename:
                key = self.LEGACY_KEY
            elif path.startswith(self.stream.value + "/"):
                key = Path(path).stem
            else:
                continue
            files.append((key, _RemoteFile(path, item["size"], item["first_ts"], item["last_ts"])))
        return sorted(files, key=lambda item: item[0])

    def _block(self, key, path):
        with self._lock:
            offset = self._offsets.get(key, 0)
            if path.size > offset:
                raw = self.owner.database.read(self.owner.run_id, path.path, offset, path.size)
                new = self._parse(raw, Path(path.path), offset)
                previous = [self._blocks[key]] if key in self._blocks else []
                self._blocks[key] = self._concat(previous + [new])
                self._offsets[key] = path.size
            elif key not in self._blocks:
                self._blocks[key] = self._concat([])
            self._blocks.move_to_end(key)
            while len(self._blocks) > self._max_blocks:
                evicted, _ = self._blocks.popitem(last=False)
                self._offsets.pop(evicted, None)
            return self._blocks[key]

    def edge_stamps(self):
        return [ts for _, path in self.files() for ts in (path.first_ts, path.last_ts) if ts is not None]


class PostgresMetricStore(MetricStore):
    def __init__(self, database: DashboardDatabase, run_id: str, cache_dir: Path):
        self.database = database
        self.run_id = run_id
        self.files = {}
        self.metadata = {}
        super().__init__(cache_dir)
        self.follow()

    def _make_reader(self, stream):
        return _DatabasePartitionReader(self, stream, super()._make_reader(stream))

    def follow(self):
        self.metadata, self.files = self.database.describe(self.run_id)
        if self.metadata.get("meta"):
            self.meta = Meta(**self.metadata["meta"])
        added = 0
        for stream in self.records:
            size = self.files.get(stream.filename, {}).get("size", 0)
            offset = self._offsets[stream]
            if size > offset:
                raw = self.database.read(self.run_id, stream.filename, offset, size)
                records = [_RECORD_TYPE_OF_STREAM[stream].from_dict(json.loads(line)) for line in raw.splitlines()]
                self.records[stream].extend(records)
                self._offsets[stream] = size
                added += len(records)
        # Cached partition blocks refresh on demand using the new size snapshot.
        return added

    def lane_index(self):
        lanes = self.metadata.get("lanes", {})
        if not lanes:
            return super().lane_index()
        result = []
        for key, entry in lanes.items():
            node, _, gpu = key.rpartition(":")
            result.append(
                dict(
                    node=node,
                    gpu=int(gpu),
                    ranks=entry["ranks"],
                    engine_addrs=entry["engine_addrs"],
                    roles=entry["roles"],
                )
            )
        result.sort(key=lambda entry: (entry["node"], entry["gpu"]))
        return [dict(entry, index=i) for i, entry in enumerate(result)]

    def engine_metric_names(self):
        raw = {
            key
            for path, item in self.files.items()
            if path == Stream.ENGINE_SERIES.filename or path.startswith(Stream.ENGINE_SERIES.value + "/")
            for key in item["keys"]
        }
        names = {name for name in raw if not name.endswith(self._CUMULATIVE_SUFFIXES)}
        names.update(derived for derived, base in self.ENGINE_RATE_METRICS.items() if base in raw)
        names.update(
            derived
            for derived, base in self.ENGINE_MEAN_METRICS.items()
            if base + "_sum" in raw and base + "_count" in raw
        )
        return sorted(names)
