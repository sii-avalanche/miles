"""Durable, bounded PostgreSQL recording for AvaCore rollout attempts."""

import asyncio
import fcntl
import gzip
import hashlib
import json
import logging
import os
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any], *, compressed: bool = False) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            output.write(gzip.compress(encoded, compresslevel=1) if compressed else encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


class DurableRolloutOutbox:
    """Accept on disk first; retry the same immutable database key until acknowledged.

    One rollout actor owns a directory. Atomic files work on shared run mounts;
    no SQLite WAL or node-local temporary directory is required. Successful files
    remain in ``committed`` for recovery and inspection.
    """

    def __init__(
        self,
        path: Path,
        write: Callable[[dict[str, Any]], Awaitable[None]],
        *,
        workers: int = 16,
        retry_initial_seconds: float = 1,
        retry_max_seconds: float = 30,
        on_event: Callable[[str, Exception | None], None] | None = None,
    ) -> None:
        if workers < 1 or retry_initial_seconds <= 0 or retry_max_seconds < retry_initial_seconds:
            raise ValueError("Recording workers and retry intervals must be positive and ordered")
        self.path = Path(path)
        self.pending = self.path / "pending"
        self.committed = self.path / "committed"
        self.counters = self.path / "trials"
        for directory in (self.pending, self.committed, self.counters):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._write = write
        self._worker_count = workers
        self._retry_initial = retry_initial_seconds
        self._retry_max = retry_max_seconds
        self._on_event = on_event or (lambda event, error: None)
        self._queue: asyncio.Queue[Path] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._pending_since: dict[Path, float] = {}
        self._rounds: dict[tuple, tuple[int, set[int]]] = {}
        self._allocation_lock = asyncio.Lock()
        self._enqueue_lock = asyncio.Lock()
        self._in_flight = 0
        self._closed = False

    async def start(self) -> None:
        if self._workers:
            return
        for path in sorted(self.pending.glob("*.json.gz")):
            if path in self._pending_since:
                continue
            self._pending_since[path] = path.stat().st_mtime
            self._queue.put_nowait(path)
            self._emit("recovered", None)
        self._workers = [asyncio.create_task(self._worker()) for _ in range(self._worker_count)]

    def _reserve_round(self, query_id: str, n: int, floor: int, *, reserve: bool = True) -> int:
        key = hashlib.sha256(query_id.encode()).hexdigest()
        # The counter survives even after all pending records have committed.
        with (self.counters / f"{key}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.counters / f"{key}.json"
            state = json.loads(path.read_text()) if path.exists() else {"n": n, "next_round": floor}
            if state["n"] != n:
                raise ValueError("Cannot change n_samples_per_prompt in an existing recording directory")
            current = max(state["next_round"], floor)
            _atomic_json(path, {"query_id": query_id, "n": n, "next_round": current + int(reserve)})
            return current

    async def allocate_trial(self, sample, n: int) -> int:
        if n < 1 or sample.index is None or sample.epoch is None:
            raise ValueError("Recording requires a sample index, epoch and positive group size")
        query_id = str(sample.metadata["_index"])
        key = (query_id, sample.epoch, sample.group_index, getattr(sample, "generation_attempt", 0), n)
        slot = sample.index % n
        async with self._allocation_lock:
            if key not in self._rounds or slot in self._rounds[key][1]:
                generation_round = await asyncio.to_thread(self._reserve_round, query_id, n, sample.epoch)
                self._rounds[key] = (generation_round, set())
            generation_round, slots = self._rounds[key]
            slots.add(slot)
            return generation_round * n + slot

    async def seed_trials(self, keys: Sequence[tuple[str, int]], n: int) -> None:
        floors: dict[str, int] = {}
        for query_id, trial_id in keys:
            floors[query_id] = max(floors.get(query_id, 0), trial_id // n + 1)
        async with self._allocation_lock:
            for query_id, floor in floors.items():
                await asyncio.to_thread(self._reserve_round, query_id, n, floor, reserve=False)

    async def enqueue(self, record: dict[str, Any]) -> Path:
        if self._closed:
            raise RuntimeError("Recording outbox is closed")
        identity = f"{record['query_id']}:{record['trial_id']}"
        path = self.pending / (hashlib.sha256(identity.encode()).hexdigest() + ".json.gz")
        async with self._enqueue_lock:
            existing = path if path.exists() else self.committed / path.name
            if existing.exists():
                if await asyncio.to_thread(self._read, existing) != record:
                    raise ValueError(f"A different recording already owns Trial {identity}")
                return existing
            operation = asyncio.create_task(asyncio.to_thread(_atomic_json, path, record, compressed=True))
            cancelled = False
            try:
                await asyncio.shield(operation)
            except asyncio.CancelledError:
                await operation
                cancelled = True
            self._pending_since[path] = time.time()
            self._queue.put_nowait(path)
            self._emit("scheduled", None)
            if cancelled:
                raise asyncio.CancelledError
        return path

    def _emit(self, event: str, error: Exception | None) -> None:
        try:
            self._on_event(event, error)
        except Exception:
            logger.exception("Recording telemetry callback failed")

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            return json.load(source)

    def _acknowledge(self, path: Path) -> None:
        if path.exists():
            os.replace(path, self.committed / path.name)
        _sync_directory(self.pending)
        _sync_directory(self.committed)

    async def _deliver(self, path: Path) -> None:
        delay = self._retry_initial
        while True:
            try:
                source = path if path.exists() else self.committed / path.name
                record = await asyncio.to_thread(self._read, source)
                await self._write(record)
                await asyncio.to_thread(self._acknowledge, path)
                return
            except Exception as error:
                self._emit("failed", error)
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._retry_max)

    async def _worker(self) -> None:
        while True:
            path = await self._queue.get()
            self._in_flight += 1
            started = time.monotonic()
            try:
                await self._deliver(path)
                self._pending_since.pop(path, None)
                self._queue.task_done()
                self._emit("success", None)
            finally:
                self._in_flight -= 1
                logger.debug("Recording attempt took %.2fs", time.monotonic() - started)

    def stats(self) -> dict[str, float]:
        return {
            "pending_writes": len(self._pending_since),
            "in_flight_writes": self._in_flight,
            "oldest_pending_seconds": max((time.time() - t for t in self._pending_since.values()), default=0),
        }

    async def flush(self, timeout: float | None = None) -> bool:
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
            return True
        except TimeoutError:
            return False

    async def close(self, timeout: float | None = 300) -> bool:
        self._closed = True
        drained = await self.flush(timeout)
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        return drained
