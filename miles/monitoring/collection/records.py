"""Raw monitoring record schemas shared by producers, reducers and transports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, ClassVar

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum


class Stream(StrEnum):
    METRICS = "metrics"
    PHASES = "phases"
    TOPOLOGY = "topology"
    GPU_UTIL = "gpu_util"
    ENGINE_SERIES = "engine_series"
    TRAJECTORIES = "trajectories"
    GPU_PROCESSES = "gpu_processes"
    DATA_BUFFER = "data_buffer"
    RUNTIME_EVENTS = "runtime_events"

    @property
    def filename(self) -> str:
        return f"{self.value}.jsonl"


@dataclass
class Record:
    stream: ClassVar[Stream]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Record:
        return cls(**data)

    def timestamps(self) -> tuple[float, ...]:
        return (self.ts,)


@dataclass
class MetricsRecord(Record):
    """One ``tracking_utils.log()`` payload."""

    stream: ClassVar[Stream] = Stream.METRICS
    ts: float
    step_key: str | None
    step: int | None
    metrics: dict[str, Any]


class Role(StrEnum):
    TRAIN = "train"
    ROLLOUT_MANAGER = "rollout_manager"
    DERIVED = "derived"  # synthesized by the read side, not observed


@dataclass
class RuntimeEvent(Record):
    """Observed phase status. A failure stage does not identify its root cause."""

    stream: ClassVar[Stream] = Stream.RUNTIME_EVENTS
    ts: float
    role: str
    name: str
    rollout_id: int | None
    t0: float
    t1: float | None
    status: str
    error_type: str | None = None
    details: dict | None = None


# phase name synthesized per lane for [meta.start_ts, first observed event)
INITIALIZE_PHASE = "initialize"


@dataclass
class PhaseEvent(Record):
    """One completed timer interval on one process (Timer sink event).

    ``rollout_manager`` events carry no GPUs (the manager is a driver-side
    process); the timeline queries expand them onto rollout-engine lanes."""

    stream: ClassVar[Stream] = Stream.PHASES
    name: str
    t0: float
    t1: float
    node: str
    gpus: list[int]
    rank: int
    role: str  # a Role value

    # t1 sentinel: the interval was still running when this event was written
    # (Timer.start emits it so long phases are visible before they end); the
    # closing event with the real t1 supersedes it on the read side
    OPEN_T1: ClassVar[float] = -1.0

    @property
    def open(self) -> bool:
        return self.t1 == self.OPEN_T1

    def timestamps(self) -> tuple[float, ...]:
        return (self.t0,) if self.open else (self.t0, self.t1)


@dataclass
class EngineInfo:
    addr: str
    worker_type: str
    engine_rank: int
    gpus: list[list]  # [node_ip, gpu_id] pairs
    gpu_uuids: list[str | None]


@dataclass
class TopologySnapshot(Record):
    """Full engine topology; snapshot N is valid until snapshot N+1."""

    stream: ClassVar[Stream] = Stream.TOPOLOGY
    ts: float
    engines: list[EngineInfo]

    @classmethod
    def from_dict(cls, data: dict) -> TopologySnapshot:
        return cls(ts=data["ts"], engines=[EngineInfo(**e) for e in data["engines"]])


@dataclass
class DataBufferSample(Record):
    """One report of ``RolloutDataSourceWithBuffer.get_buffer_length()``
    (design doc's "show the data status in databuffer" ask). Appended once
    per ``RolloutExecutor.get()`` call — a no-op for plain
    ``RolloutDataSource`` runs, which never buffer samples across steps.
    Low-rate and unpartitioned like ``TopologySnapshot``: only the latest
    value matters for display."""

    stream: ClassVar[Stream] = Stream.DATA_BUFFER
    ts: float
    length: int


class TrajectoryEventKind(StrEnum):
    ATTEMPT_START = "attempt_start"
    GEN_START = "gen_start"
    GEN_END = "gen_end"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    ATTEMPT_END = "attempt_end"


@dataclass
class TrajectoryEvent(Record):
    """One lifecycle boundary of one sample's rollout (design §18.3).

    ``sample_index`` is the run-global ``Sample.index`` — the same key the
    dump join uses, so the read side can resolve a consuming step's batch to
    its events. ``turn`` is 1-based for gen/tool segments and -1 for attempt
    boundaries; ``weight_version`` is the engine-reported version of the
    segment ("" when unknown)."""

    stream: ClassVar[Stream] = Stream.TRAJECTORIES
    ts: float
    kind: str  # a TrajectoryEventKind value
    sample_index: int
    group_index: int  # -1 when the sample carries none
    turn: int
    weight_version: str
    detail: str


@dataclass
class GpuSample(Record):
    """One NVML sample of one GPU."""

    stream: ClassVar[Stream] = Stream.GPU_UTIL
    ts: float
    node: str
    gpu: int
    util: int
    mem_mb: int
    power_w: int


@dataclass
class EngineSample(Record):
    """One scraped sglang engine metric value."""

    stream: ClassVar[Stream] = Stream.ENGINE_SERIES
    ts: float
    addr: str
    metric: str
    labels: dict[str, str]
    value: float


@dataclass
class GpuProcessSample(Record):
    """One NVML per-process memory reading on one GPU, at a coarser cadence
    than GpuSample (design doc's GPU/CPU/disk memory-breakdown TODO — this
    covers the GPU side: who is actually holding the VRAM, not just the
    per-GPU aggregate)."""

    stream: ClassVar[Stream] = Stream.GPU_PROCESSES
    ts: float
    node: str
    gpu: int
    pid: int
    name: str
    mem_mb: int


_RECORD_TYPE_OF_STREAM: dict[Stream, type[Record]] = {
    cls.stream: cls
    for cls in (
        MetricsRecord,
        PhaseEvent,
        TopologySnapshot,
        GpuSample,
        EngineSample,
        TrajectoryEvent,
        GpuProcessSample,
        DataBufferSample,
        RuntimeEvent,
    )
}


@dataclass
class Meta:
    run_name: str
    start_ts: float
    args: dict[str, Any]
    schema_version: int = 1

    FILENAME: ClassVar[str] = "meta.json"
