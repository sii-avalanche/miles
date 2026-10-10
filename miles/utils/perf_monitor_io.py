"""Small file-only helpers: no CUDA context, database or network probes."""

import json
import os
import resource
from pathlib import Path


def append_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")


def host_memory_metrics() -> dict[str, float]:
    metrics = {}
    usage = resource.getrusage(resource.RUSAGE_SELF)
    metrics["process_cpu_seconds_total"] = usage.ru_utime + usage.ru_stime
    try:
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        metrics["process_rss_gb"] = resident_pages * os.sysconf("SC_PAGE_SIZE") / 2**30
        fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        metrics["node_memory_available_gb"] = int(fields["MemAvailable"].split()[0]) / 2**20
        status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
        metrics["process_threads"] = int(status["Threads"])
        metrics["process_virtual_memory_gb"] = int(status["VmSize"].split()[0]) / 2**20
    except (OSError, ValueError, KeyError, IndexError):
        pass
    # Container namespaces usually mount the current cgroup at this root.
    root = Path("/sys/fs/cgroup")
    for key, paths in {
        "container_memory_gb": ("memory.current", "memory/memory.usage_in_bytes"),
        "container_memory_limit_gb": ("memory.max", "memory/memory.limit_in_bytes"),
    }.items():
        for relative in paths:
            try:
                value = int((root / relative).read_text())
                if value < 2**60:  # v1 unlimited sentinel; v2 'max' fails int()
                    metrics[key] = value / 2**30
                break
            except (OSError, ValueError):
                continue
    try:
        events = dict(line.split() for line in (root / "memory.events").read_text().splitlines())
        metrics["container_oom_kill_total"] = int(events["oom_kill"])
    except (OSError, ValueError, KeyError):
        pass
    return metrics
