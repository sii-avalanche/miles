"""Fleet gauges computed from the existing GPU and SGLang observations."""

from miles.monitoring.collection.records import EngineSample, GpuSample, PhaseEvent, Record


def runtime_snapshot(*, gpu_samples, train_gpus, topology, engine_samples, args, gpu_interval, scrape_interval, now):
    inference_gpus = (
        {(node, int(gpu)) for engine in topology.engines for node, gpu in engine.gpus}
        if topology is not None
        else set()
    )
    result = {}
    gpu_max_age = max(10.0, 3 * gpu_interval)
    for role, keys in (("train", train_gpus), ("inference", inference_gpus)):
        fresh = [gpu_samples[key] for key in keys if key in gpu_samples and now - gpu_samples[key].ts <= gpu_max_age]
        result[f"{role}/mapped_gpus"] = len(keys)
        result[f"{role}/sampled_gpus"] = len(fresh)
        snapshot = args
        configured = (
            (snapshot.get("actor_num_nodes") or 0) * (snapshot.get("actor_num_gpus_per_node") or 0)
            if role == "train"
            else (snapshot.get("rollout_num_gpus") or 0)
        )
        expected = max(configured, len(keys))
        result[f"{role}/expected_gpus"] = expected
        if expected:
            result[f"{role}/coverage_fraction"] = len(fresh) / expected
        if fresh:
            result[f"{role}/gpu_util_mean_pct"] = sum(point.util for point in fresh) / len(fresh)
            result[f"{role}/gpu_util_min_pct"] = min(point.util for point in fresh)
            result[f"{role}/gpu_util_max_pct"] = max(point.util for point in fresh)
            result[f"{role}/gpu_memory_max_gb"] = max(point.mem_mb for point in fresh) / 1024
            result[f"{role}/gpu_active_fraction"] = sum(point.util >= 10 for point in fresh) / len(fresh)
    engine_max_age = max(10.0, 3 * scrape_interval)
    active_addrs = {engine.addr for engine in topology.engines} if topology is not None else set()
    aggregations = {
        "sglang_num_running_reqs": ("running_requests", "sum"),
        "sglang_num_queue_reqs": ("queued_requests", "sum"),
        "sglang_gen_throughput": ("generated_tokens_per_s", "sum"),
        "sglang_token_usage": ("kv_usage_mean", "mean"),
        "sglang_mamba_usage": ("mamba_usage_mean", "mean"),
    }
    for metric, (name, aggregation) in aggregations.items():
        fresh = [
            point
            for point in engine_samples
            if point.metric == metric and point.addr in active_addrs and now - point.ts <= engine_max_age
        ]
        result[f"inference/{name}_reporting_engines"] = len({point.addr for point in fresh})
        if fresh:
            total = sum(point.value for point in fresh)
            result[f"inference/{name}"] = total / len(fresh) if aggregation == "mean" else total
            if aggregation == "mean":
                result[f"inference/{name.removesuffix('_mean')}_max"] = max(point.value for point in fresh)
    return result


class FleetStatistics:
    """Latest observations, guarded by the collector's existing ingest lock."""

    def __init__(self):
        self._latest_gpu: dict[tuple[str, int], GpuSample] = {}
        self._latest_running_reqs: dict[str, tuple[float, float]] = {}
        self._latest_phase_seconds: dict[str, float] = {}
        self._train_gpus: set[tuple[str, int]] = set()
        self._latest_engine_samples: dict[tuple, EngineSample] = {}

    def observe(self, record: Record) -> None:
        if isinstance(record, GpuSample):
            self._latest_gpu[(record.node, record.gpu)] = record
        elif isinstance(record, EngineSample):
            self._latest_engine_samples[(record.addr, record.metric, tuple(sorted(record.labels.items())))] = record
            if record.metric == "sglang_num_running_reqs":
                ts, total = self._latest_running_reqs.get(record.addr, (None, 0.0))
                self._latest_running_reqs[record.addr] = (
                    record.ts,
                    (total if ts == record.ts else 0.0) + record.value,
                )
        elif isinstance(record, PhaseEvent):
            self._latest_phase_seconds[record.name] = record.t1 - record.t0
            if record.role == "train":
                self._train_gpus.update((record.node, gpu) for gpu in record.gpus)

    def prometheus_snapshot(self) -> dict[str, float]:
        """Latest-value gauges for external Grafana. Keys avoid characters the
        prometheus collector's sanitizer does not handle ('.', ':')."""

        def safe(text: str) -> str:
            return text.replace("http://", "").replace(".", "_").replace(":", "_")

        snapshot = {
            f"dashboard/gpu_{safe(node)}_{gpu}_util": float(sample.util)
            for (node, gpu), sample in self._latest_gpu.items()
        }
        snapshot |= {
            f"dashboard/gpu_{safe(node)}_{gpu}_mem_mb": float(sample.mem_mb)
            for (node, gpu), sample in self._latest_gpu.items()
        }
        snapshot |= {
            f"dashboard/engine_{safe(addr)}_running_reqs": value
            for addr, (_, value) in self._latest_running_reqs.items()
        }
        snapshot |= {
            f"dashboard/phase_{name}_seconds": seconds for name, seconds in self._latest_phase_seconds.items()
        }
        return snapshot
