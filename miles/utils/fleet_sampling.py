"""Device and engine reads for fleet metrics; no history store or dashboard events."""

from __future__ import annotations

import importlib
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class _NvmlProvider:
    name = "NVML"
    module = "pynvml"

    def __init__(self, api):
        self._api = api

    def initialize(self) -> list:
        self._api.nvmlInit()
        count = self._api.nvmlDeviceGetCount()
        if count == 0:
            raise RuntimeError("no NVML devices")
        return [self._api.nvmlDeviceGetHandleByIndex(i) for i in range(count)]

    def read_device(self, handle) -> tuple[int, int, int]:
        util = int(self._api.nvmlDeviceGetUtilizationRates(handle).gpu)
        mem_mb = int(self._api.nvmlDeviceGetMemoryInfo(handle).used) >> 20
        power_w = int(self._api.nvmlDeviceGetPowerUsage(handle)) // 1000
        return (util, mem_mb, power_w)


class _AmdSmiProvider:
    name = "AMD SMI"
    module = "amdsmi"

    def __init__(self, api):
        self._api = api

    def initialize(self) -> list:
        self._api.amdsmi_init()
        handles = self._api.amdsmi_get_processor_handles()
        if not handles:
            raise RuntimeError("no AMD SMI devices")
        return list(handles)

    def read_device(self, handle) -> tuple[int, int, int]:
        util = int(self._api.amdsmi_get_gpu_activity(handle)["gfx_activity"])
        mem_mb = int(self._api.amdsmi_get_gpu_vram_usage(handle)["vram_used"])
        try:
            power_w = _amd_socket_power(self._api.amdsmi_get_power_info(handle))
        except Exception:
            power_w = 0
        return (util, mem_mb, power_w)


def _amd_socket_power(power: dict) -> int:
    for field in ("socket_power", "current_socket_power", "average_socket_power"):
        value = power.get(field, "N/A")
        if value != "N/A" and 0 <= value < 65535:
            return int(value)
    raise ValueError(f"AMD SMI socket power unavailable: {power!r}")


class GpuReader:
    """One lightweight reader per Ray GPU node; no CUDA context or sample buffers."""

    def __init__(self):
        self._provider = None
        self._handles = []
        for cls in (_NvmlProvider, _AmdSmiProvider):
            try:
                provider = cls(importlib.import_module(cls.module))
                self._handles = provider.initialize()
                self._provider = provider
                break
            except Exception:
                logger.debug("%s unavailable", cls.name, exc_info=True)
        if self._provider is None:
            logger.warning("GPU telemetry unavailable: neither NVML nor AMD SMI initialized")

    def read(self):
        result = {}
        stamp = time.time()
        for gpu, handle in enumerate(self._handles):
            try:
                util, memory, power = self._provider.read_device(handle)
                result[gpu] = (stamp, util, memory, power)
            except Exception:
                logger.debug("GPU %s sample unavailable", gpu, exc_info=True)
        return result


@dataclass(frozen=True)
class EngineSample:
    ts: float
    addr: str
    metric: str
    labels: tuple[tuple[str, str], ...]
    value: float


def parse_engine_metrics(text: str, *, addr: str | None, stamp: float):
    from prometheus_client.parser import text_string_to_metric_families

    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            name = sample.name.replace(":", "_")
            if name not in ENGINE_METRICS or not math.isfinite(sample.value):
                continue
            origin = addr or sample.labels.get("worker_addr")
            if origin:
                labels = tuple(sorted((k, v) for k, v in sample.labels.items() if k in ("dp_rank", "engine_type")))
                yield EngineSample(stamp, origin, name, labels, float(sample.value))


def read_engine_metrics(addrs, *, router=None):
    import requests

    def fetch(addr):
        try:
            response = requests.get(f"{addr}/engine_metrics" if router else f"{addr}/metrics", timeout=2)
            response.raise_for_status()
            return list(parse_engine_metrics(response.text, addr=None if router else addr, stamp=time.time()))
        except Exception:
            logger.debug("Engine metrics unavailable at %s", addr, exc_info=True)
            return []

    targets = [router] if router else list(addrs)
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(targets)))) as pool:
        return [sample for samples in pool.map(fetch, targets) for sample in samples]


ENGINE_METRICS = (
    "sglang_num_running_reqs",
    "sglang_num_queue_reqs",
    "sglang_gen_throughput",
    "sglang_token_usage",
    "sglang_mamba_usage",
    "sglang_cache_hit_rate",
    "sglang_num_prefill_prealloc_queue_reqs",
    "sglang_num_prefill_inflight_queue_reqs",
    "sglang_num_decode_prealloc_queue_reqs",
    "sglang_num_decode_transfer_queue_reqs",
    "sglang_kv_transfer_speed_gb_s",
    "sglang_kv_transfer_latency_ms",
    "sglang_prompt_tokens_total",
    "sglang_generation_tokens_total",
    "sglang_num_requests_total",
    "sglang_num_aborted_requests_total",
    "sglang_time_to_first_token_seconds_sum",
    "sglang_time_to_first_token_seconds_count",
    "sglang_inter_token_latency_seconds_sum",
    "sglang_inter_token_latency_seconds_count",
    "sglang_time_per_output_token_seconds_sum",
    "sglang_time_per_output_token_seconds_count",
    "sglang_e2e_request_latency_seconds_sum",
    "sglang_e2e_request_latency_seconds_count",
)
