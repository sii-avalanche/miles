"""Latest fleet gauges for W&B, without a dashboard server, history store or event sinks."""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict

from miles.utils.fleet_sampling import GpuReader, read_engine_metrics

logger = logging.getLogger(__name__)
_ACTOR_NAME = "miles_wandb_fleet"
_handle = None
_primary = False


class FleetTelemetry:
    def __init__(self, *, train_gpus, inference_gpus, gpu_interval, engine_interval, scrape_mode):
        self.expected = {"train": train_gpus, "inference": inference_gpus}
        self.gpu_interval = gpu_interval
        self.engine_interval = engine_interval
        self.scrape_mode = scrape_mode
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._threads = []
        self._samplers = {}
        self._pending = {}
        self._gpu = {}
        self._train = set()
        self._engines = []
        self._router = None
        self._samples = {}
        self._previous = {}
        self._warned = {}

    def start(self):
        for name, callback, interval in (
            ("gpu", self._read_gpus, self.gpu_interval),
            ("engine", self._read_engines, self.engine_interval),
        ):
            thread = threading.Thread(target=self._loop, args=(name, callback, interval), daemon=True)
            self._threads.append(thread)
            thread.start()

    def _loop(self, name, callback, interval):
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                callback()
            except Exception:
                if started - self._warned.get(name, float("-inf")) >= 60:
                    logger.warning("Fleet %s sampling failed; leaving a gap", name, exc_info=True)
                    self._warned[name] = started
            self._stop.wait(max(0.1, interval - (time.monotonic() - started)))

    def register_train(self, node, gpus):
        with self._lock:
            self._train.update((node, int(gpu)) for gpu in gpus)

    def set_router(self, router):
        with self._lock:
            self._router = router

    def set_engines(self, engines):
        with self._lock:
            self._engines = engines
            active = {e["addr"] for e in engines}
            self._samples = {k: p for k, p in self._samples.items() if p.addr in active}
            self._previous = {k: p for k, p in self._previous.items() if p.addr in active}

    def _read_gpus(self):
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        alive = {
            n["NodeID"]: n["NodeManagerAddress"]
            for n in ray.nodes()
            if n.get("Alive") and n.get("Resources", {}).get("GPU", 0) > 0
        }
        for node_id in list(self._samplers):
            if node_id not in alive:
                self._pending.pop(node_id, None)
                ray.kill(self._samplers.pop(node_id))
        for node_id, node in alive.items():
            if self._stop.is_set():
                return
            if node_id not in self._samplers:
                self._samplers[node_id] = (
                    ray.remote(GpuReader)
                    .options(
                        num_cpus=0, scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
                    )
                    .remote()
                )
            pending = self._pending.get(node_id)
            if pending is not None:
                ready, _ = ray.wait([pending], timeout=0)
                if not ready:
                    continue
                self._pending.pop(node_id)
                try:
                    samples = ray.get(pending)
                except Exception:
                    ray.kill(self._samplers.pop(node_id))
                    continue
                with self._lock:
                    self._gpu.update({(node, gpu): sample for gpu, sample in samples.items()})
            self._pending[node_id] = self._samplers[node_id].read.remote()
        with self._lock:
            self._gpu = {key: value for key, value in self._gpu.items() if key[0] in alive.values()}

    def _read_engines(self):
        with self._lock:
            addrs = [e["addr"] for e in self._engines]
            router = self._router if self.scrape_mode == "router" else None
        samples = read_engine_metrics(addrs, router=router)
        with self._lock:
            active = {e["addr"] for e in self._engines}
            self._samples.update({(p.addr, p.labels, p.metric): p for p in samples if p.addr in active})

    def snapshot(self):
        now = time.time()
        with self._lock:
            train = set(self._train)
            inference = {(node, int(gpu)) for e in self._engines for node, gpu in e["gpus"]}
            gpu = dict(self._gpu)
            current = {k: p for k, p in self._samples.items() if 0 <= now - p.ts <= max(10, 3 * self.engine_interval)}
            previous, self._previous = self._previous, current
        result = {"available": 1}
        for role, mapped in (("train", train), ("inference", inference)):
            fresh = [gpu[k] for k in mapped if k in gpu and 0 <= now - gpu[k][0] <= max(10, 3 * self.gpu_interval)]
            result.update(_gpu_metrics(role, mapped, fresh, expected=self.expected[role]))
        result.update(_engine_metrics(current, previous))
        return result

    def shutdown(self):
        import ray

        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=5)
        # The sampling loop checks stop before sleeping; no training actor is affected.
        for handle in list(self._samplers.values()):
            try:
                ray.kill(handle)
            except Exception:
                logger.debug("Fleet reader already stopped", exc_info=True)
        self._samplers.clear()


def _gpu_metrics(role, mapped, fresh, *, expected):
    expected = max(expected, len(mapped))
    result = {"mapped_gpus": len(mapped), "sampled_gpus": len(fresh), "expected_gpus": expected}
    if expected:
        result["coverage_fraction"] = len(fresh) / expected
    if fresh:
        util = [p[1] for p in fresh]
        result.update(
            gpu_util_mean_pct=sum(util) / len(util),
            gpu_util_min_pct=min(util),
            gpu_util_max_pct=max(util),
            gpu_memory_max_gb=max(p[2] for p in fresh) / 1024,
            gpu_active_fraction=sum(x >= 10 for x in util) / len(util),
            gpu_power_total_w=sum(p[3] for p in fresh),
        )
    return {f"{role}/{k}": v for k, v in result.items()}


def _engine_metrics(current, previous):
    gauges = {
        "sglang_num_running_reqs": ("running_requests", False),
        "sglang_num_queue_reqs": ("queued_requests", False),
        "sglang_gen_throughput": ("generated_tokens_per_s", False),
        "sglang_token_usage": ("kv_usage_mean", True),
        "sglang_mamba_usage": ("mamba_usage_mean", True),
        "sglang_cache_hit_rate": ("cache_hit_rate_mean", True),
    }
    result = {}
    families = defaultdict(list)
    for p in current.values():
        families[p.metric].append(p)
    for metric, (name, mean) in gauges.items():
        points = families.get(metric, [])
        result[f"{name}_reporting_engines"] = len({p.addr for p in points})
        if points:
            result[name] = sum(p.value for p in points) / (len(points) if mean else 1)
            if mean:
                result[name.removesuffix("_mean") + "_max"] = max(p.value for p in points)
    # Expose every sampled family, including counters and PD queues previously discarded.
    for metric, points in families.items():
        mean = gauges.get(metric, (None, False))[1]
        result[f"sglang/{metric}"] = sum(p.value for p in points) / (len(points) if mean else 1)
    changes = {
        k: (p.value - old.value, p.ts - old.ts)
        for k, p in current.items()
        if (old := previous.get(k)) is not None and p.ts > old.ts and p.value >= old.value
    }
    for stem in ("time_to_first_token", "inter_token_latency", "time_per_output_token", "e2e_request_latency"):
        pairs = [
            (delta, changes[(addr, labels, f"sglang_{stem}_seconds_count")][0])
            for (addr, labels, metric), (delta, _) in changes.items()
            if metric == f"sglang_{stem}_seconds_sum" and (addr, labels, f"sglang_{stem}_seconds_count") in changes
        ]
        if (count := sum(n for _, n in pairs)) > 0:
            result[f"{stem}_seconds"] = sum(value for value, _ in pairs) / count
    for metric in families:
        if metric.endswith("_total"):
            rates = [delta / elapsed for (_, _, name), (delta, elapsed) in changes.items() if name == metric]
            if rates:
                result[metric.removeprefix("sglang_").removesuffix("_total") + "_per_s"] = sum(rates)
    return {f"inference/{k}": v for k, v in result.items()}


def _enabled(args):
    return (
        getattr(args, "use_wandb", False)
        and getattr(args, "perf_monitor_interval", 0) > 0
        and getattr(args, "cluster_backend", "ray") == "ray"
    )


def _resolve():
    global _handle
    if _handle is None:
        import ray

        try:
            _handle = ray.get_actor(_ACTOR_NAME)
        except Exception:
            logger.warning("W&B fleet telemetry unavailable; training continues", exc_info=True)
    return _handle


def init_fleet(args, *, primary):
    global _handle, _primary
    if not _enabled(args):
        return
    if not primary:
        _resolve()
        return
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    _handle = (
        ray.remote(FleetTelemetry)
        .options(
            name=_ACTOR_NAME,
            num_cpus=0,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(), soft=False
            ),
        )
        .remote(
            train_gpus=(getattr(args, "actor_num_nodes", 0) or 0) * (getattr(args, "actor_num_gpus_per_node", 0) or 0),
            inference_gpus=getattr(args, "rollout_num_gpus", 0) or 0,
            gpu_interval=args.perf_monitor_gpu_interval,
            engine_interval=args.perf_monitor_engine_interval,
            scrape_mode=args.perf_monitor_engine_mode,
        )
    )
    _primary = True
    ray.get(_handle.start.remote(), timeout=30)


def register_train_actor(args):
    if not _enabled(args) or (handle := _resolve()) is None:
        return
    import ray
    from miles.utils.misc import get_current_node_ip

    try:
        handle.register_train.remote(get_current_node_ip(), ray.get_gpu_ids())
    except Exception:
        logger.warning("Train GPU mapping unavailable; training continues", exc_info=True)


def register_router(args):
    if not _enabled(args) or (handle := _resolve()) is None:
        return
    assert args.sglang_router_ip is not None, "register_router must run after start_rollout_servers"
    try:
        handle.set_router.remote(f"http://{args.sglang_router_ip}:{args.sglang_router_port}")
    except Exception:
        logger.warning("Engine metrics router unavailable; training continues", exc_info=True)


async def register_engines(servers, *, provider):
    if _handle is None:
        return
    try:
        _register_engines(servers, provider=provider)
    except Exception:
        logger.warning("Engine GPU mapping unavailable; training continues", exc_info=True)


def _register_engines(servers, *, provider):
    cells = [
        cell
        for server in servers.values()
        for cell in server.server_cells.values()
        if cell.is_pending_weights_or_serving
    ]
    infos = provider.get_worker_infos(cell_ids=[cell.meta.cell_id for cell in cells])
    engines = [
        {
            "addr": cell.server_url,
            "gpus": [(info.self_addrs["primary"].host.strip("[]"), gpu) for info in workers for gpu in info.gpu_ids],
        }
        for cell, workers in zip(cells, infos, strict=True)
    ]
    _handle.set_engines.remote(engines)


def fleet_snapshot():
    if _handle is None:
        return {"available": 0}
    import ray

    try:
        return ray.get(_handle.snapshot.remote(), timeout=2)
    except Exception:
        logger.debug("Fleet snapshot unavailable", exc_info=True)
        return {"available": 0}


def finish_fleet():
    global _handle, _primary
    if _handle is not None and _primary:
        import ray

        try:
            ray.get(_handle.shutdown.remote(), timeout=15)
        except Exception:
            logger.warning("Fleet telemetry shutdown incomplete", exc_info=True)
        finally:
            try:
                ray.kill(_handle)
            except Exception:
                logger.debug("Fleet sampler already stopped", exc_info=True)
    _handle = None
    _primary = False
