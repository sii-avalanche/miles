"""CPU-only coverage of engine identity, missing GPU samples, and phase progress."""

import time
import unittest
from argparse import Namespace
from unittest.mock import patch

from miles.utils.fleet_monitor import FleetTelemetry
from miles.utils.fleet_sampling import EngineSample
from miles.utils.perf_monitor import RuntimeMonitor


class FleetDashboardTest(unittest.TestCase):
    def setUp(self):
        self.fleet = FleetTelemetry(
            train_gpus=2, inference_gpus=1, gpu_interval=1, engine_interval=2, scrape_mode="direct"
        )
        self.now = 100.0

    def test_engines_keep_their_own_gpus_and_request_rates(self):
        fleet = self.fleet
        fleet.register_train("host", [0], engine_id="actor-0", worker_id=0)
        fleet.register_train("host", [1], engine_id="actor-0", worker_id=1)
        fleet.set_engines([{"id": "rollout-0", "addr": "http://engine", "gpus": [("host", 2)]}])
        fleet._gpu = {("host", i): (100, util, 2048, 100) for i, util in enumerate((20, 80, 90))}
        point = EngineSample(100, "http://engine", "sglang_num_requests_total", (), 12)
        key = (point.addr, point.labels, point.metric)
        fleet._samples[key] = point
        fleet._previous[key] = EngineSample(98, point.addr, point.metric, (), 8)
        with patch("miles.utils.fleet_monitor.time.time", return_value=self.now):
            metrics = fleet.snapshot()
        self.assertEqual(metrics["train/gpu_util_mean_pct"], 50)
        self.assertEqual(metrics["engines/train/actor-0/gpu_util_mean_pct"], 50)
        self.assertEqual(metrics["engines/train/actor-0/devices/host/gpu_1/gpu_util_pct"], 80)
        self.assertEqual(metrics["engines/inference/rollout-0/gpu_util_mean_pct"], 90)
        self.assertEqual(metrics["engines/inference/rollout-0/num_requests_per_s"], 2)
        self.assertEqual(metrics["inference/num_requests_per_s"], 2)

    def test_missing_and_expired_samples_are_not_idle_gpus(self):
        self.fleet.register_train("host", [0, 1], engine_id="actor-0", worker_id=0)
        self.fleet._gpu[("host", 0)] = (80, 95, 2048, 100)
        with patch("miles.utils.fleet_monitor.time.time", return_value=self.now):
            metrics = self.fleet.snapshot()
        prefix = "engines/train/actor-0/"
        self.assertEqual(metrics[prefix + "coverage_fraction"], 0)
        self.assertNotIn(prefix + "gpu_util_mean_pct", metrics)
        self.assertEqual(metrics[prefix + "devices/host/gpu_0/available"], 0)
        self.assertEqual(metrics[prefix + "devices/host/gpu_0/sample_age_seconds"], 20)
        self.assertEqual(metrics[prefix + "devices/host/gpu_1/available"], 0)

    def test_worker_restart_replaces_old_assignment(self):
        self.fleet.register_train("old", [0], engine_id="actor-0", worker_id=0)
        self.fleet.register_train("new", [1], engine_id="actor-0", worker_id=0)
        with patch("miles.utils.fleet_monitor.time.time", return_value=self.now):
            metrics = self.fleet.snapshot()
        self.assertEqual(metrics["engines/train/actor-0/mapped_gpus"], 1)
        self.assertFalse(any("/devices/old/" in key for key in metrics))

    def test_removed_engine_does_not_remain_in_snapshot(self):
        self.fleet.set_engines([{"id": "rollout-0", "addr": "http://engine", "gpus": []}])
        self.fleet.set_engines([])
        self.assertFalse(any(key.startswith("engines/inference/") for key in self.fleet.snapshot()))


class ProgressDashboardTest(unittest.TestCase):
    def test_phase_concurrency_failure_and_resumed_progress(self):
        args = Namespace(perf_monitor_interval=10, use_wandb=True, num_rollout=10, start_rollout_id=4)
        monitor = RuntimeMonitor(args=args, role="driver")
        self.assertEqual(monitor.snapshot()["runtime/driver/phase/update_weights/waiting"], 0)
        self.assertEqual(monitor.snapshot()["runtime/driver/progress/fraction"], 0.4)
        with monitor.phase("train"):
            with monitor.phase("wait_rollout"):
                metrics = monitor.snapshot()
                self.assertEqual(metrics["runtime/driver/phase/train/waiting"], 1)
                self.assertEqual(metrics["runtime/driver/phase/wait_rollout/waiting"], 1)
        with self.assertRaises(ValueError):
            with monitor.phase("update_weights"):
                raise ValueError("sync failed")
        monitor.completed_step(4, started=time.time())
        metrics = monitor.snapshot()
        self.assertEqual(metrics["runtime/driver/progress/fraction"], 0.5)
        self.assertEqual(metrics["runtime/driver/phase/update_weights/waiting"], 0)
        self.assertEqual(metrics["runtime/driver/phase/update_weights/failed_total"], 1)


if __name__ == "__main__":
    unittest.main()
