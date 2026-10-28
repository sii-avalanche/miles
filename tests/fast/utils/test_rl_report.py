"""No GPU, Ray cluster, W&B server or torch needed."""

import json
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from miles.utils.perf_monitor import RuntimeMonitor
from miles.utils.rl_report import render_report, report_metrics, report_payload
from miles.utils.fleet_monitor import FleetTelemetry


class ReportTest(unittest.TestCase):
    def test_html_is_self_contained_and_escapes_script_termination(self):
        events = [
            {
                "kind": "phase",
                "role": "driver",
                "name": "train",
                "t0": 1,
                "t1": 2,
                "status": "completed",
                "rollout_id": 0,
                "details": {"label": "</script><script>alert(1)</script>"},
            }
        ]
        result = render_report(events, title="<script>oops</script>")
        self.assertIn("&lt;script&gt;oops", result)
        self.assertNotIn('src="http', result)
        self.assertNotIn("</script><script>", result)
        self.assertIn("canvas.onwheel", result)
        self.assertIn("Rollout, data pool, training and weight sync", result)

    def test_filter_drops_unconsumed_diagnostics(self):
        values = {
            "runtime/time_s": 1,
            "runtime/driver/fleet/engines/inference/e/running_requests": 3,
            "runtime/rollout/record/end_to_end/p95_seconds": 4,
            "runtime/rollout/buffer/versions/lag_0_groups": 5,
            "secret": 42,
            "runtime/driver/fleet/train/gpu_util_mean_pct": float("nan"),
        }
        clean = report_metrics(values)
        self.assertEqual(len(clean), 3)
        payload = report_payload([{"kind": "metrics", "ts": 10, "metrics": values}], 10)
        json.dumps(payload, allow_nan=False)
        self.assertEqual(payload["series"]["runtime/time_s"]["ts"], [10])

    def test_wandb_mode_retains_short_phase_events(self):
        m = RuntimeMonitor(args=SimpleNamespace(perf_monitor_interval=10, use_wandb=True), role="driver")
        with m.phase("update_weights", rollout_id=7):
            pass
        self.assertEqual(len(m._events), 2)
        self.assertIsNone(m._events[0]["t1"])
        self.assertGreaterEqual(m._events[1]["t1"], m._events[1]["t0"])

    def test_wandb_drops_runtime_scalars_but_keeps_alerts(self):
        from miles.utils.tracking_utils.base import WandbBackend

        backend = WandbBackend()
        fake = Mock()
        with patch.dict("sys.modules", {"wandb": fake}):
            backend.log({"train/loss": 0.5, "runtime/time_s": 10, "runtime/driver/alerts/collector_unavailable": 1})
            fake.log.assert_called_once_with({"train/loss": 0.5})
            fake.alert.assert_called_once()
            backend.log({"runtime/time_s": 20})
            fake.log.assert_called_once()
            backend.define_step_key_metric_group("runtime", "runtime/time_s")
            fake.define_metric.assert_not_called()

    def test_fleet_collects_multiple_roles_and_drains_once(self):
        f = FleetTelemetry(train_gpus=0, inference_gpus=0, gpu_interval=1, engine_interval=2, scrape_mode="direct")
        self.assertEqual(f.report_records([{"role": "rollout"}]), [])
        self.assertEqual(f.report_records([{"role": "driver"}], drain=True), [{"role": "rollout"}, {"role": "driver"}])
        self.assertEqual(f.report_records([], drain=True), [])


if __name__ == "__main__":
    unittest.main()
