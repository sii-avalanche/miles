import logging
import time

import torch

from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import MetricEvent
from miles.utils.perf_monitor import start_monitor, stop_monitors, wandb_runtime_enabled

from .base import (
    MlflowBackend,
    PrometheusBackend,
    TensorboardBackend,
    TrackingBackend,
    TrackingManager,
    WandbBackend,
)
from .ci_history import CiHistoryBackend

# The full registry lives here, not base.py: base must never import a backend
# module (ci_history imports TrackingBackend from base -> circular). This
# module is the tracking entry point and the one place that imports every
# backend, so it owns the registry.
BACKEND_REGISTRY: dict[str, tuple[type[TrackingBackend], str]] = {
    "wandb": (WandbBackend, "use_wandb"),
    "tensorboard": (TensorboardBackend, "use_tensorboard"),
    "mlflow": (MlflowBackend, "use_mlflow"),
    "prometheus": (PrometheusBackend, "use_prometheus"),
    "ci_history": (CiHistoryBackend, "ci_enable_metrics_capture"),
}

logger = logging.getLogger(__name__)
_manager = TrackingManager(BACKEND_REGISTRY)


def init_tracking(args, primary: bool = True, **kwargs):
    if wandb_runtime_enabled(args):
        from miles.utils.fleet_monitor import init_fleet

        try:
            init_fleet(args, primary=primary)
        except Exception:
            logger.warning("W&B fleet sampling unavailable; scalar tracking continues", exc_info=True)
    if primary and getattr(args, "perf_monitor_interval", 0) > 0:
        # One origin is serialized with args to every worker; wall-clock series
        # remain aligned even when rollout and trainer start minutes apart.
        args.perf_monitor_start_ts = time.time()
    _manager.init(args, primary=primary, **kwargs)
    start_monitor(args, primary=primary, router_addr=kwargs.get("router_addr"))


def define_step_key_metric_group(prefix: str, step_key: str) -> None:
    """Declare a metric group plotted against its own step key (e.g. ``{name}/*`` vs ``{name}/step``).
    Only wandb acts on this; must be called from the primary tracking process or definitions may be lost."""
    _manager.define_step_key_metric_group(prefix, step_key)


def log(args, metrics, step_key: str):
    step = metrics.get(step_key)
    _manager.log(metrics, step=step, step_key=step_key)

    if not getattr(args, "use_wandb", False) and is_event_logger_initialized():
        serializable_metrics = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in metrics.items()}
        get_event_logger().log(MetricEvent, {"metrics": serializable_metrics}, print_log=False)


def finish_tracking():
    stop_monitors()
    _manager.finish()
    from miles.utils.fleet_monitor import finish_fleet

    finish_fleet()
