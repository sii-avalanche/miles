"""Strict DModel checkpoints; failed restores require abandoning the actor.

The actor-facing API matches fsdp_utils.checkpoint. Only checkpoints written by
this adapter, with the same world size, are accepted. RNG sidecars are trusted
local checkpoint files (they contain Python and NumPy RNG state).
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict

from miles.backends.training_utils.checkpoint.tracker import CHECKPOINT_TRACKER_FILENAME
from miles.utils.distributed_utils import get_gloo_group

_T = TypeVar("_T")
_FORMAT = "miles-dmodel-v1"


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _control_group() -> dist.ProcessGroup | None:
    if not dist.is_initialized():
        return None
    group = get_gloo_group()
    if dist.get_backend(group) != "gloo" or dist.get_process_group_ranks(group) != list(range(_world_size())):
        raise ValueError("DModel checkpoint requires a Gloo control group matching the full world in rank order")
    return group


def _all_ranks(operation: Callable[[], _T]) -> _T:
    """Propagate local I/O/preparation failures before the next collective."""
    group = _control_group()
    result, error = None, None
    try:
        result = operation()
    except Exception as exc:
        error = exc
    if dist.is_initialized():
        errors = [None] * _world_size()
        dist.all_gather_object(errors, None if error is None else f"{type(error).__name__}: {error}", group=group)
        if error is None and any(message is not None for message in errors):
            raise RuntimeError(f"DModel checkpoint failed on another rank: {errors}")
    if error is not None:
        raise error
    return result


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _optimizer_state(model, optimizer, expected_keys: set[str] | None = None) -> dict:
    # DModel is an optional backend with heavyweight package initialization.
    from dmodel.training.optimizer.state import (
        _optimizer_leaves,
        get_flattened_optimizer_state_dict,
        materialize_optimizer_state,
    )

    supported = (torch.optim.Adam, torch.optim.AdamW, torch.optim.Muon)
    if optimizer is None or any(not isinstance(leaf, supported) for leaf in _optimizer_leaves(optimizer)):
        raise TypeError("DModel checkpoint requires native Adam/AdamW/Muon optimizer leaves")
    # Otherwise PyTorch's state-dict API may call optimizer.step() to create state.
    materialize_optimizer_state(optimizer)
    state = get_flattened_optimizer_state_dict(model, optimizer)
    if expected_keys is not None and set(state) != expected_keys:
        raise ValueError("Checkpoint optimizer layout does not match the current native optimizer")
    return state


def _capture_rng() -> dict:
    state = {"torch": torch.get_rng_state(), "python": random.getstate(), "numpy": np.random.get_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return {"rank": _rank(), "world_size": _world_size(), "state": state}


def _restore_rng(payload: dict) -> None:
    state = payload["state"]
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _read_rng(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["rank"] != _rank() or payload["world_size"] != _world_size():
        raise ValueError(f"RNG rank/world-size mismatch in {path}")
    state = payload["state"]
    # Validate with independent generators, without changing the actor's RNG.
    torch.Generator().set_state(state["torch"])
    random.Random().setstate(state["python"])
    np.random.RandomState().set_state(state["numpy"])
    if torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError(f"CUDA RNG device-count mismatch in {path}")
    elif "cuda" in state:
        raise ValueError("CUDA RNG checkpoint requires CUDA, or --no-load-rng")
    return payload


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"Invalid checkpoint {name}: {value!r}")
    return value


def _scheduler_type(scheduler) -> str | None:
    if scheduler is None:
        return None
    cls = type(scheduler)
    return f"{cls.__module__}.{cls.__qualname__}"


def _checked_state(name: str, state: dict, expected_keys: list[str], checkpoint_path: Path | None = None) -> dict:
    wrapped = {name: state}
    # Use the same native flattening as DCP save, including nested scheduler state.
    planner = dcp.DefaultSavePlanner()
    planner.set_up_planner(wrapped)
    if sorted(planner.state_dict) != expected_keys:
        raise ValueError(f"Checkpoint {name} layout does not match the current state inventory")
    if checkpoint_path is not None:
        stored = dcp.FileSystemReader(checkpoint_path).read_metadata().state_dict_metadata
        if sorted(stored) != expected_keys:
            raise ValueError(f"Checkpoint {name} stored keys differ from committed inventory")
        for item in planner.create_local_plan().items:
            current, saved = item.tensor_data, stored[item.index.fqn]
            if (current is not None) != isinstance(saved, TensorStorageMetadata):
                raise ValueError(f"Checkpoint {name} tensor kind differs for {item.index.fqn}")
            # Native planner sizes are global, not the rank-local DTensor shard sizes.
            if current is not None and (current.size != saved.size or current.properties.dtype != saved.properties.dtype):
                raise ValueError(
                    f"Checkpoint {name} tensor global size/dtype mismatch for {item.index.fqn}: "
                    f"saved {tuple(saved.size)}/{saved.properties.dtype}, "
                    f"current {tuple(current.size)}/{current.properties.dtype}"
                )
    return wrapped


def _preflight(actor) -> tuple[Path, dict, dict | None]:
    root = Path(actor.args.load).expanduser()
    step = getattr(actor.args, "ckpt_step", None)
    if step is None:
        text = (root / CHECKPOINT_TRACKER_FILENAME).read_text().strip()
        if not text.isdigit():
            raise ValueError(f"Invalid checkpoint tracker in {root}")
        step = int(text)
    step = _nonnegative_int(step, "iteration")
    directory = root / f"iter_{step:07d}"
    metadata = json.loads((directory / "meta.json").read_text())
    if metadata["format"] != _FORMAT:
        raise ValueError("Unsupported DModel checkpoint format; legacy partial resumes are not accepted")
    if metadata["world_size"] != _world_size():
        raise ValueError("DModel checkpoint resume requires the same world size")
    for name in ("iteration", "next_rollout_id", "global_step", "micro_step"):
        _nonnegative_int(metadata[name], name)
    if metadata["iteration"] != step or metadata["next_rollout_id"] != step:
        raise ValueError("Checkpoint iteration/cursor mismatch")

    required = ["model"]
    if not getattr(actor.args, "no_load_optim", False):
        required.append("optimizer")
        if actor.lr_scheduler is not None:
            required.append("lr_scheduler")
    for name in required:
        if metadata["components"].get(name) is not True:
            raise ValueError(f"Checkpoint is missing required {name} state")

    if not getattr(actor.args, "no_load_optim", False):
        has_scheduler = actor.lr_scheduler is not None
        if metadata["components"].get("lr_scheduler") is not has_scheduler:
            raise ValueError("Checkpoint lr_scheduler presence does not match the current scheduler")
        if has_scheduler and metadata.get("lr_scheduler_type") != _scheduler_type(actor.lr_scheduler):
            raise ValueError("Checkpoint lr_scheduler type does not match the current scheduler")

    rng = None
    if not getattr(actor.args, "no_load_rng", False):
        expected_files = [f"rng_rank_{rank:05d}.pt" for rank in range(_world_size())]
        if metadata["rng_files"] != expected_files:
            raise ValueError("Checkpoint is missing required per-rank RNG state; use --no-load-rng to opt out")
        rng = _read_rng(directory / expected_files[_rank()])
    return directory, metadata, rng


def load(actor: Any) -> dict[str, Any] | None:
    """Restore required state or raise; never accept a partial-resume cursor.

    ``args.load is None`` is the only optional no-resume case. On any exception,
    initialization must abort: tensors may have been modified before the error.
    """
    if getattr(actor.args, "load", None) is None:
        if getattr(actor.args, "ckpt_step", None) is not None:
            raise ValueError("--ckpt-step requires --load")
        return None
    group = _control_group()
    directory, metadata, rng = _all_ranks(lambda: _preflight(actor))
    model_state = _all_ranks(
        lambda: _checked_state("model", get_model_state_dict(actor.model), metadata["keys"]["model"], directory / "model")
    )
    optimizer_state, scheduler_state = None, None

    if not getattr(actor.args, "no_load_optim", False):
        from dmodel.training.optimizer.state import set_flattened_optimizer_state_dict

        # An exact inventory prevents absent moments from silently retaining zeros.
        expected_keys = {key.removeprefix("optimizer.") for key in metadata["keys"]["optimizer"]}
        optimizer_state = _all_ranks(
            lambda: _checked_state(
                "optimizer", _optimizer_state(actor.model, actor.optimizer, expected_keys),
                metadata["keys"]["optimizer"], directory / "optimizer",
            )
        )
        if actor.lr_scheduler is not None:
            scheduler_state = _all_ranks(
                lambda: _checked_state(
                    "lr_scheduler", actor.lr_scheduler.state_dict(), metadata["keys"]["lr_scheduler"], directory / "lr_scheduler"
                )
            )

    dcp.load(model_state, checkpoint_id=directory / "model", process_group=group)
    _all_ranks(lambda: set_model_state_dict(actor.model, model_state["model"]))
    if optimizer_state is not None:
        dcp.load(optimizer_state, checkpoint_id=directory / "optimizer", process_group=group)
        _all_ranks(lambda: set_flattened_optimizer_state_dict(actor.model, actor.optimizer, optimizer_state["optimizer"]))
    if scheduler_state is not None:
        dcp.load(scheduler_state, checkpoint_id=directory / "lr_scheduler", process_group=group)
        _all_ranks(lambda: actor.lr_scheduler.load_state_dict(scheduler_state["lr_scheduler"]))
    return {"metadata": metadata, "rng": rng}


def finalize_load(actor: Any, checkpoint_payload: dict[str, Any] | None) -> None:
    """Commit RNG/counters/cursor only after a successful ``load`` on all ranks."""
    if checkpoint_payload is None:
        return
    if checkpoint_payload["rng"] is not None:
        _all_ranks(lambda: _restore_rng(checkpoint_payload["rng"]))
    _synchronize()
    if dist.is_initialized():
        dist.barrier(group=_control_group())
    metadata = checkpoint_payload["metadata"]
    actor.global_step = metadata["global_step"]
    actor.micro_step = metadata["micro_step"]
    actor.args.start_rollout_id = metadata["next_rollout_id"]


def save(actor: Any, iteration: int) -> None:
    """Write a new immutable iteration, then atomically publish its tracker last."""
    group = _control_group()
    _synchronize()
    step = _nonnegative_int(iteration, "rollout_id") + 1
    root = Path(actor.args.save).expanduser()
    directory = root / f"iter_{step:07d}"
    save_optimizer = not getattr(actor.args, "no_save_optim", False)
    save_scheduler = save_optimizer and actor.lr_scheduler is not None
    save_rng = not getattr(actor.args, "no_save_rng", False)
    metadata = {
        "format": _FORMAT,
        "iteration": step,
        "next_rollout_id": step,
        "global_step": _nonnegative_int(actor.global_step, "global_step"),
        "micro_step": _nonnegative_int(actor.micro_step, "micro_step"),
        "world_size": _world_size(),
        "components": {"model": True, "optimizer": save_optimizer, "lr_scheduler": save_scheduler},
        "lr_scheduler_type": _scheduler_type(actor.lr_scheduler) if save_scheduler else None,
        "rng_files": [f"rng_rank_{rank:05d}.pt" for rank in range(_world_size())] if save_rng else [],
        "keys": {},
    }

    def prepare_directory() -> None:
        if _rank() == 0:
            root.mkdir(parents=True, exist_ok=True)
            directory.mkdir()  # Never overwrite a committed or incomplete iteration.

    _all_ranks(prepare_directory)
    state = {"model": _all_ranks(lambda: get_model_state_dict(actor.model))}
    saved = dcp.save(state, checkpoint_id=directory / "model", process_group=group)
    metadata["keys"]["model"] = sorted(saved.state_dict_metadata)
    if save_optimizer:
        state = {"optimizer": _all_ranks(lambda: _optimizer_state(actor.model, actor.optimizer))}
        saved = dcp.save(state, checkpoint_id=directory / "optimizer", process_group=group)
        metadata["keys"]["optimizer"] = sorted(saved.state_dict_metadata)
    if save_scheduler:
        state = {"lr_scheduler": _all_ranks(lambda: actor.lr_scheduler.state_dict())}
        saved = dcp.save(state, checkpoint_id=directory / "lr_scheduler", process_group=group)
        metadata["keys"]["lr_scheduler"] = sorted(saved.state_dict_metadata)

    def save_local_rng() -> None:
        if save_rng:
            path = directory / metadata["rng_files"][_rank()]
            temporary = path.with_suffix(".tmp")
            torch.save(_capture_rng(), temporary)
            temporary.replace(path)

    _all_ranks(save_local_rng)

    def publish() -> None:
        if _rank() == 0:
            _atomic_write(directory / "meta.json", json.dumps(metadata, sort_keys=True, indent=2))
            _atomic_write(root / CHECKPOINT_TRACKER_FILENAME, str(step))

    _all_ranks(publish)
