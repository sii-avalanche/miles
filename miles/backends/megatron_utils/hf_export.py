"""Backend selection and checkpoint writing for Megatron HF exports."""

import json
import logging
from collections.abc import Sequence
from contextlib import ExitStack
from functools import cache
from pathlib import Path

import torch
from megatron.core.distributed import DistributedDataParallel as DDP
from safetensors import safe_open
from safetensors.torch import save_file

from miles.backends.megatron_utils.lora.utils import is_lora_model
from miles.backends.megatron_utils.named_weights import named_params_and_buffers
from miles.backends.training_utils.checkpoint.io import write_checkpoint_dir
from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.weight_update.snapshot_publisher import SnapshotPublisher
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.hf_utils.config import HF_EXPORT_COMPLETE_MARKER
from miles.utils.megatron_bridge_utils import patch_megatron_model

logger = logging.getLogger(__name__)


@cache
def _get_hf_bridge(hf_checkpoint: str):
    # Local: megatron.bridge is only needed on the bridge export path.
    from megatron.bridge import AutoBridge

    return AutoBridge.from_hf_pretrained(hf_checkpoint, trust_remote_code=True)


def save_hf_model(
    args,
    rollout_id: int,
    model: Sequence[DDP],
    *,
    publisher: SnapshotPublisher,
    path: str | Path | None = None,
    raise_on_error: bool = False,
) -> None:
    """Collectively write an HF model, with an additional HF adapter for LoRA.

    Writes a ``.complete`` marker after all ranks finish. Export errors are logged
    unless ``raise_on_error`` is set.
    """
    should_log = get_parallel_state().effective_dp_cp.rank == 0 and get_parallel_state().tp.rank == 0
    path = Path(path if path is not None else args.save_hf.format(rollout_id=rollout_id))

    def write_weights(checkpoint_dir: Path):
        if args.megatron_to_hf_mode == "raw" and not is_lora_model(model):
            # LoRA needs Bridge to merge the adapter into the base weights
            publisher.write_model(
                checkpoint_dir,
                weights=dict(named_params_and_buffers(args, model, convert_to_global_name=True)),
                hf_checkpoint=args.hf_checkpoint,
            )
            if torch.distributed.get_rank() == 0:
                add_uncovered_source_tensors(checkpoint_dir, Path(args.hf_checkpoint))
        else:
            bridge = _get_hf_bridge(args.hf_checkpoint)
            with patch_megatron_model(model):
                bridge.save_hf_pretrained(model, path=checkpoint_dir)
            torch.distributed.barrier(group=get_gloo_group())
            missing_weights = [False]
            if torch.distributed.get_rank() == 0:
                missing_weights[0] = not any(checkpoint_dir.glob("*.safetensors")) and not any(
                    checkpoint_dir.glob("*.bin")
                )
            torch.distributed.broadcast_object_list(missing_weights, src=0, group=get_gloo_group())
            if missing_weights[0]:
                raise RuntimeError(
                    f"HF export to {path} produced no weight files — the megatron "
                    f"bridge likely has no mapping for this model architecture."
                )
        if is_lora_model(model):
            publisher.write_adapter(None, checkpoint_dir / "adapter")

    if should_log:
        logger.info(f"Saving model in HuggingFace format to {path}")
    try:
        write_checkpoint_dir(path, write_weights, completion_marker=HF_EXPORT_COMPLETE_MARKER)
    except Exception as e:
        if raise_on_error:
            raise
        if should_log:
            logger.error(f"Failed to save HuggingFace format: {e}")
    else:
        if should_log:
            logger.info(f"Successfully saved HuggingFace model to {path}")


def add_uncovered_source_tensors(checkpoint_dir: Path, source: Path) -> None:
    """Add to the exported checkpoint the ``source`` tensors of modules the export does not cover.

    Exported tensors ``source`` also holds must match its shapes. Modules the Megatron model does not hold, such as a
    vision tower or disabled MTP layers, keep their ``source`` tensors; a module the export covers keeps only its
    exported layout, since ``source`` may store it differently (fused MoE experts against per-expert weights).
    """
    index_name = "model.safetensors.index.json"
    index = json.loads((checkpoint_dir / index_name).read_text())
    exported = index["weight_map"]
    source_map = json.loads((source / index_name).read_text())["weight_map"]
    covered = {name.rsplit(".", depth)[0] for name in exported for depth in range(1, name.count(".") + 1)}
    with ExitStack() as stack:
        source_files = {name: stack.enter_context(safe_open(source / name, "pt")) for name in set(source_map.values())}
        exported_files = {
            name: stack.enter_context(safe_open(checkpoint_dir / name, "pt")) for name in set(exported.values())
        }
        mismatched = sorted(
            name
            for name, file in exported.items()
            if name in source_map
            and source_files[source_map[name]].get_slice(name).get_shape()
            != exported_files[file].get_slice(name).get_shape()
        )
        assert not mismatched, f"HF tensors of another shape than in {source}: {mismatched[:8]}"
        kept = {
            name: source_files[file].get_tensor(name)
            for name, file in source_map.items()
            if name not in exported and name.rsplit(".", 1)[0] not in covered
        }
    if not kept:
        return
    shard = f"model-{len(set(exported.values())) + 1:05d}.safetensors"
    save_file(
        {name: tensor.contiguous() for name, tensor in kept.items()}, checkpoint_dir / shard, metadata={"format": "pt"}
    )
    index["weight_map"] = exported | dict.fromkeys(kept, shard)
    index["metadata"]["total_size"] += sum(tensor.numel() * tensor.element_size() for tensor in kept.values())
    (checkpoint_dir / index_name).write_text(json.dumps(index, indent=2))
    logger.info(f"Saved {checkpoint_dir}: {len(exported)} trained tensors, {len(kept)} kept from {source}")
