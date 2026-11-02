import logging
import re

from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

logger = logging.getLogger(__name__)

_DECODER_LAYER = re.compile(r"(?:^|\.)decoder\.layers\.(\d+)\.")


def register_replay_list_moe(replay_list, replay_data, *, models, **_kwargs):
    """Give each main-decoder router the replay stream of its own global layer."""
    routers = [
        (
            get_transformer_layer_offset(model.module.config, vp_stage=vp_stage) + int(found.group(1)),
            module.routing_replay,
        )
        for vp_stage, model in enumerate(models)
        for name, module in model.named_modules()
        if getattr(module, "routing_replay", None) is not None
        and "vision_model" not in name
        and (found := _DECODER_LAYER.search(name)) is not None
    ]
    if len(routers) != len(replay_list):
        logger.warning("%d decoder routers among %d registered replays", len(routers), len(replay_list))
    for layer_idx, replay in routers:
        replay.record(replay_data[:, layer_idx])
