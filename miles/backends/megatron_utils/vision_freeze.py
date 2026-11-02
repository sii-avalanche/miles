"""Explicit Megatron module boundaries for vision freezing, before optimizer setup."""

import logging

import torch

logger = logging.getLogger(__name__)


def freeze_vision_modules(model, *, tower_paths, projection_paths, freeze_tower, freeze_projection):
    """Resolve every path first; projection parameters win when nested in a tower."""
    groups = []
    for paths, frozen in ((tower_paths, freeze_tower), (projection_paths, freeze_projection)):
        modules = []
        for path in paths:
            if not isinstance(path, str) or not path or any(not part for part in path.split(".")):
                raise ValueError(f"Expected a nonempty module path, got {path!r}")
            try:
                module = model.get_submodule(path)
            except AttributeError as error:
                raise ValueError(f"Vision module {path!r} does not exist on {type(model).__name__}") from error
            if not isinstance(module, torch.nn.Module) or not any(True for _ in module.parameters()):
                raise ValueError(f"Vision module {path!r} must contain parameters")
            modules.append((path, module))
        groups.append((modules, frozen))

    # Apply broad tower policy first, then the more specific projector policy.
    for modules, frozen in groups:
        for _, module in modules:
            module.requires_grad_(not frozen)
    for modules, _ in groups:
        for path, module in modules:
            parameters = list(module.parameters())
            logger.info(
                "Vision module %s: parameters=%d trainable=%d (after projection precedence)",
                path,
                sum(p.numel() for p in parameters),
                sum(p.numel() for p in parameters if p.requires_grad),
            )
