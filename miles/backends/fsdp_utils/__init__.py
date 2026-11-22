"""FSDP exports without importing the actor dependency graph during CLI validation."""

__all__ = ["load_fsdp_args", "FSDPTrainRayActor"]


def __getattr__(name):
    # DModel reuses the dataclass parser. Importing arguments/checkpoint helpers
    # must not also require the optional inference and actor dependencies.
    if name == "FSDPTrainRayActor":
        from miles.backends.fsdp_utils.actor import FSDPTrainRayActor

        return FSDPTrainRayActor
    if name == "load_fsdp_args":
        from miles.backends.fsdp_utils.arguments import load_fsdp_args

        return load_fsdp_args
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
