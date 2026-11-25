"""Errors shared by rollout producers and collectors."""


class FatalRolloutError(RuntimeError):
    """Stop collection instead of submitting replacement rollout work."""


class RolloutSampleError(RuntimeError):
    """A known failed trial that may be replaced after its retry budget is exhausted."""
