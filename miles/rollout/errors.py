"""Errors shared by rollout producers and collectors."""


class FatalRolloutError(RuntimeError):
    """Stop collection instead of submitting replacement rollout work."""
