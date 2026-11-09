"""Optional per-training-call hooks without coupling a backend to a recipe."""

from contextlib import contextmanager
from contextvars import ContextVar

_validator = ContextVar("training_batch_validator", default=None)


def current_train_validator():
    return _validator.get()


@contextmanager
def train_validator_scope(validator):
    token = _validator.set(validator)
    try:
        yield validator
    finally:
        _validator.reset(token)
