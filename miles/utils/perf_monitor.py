"""Compatibility import; implementation lives in miles.monitoring.collection.runtime."""

import sys

from miles.monitoring.collection import runtime as _implementation

sys.modules[__name__] = _implementation
