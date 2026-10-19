"""Compatibility import; implementation lives in miles.monitoring.collection.hooks."""

import sys

from miles.monitoring.collection import hooks as _implementation

sys.modules[__name__] = _implementation
