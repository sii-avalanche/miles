"""Compatibility import; implementation lives in miles.monitoring.transport.group_history."""

import sys

from miles.monitoring.transport import group_history as _implementation

sys.modules[__name__] = _implementation
