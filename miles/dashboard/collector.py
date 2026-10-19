"""Compatibility import; implementation lives in miles.monitoring.transport.collector."""

import sys

from miles.monitoring.transport import collector as _implementation

sys.modules[__name__] = _implementation
