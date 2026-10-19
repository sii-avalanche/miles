"""Compatibility import for miles.monitoring.transport.status."""

import sys

from miles.monitoring.transport import status as _implementation

sys.modules[__name__] = _implementation
