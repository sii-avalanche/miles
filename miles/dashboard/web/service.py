"""Compatibility import; implementation lives in miles.monitoring.transport.service."""

import sys

from miles.monitoring.transport import service as _implementation

sys.modules[__name__] = _implementation
