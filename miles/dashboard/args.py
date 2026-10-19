"""Compatibility import; implementation lives in miles.monitoring.transport.config."""

import sys

from miles.monitoring.transport import config as _implementation

sys.modules[__name__] = _implementation
