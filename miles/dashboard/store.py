"""Compatibility import; implementation lives in miles.monitoring.transport.store."""

import sys

from miles.monitoring.transport import store as _implementation

sys.modules[__name__] = _implementation
