"""Compatibility import; implementation lives in miles.monitoring.transport.backend."""

import sys

from miles.monitoring.transport import backend as _implementation

sys.modules[__name__] = _implementation
