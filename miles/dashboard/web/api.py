"""Compatibility import; implementation lives in miles.monitoring.transport.api."""

import sys

from miles.monitoring.transport import api as _implementation

sys.modules[__name__] = _implementation
