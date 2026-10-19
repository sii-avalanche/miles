"""Compatibility import; implementation lives in miles.monitoring.transport.catalog."""

import sys

from miles.monitoring.transport import catalog as _implementation

sys.modules[__name__] = _implementation
