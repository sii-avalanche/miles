"""Compatibility import; implementation lives in miles.monitoring.transport.publisher."""

import sys

from miles.monitoring.transport import publisher as _implementation

sys.modules[__name__] = _implementation
