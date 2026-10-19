"""Compatibility import; implementation lives in miles.monitoring.transport.postgres."""

import sys

from miles.monitoring.transport import postgres as _implementation

sys.modules[__name__] = _implementation
