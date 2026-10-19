"""Compatibility import; implementation lives in miles.monitoring.transport.postgres_reader."""

import sys

from miles.monitoring.transport import postgres_reader as _implementation

sys.modules[__name__] = _implementation
