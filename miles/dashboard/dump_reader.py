"""Compatibility import; implementation lives in miles.monitoring.transport.dump_reader."""

import sys

from miles.monitoring.transport import dump_reader as _implementation

sys.modules[__name__] = _implementation
