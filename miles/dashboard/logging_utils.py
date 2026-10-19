"""Compatibility import; implementation lives in miles.monitoring.transport.logging_utils."""

import sys

from miles.monitoring.transport import logging_utils as _implementation

sys.modules[__name__] = _implementation
