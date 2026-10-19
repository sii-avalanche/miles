"""Compatibility import for miles.monitoring.aggregation.diagnostics."""

import sys

from miles.monitoring.aggregation import diagnostics as _implementation

sys.modules[__name__] = _implementation
