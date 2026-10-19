"""Legacy dashboard HTTP API import."""

import sys
from miles.monitoring.transport import api as _implementation

sys.modules[__name__] = _implementation
