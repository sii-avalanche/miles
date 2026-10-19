"""Compatibility import; implementation lives in miles.monitoring.collection.gpu_sampler."""

import sys

from miles.monitoring.collection import gpu_sampler as _implementation

sys.modules[__name__] = _implementation
