"""Compatibility import; implementation lives in miles.monitoring.collection.sglang_scraper."""

import sys

from miles.monitoring.collection import sglang_scraper as _implementation

sys.modules[__name__] = _implementation
