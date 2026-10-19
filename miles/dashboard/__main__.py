"""Compatible dashboard command entrypoint."""

import sys

from miles.monitoring.transport import cli as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
