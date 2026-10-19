"""Legacy single-run dashboard entrypoint."""

import sys

from miles.monitoring.transport.cli import main as _main
from miles.monitoring.transport.cli import make_demo_dir as make_demo_dir


def main(argv=None):
    _main(["view", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    main()
