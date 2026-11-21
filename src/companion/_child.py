"""Run a child command and make sure it dies with the companion.

macOS has no "kill me when my parent dies" flag, so the Carle daemon and speak server are
started through this wrapper: it polls its parent and, if the companion disappears for any
reason (crash, force quit, SIGKILL), terminates the child so no orphan keeps the BLE link or
the loopback port.

    python -m companion._child -- carle daemon start <uuid> --foreground
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        print("usage: python -m companion._child -- <command> [args...]", file=sys.stderr)
        return 2
    parent = os.getppid()
    child = subprocess.Popen(argv)

    def forward(signum, frame):  # noqa: ARG001
        child.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    while True:
        rc = child.poll()
        if rc is not None:
            return rc
        if os.getppid() != parent:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
            return 0
        time.sleep(1.0)


if __name__ == "__main__":
    sys.exit(main())
