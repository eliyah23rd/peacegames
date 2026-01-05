from __future__ import annotations

import subprocess
import sys


def _run(cmd: list[str]) -> int:
    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd)


def main() -> int:
    cmds = [
        [sys.executable, "-m", "tests.run_phase0_tests"],
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
        [sys.executable, "ui_app/smoke_test.py"],
    ]
    for cmd in cmds:
        code = _run(cmd)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
