from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TEST_PREFIXES = (
    "phase0_tests",
    "simulation_tests",
    "resource_test",
    "resource_grant_test",
    "resource_attack_test",
    "resource_seed_test",
    "resource_welfare_formula",
)


def _run(cmd: list[str]) -> int:
    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd)


def _cleanup_test_artifacts() -> None:
    for folder in ("logs", "round_data"):
        path = Path(folder)
        if not path.exists():
            continue
        for entry in path.iterdir():
            if not entry.is_file():
                continue
            if not any(entry.name.startswith(prefix) for prefix in TEST_PREFIXES):
                continue
            entry.unlink()


def main() -> int:
    cmds = [
        [sys.executable, "-m", "tests.run_phase0_tests"],
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
        [sys.executable, "ui_app/smoke_test.py"],
    ]
    for cmd in cmds:
        code = _run(cmd)
        _cleanup_test_artifacts()
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
