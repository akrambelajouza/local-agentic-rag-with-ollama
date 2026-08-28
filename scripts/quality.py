"""Run the same deterministic quality gate used by GitHub Actions."""

from __future__ import annotations

import subprocess
import sys

COMMANDS = (
    (sys.executable, "-m", "ruff", "format", "--check", "."),
    (sys.executable, "-m", "ruff", "check", "."),
    (sys.executable, "-m", "coverage", "erase"),
    (sys.executable, "-m", "coverage", "run", "-m", "unittest", "discover"),
    (sys.executable, "-m", "coverage", "report"),
    (sys.executable, "-m", "pip", "check"),
)


def main() -> None:
    for command in COMMANDS:
        print(f"+ {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
