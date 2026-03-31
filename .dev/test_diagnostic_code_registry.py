"""Contract test for diagnostic code registry coverage.

Run:
    python .dev/test_diagnostic_code_registry.py
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure project root is importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Add .dev/ itself to path so dev_healthcheck (in a dot-prefixed folder) is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dev_healthcheck import _all_py_files, check_diagnostic_codes


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    files = _all_py_files()
    violations, registry_exists = check_diagnostic_codes(files)
    _assert(registry_exists, "contracts/diagnostic_codes.py must exist")
    _assert(
        not violations,
        "All diagnostics.emit() code strings must be registered in contracts/diagnostic_codes.py\n"
        + "\n".join(violations),
    )


if __name__ == "__main__":
    main()
