"""Report whether this interpreter can run the application.

Why a file rather than ``python -c``
------------------------------------
The launchers used to inline this probe. ``run_app.ps1`` passed it as a
PowerShell here-string and PowerShell stripped the quotes out of it, so the
probe died with a ``SyntaxError`` and the launcher reported "the interpreter
could not run the dependency probe" -- about a probe that had been mangled on
the way in rather than about the interpreter.

A file has no quoting layer to get wrong, is the same code for both launchers,
and can be tested.

Exit codes
----------
``0``  everything needed is importable
``1``  something is missing; the names and their fix go to stdout

Run it directly to see what an interpreter is missing::

    python scripts/check_deps.py
    E:\\anaconda\\envs\\Quant1.0\\python.exe scripts/check_deps.py
"""

from __future__ import annotations

import os
import sys

#: What the application cannot start without.
#:
#: ``otcpricer`` is the project's own library. It is included because a clone
#: that has not been installed is the normal case for a reader, and the
#: ``src/`` append below is what makes it importable -- so if this still fails,
#: the checkout itself is wrong rather than the environment.
REQUIRED = (
    ("flask", "the web framework"),
    ("numpy", "numerics"),
    ("scipy", "numerics"),
    ("pandas", "market data tables"),
    ("pydantic", "settings and environment overrides"),
    ("openpyxl", "the /api/export spreadsheet"),
    ("otcpricer", "this project's pricing library"),
    ("waitress", "serves the application; see serve.py"),
)

#: Needed only for part of the surface, and named separately so a partial
#: environment is reported as partial rather than broken.
OPTIONAL = (
    ("pyarrow", "reading the Wind panel, which is parquet"),
    ("matplotlib", "the CLI chart output in main.py / visualization.py"),
    ("QuantLib", "cross-validation tests only, never the application"),
    ("numba", "optional JIT paths; the engines fall back to numpy"),
)


def _add_src_to_path() -> None:
    """Let a clone import ``otcpricer`` without having been installed.

    Appended, not inserted, so an installed copy still wins.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(here, "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.append(src)


def _missing(candidates: tuple[tuple[str, str], ...]) -> list[tuple[str, str]]:
    out = []
    for name, why in candidates:
        try:
            __import__(name)
        except ImportError:
            out.append((name, why))
    return out


def main() -> int:
    """Print a verdict and return a process exit code."""
    _add_src_to_path()

    required_missing = _missing(REQUIRED)
    optional_missing = _missing(OPTIONAL)

    if not required_missing:
        print(f"deps-ok  python {sys.version.split()[0]}  {sys.executable}")
        for name, why in optional_missing:
            print(f"  optional, absent: {name:<12} ({why})")
        return 0

    print(f"deps-missing  python {sys.version.split()[0]}  {sys.executable}")
    for name, why in required_missing:
        print(f"  REQUIRED, absent: {name:<12} ({why})")
    print()
    print("  Fix:  " + sys.executable + " -m pip install -r requirements.txt")
    if any(name == "waitress" for name, _ in required_missing):
        print(
            "  waitress is what serves the app. For the development server "
            "instead\n"
            "  (interactive debugger, one request at a time):  " + sys.executable + " app.py"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
