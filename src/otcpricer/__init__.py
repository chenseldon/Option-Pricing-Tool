"""otcpricer -- OTC derivatives pricing and risk analytics.

Covers the desk workflow from quoting a single vanilla option through to
monitoring portfolio exposure: Black-Scholes-Merton and Monte Carlo for
vanillas, Crank-Nicolson PDE and path Monte Carlo for autocallables and
barriers, plus forwards and interest rate swaps.

The pricing models are implemented directly rather than delegated to a
third-party library, so each formula and discretisation is readable in place.
QuantLib is used only in the test suite, as an independent benchmark.
"""

from __future__ import annotations

__version__ = "8.0.0.dev0"

__all__ = ["__version__"]
