"""Shared fixtures and optional-dependency handling for the test suite.

QuantLib and numba are development-only dependencies: the library runs without
them. Tests that need them are skipped rather than failed when they are absent,
so a clone with only the runtime requirements still gets a green suite.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any, NamedTuple

import pytest

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    import QuantLib as ql  # noqa: N813 -- 'ql' is the library's own conventional alias

    HAS_QUANTLIB = True
except ImportError:  # pragma: no cover - exercised only on a runtime-only install
    ql = None  # type: ignore[assignment]
    HAS_QUANTLIB = False

requires_quantlib = pytest.mark.skipif(
    not HAS_QUANTLIB, reason="QuantLib not installed (development dependency)"
)


# ---------------------------------------------------------------------------
# Market scenarios
# ---------------------------------------------------------------------------


class Market(NamedTuple):
    """One set of vanilla option inputs."""

    option_type: str
    S: float
    K: float
    T: float
    r: float
    sigma: float
    q: float

    def __repr__(self) -> str:  # keeps pytest -v output readable
        return (
            f"{self.option_type}(S={self.S:g},K={self.K:g},T={self.T:g},"
            f"r={self.r:g},sig={self.sigma:g},q={self.q:g})"
        )


#: The project's reference case, carried from the v7.0.0 baseline. Every number
#: here was measured against the pre-upgrade implementation, so it doubles as a
#: regression anchor: if a refactor moves these, something broke.
BASELINE = Market("c", 5500.0, 5500.0, 0.25, 0.02, 0.20, 0.01)

#: Expected values for :data:`BASELINE`, measured on the v7.0.0 tag.
BASELINE_EXPECTED = {
    "price": 225.4236,
    "delta": 0.52857,
    "gamma": 0.000722,
    "vega": 10.912784,
    "theta": -1.263217,
    "rho": 6.704273,
}


def _market_grid() -> Iterator[Market]:
    """A spread of inputs covering the awkward corners, not just the easy middle.

    Deliberately includes a negative rate (real, and a CNY desk sees negative
    net carry once the futures basis is folded in), a one-day maturity, a 150%
    volatility, and strikes well outside the money.
    """
    for option_type, S, ratio, T, r, sigma, q in itertools.product(
        ("c", "p"),
        (100.0, 5500.0),
        (0.7, 0.9, 1.0, 1.1, 1.3),
        (1.0 / 365.0, 0.25, 1.0, 5.0),
        (-0.01, 0.0, 0.02, 0.08),
        (0.05, 0.20, 0.60, 1.50),
        (0.0, 0.03),
    ):
        yield Market(option_type, S, S * ratio, T, r, sigma, q)


#: Materialised once; ~2560 cases, each cheap (vectorised closed form).
MARKET_GRID = tuple(_market_grid())

#: A small, readable subset for tests where the full grid would be noise.
MARKET_SAMPLE = (
    BASELINE,
    Market("p", 5500.0, 5000.0, 0.50, 0.02, 0.22, 0.01),
    Market("c", 100.0, 120.0, 1.00, 0.03, 0.35, 0.00),
    Market("p", 100.0, 80.0, 2.00, 0.05, 0.15, 0.02),
    Market("c", 100.0, 100.0, 1.0 / 365.0, 0.00, 0.80, 0.00),
    Market("p", 100.0, 100.0, 5.00, -0.01, 0.10, 0.00),
)


@pytest.fixture
def baseline() -> Market:
    """The v7.0.0 reference case."""
    return BASELINE


@pytest.fixture(params=MARKET_SAMPLE, ids=repr)
def market(request: pytest.FixtureRequest) -> Market:
    """Parametrised over the readable market sample."""
    return request.param  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# QuantLib reference pricer
# ---------------------------------------------------------------------------


def quantlib_european(
    option_type: str,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
) -> dict[str, Any]:
    """Price a European option with QuantLib, as an independent benchmark.

    ``T`` is converted to a whole number of days and the evaluation date is set
    explicitly, because QuantLib works on a date calendar while our engine
    takes a year fraction directly. Callers must pass a ``T`` that is already
    an exact multiple of 1/365 so the two clocks agree; otherwise the
    comparison measures the calendar mismatch rather than the maths.

    Returns:
        Price and Greeks, with vega and rho rescaled to our per-percentage-point
        convention.
    """
    if not HAS_QUANTLIB:  # pragma: no cover
        raise RuntimeError("QuantLib is not installed")

    today = ql.Date(26, 10, 2026)
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    expiry = today + int(round(T * 365))

    payoff = ql.PlainVanillaPayoff(ql.Option.Call if option_type == "c" else ql.Option.Put, K)
    option = ql.VanillaOption(payoff, ql.EuropeanExercise(expiry))

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
    rate_curve = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
    div_curve = ql.YieldTermStructureHandle(ql.FlatForward(today, q, day_count))
    vol_surface = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), sigma, day_count)
    )
    process = ql.BlackScholesMertonProcess(spot_handle, div_curve, rate_curve, vol_surface)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    return {
        "price": option.NPV(),
        "delta": option.delta(),
        "gamma": option.gamma(),
        "vega": option.vega() * 0.01,
        "rho": option.rho() * 0.01,
    }


def on_day_grid(T_days: int) -> float:
    """Return a maturity in years that lands exactly on QuantLib's day calendar."""
    return T_days / 365.0
