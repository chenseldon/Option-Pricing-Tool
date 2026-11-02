"""The government discount curve, and an honest account of what it is.

Two tenors is not a term structure
----------------------------------
The panel supplies the 1-year and the 10-year CGB yield, and nothing else. So
this is an *interpolated two-point curve*, not a bootstrap: there are no
deposits, no FRAs, no swaps, and no instrument stack to solve. Calling it a
bootstrap would overstate it by a wide margin, and the name is avoided
throughout.

What follows from having only two points:

- **Between them**, the zero rate is linear in maturity. With two points there
  is no information about curvature, so anything richer would be inventing
  shape.
- **Below 1 year it is flat**, deliberately. Extrapolating the 1y-10y slope
  down to zero feels principled and is not: the slope is about +6 bp a year, so
  it would move the short rate by a handful of basis points *in the wrong
  direction*, since the very short CNY end sits below the 1-year rather than
  above it.
- **Beyond 10 years it is flat** as well, which no product here reaches.

The consequence is worth stating plainly rather than leaving to be discovered:
every listed option in the panel expires inside 1.008 years, so for surface
construction this curve *is* "the 1-year yield, flat, refreshed daily". The
10-year point does real work only in a snowball's one-to-three-year tail.

Why bother, if it barely moves anything
---------------------------------------
Because the previous answer was a hardcoded 2%, and the 1-year yield actually
ranged 0.83% to 2.40% over the panel -- a worst-case error of 1.17 percentage
points, with 60.6% of trading days sitting more than 50 bp away from 2%.

But the expectation has to be set correctly: moving the rate from 1.1% to 2.8%
shifts a fitted 510300 surface's forwards only in the fourth decimal and its
at-the-money volatilities by at most 0.2 volatility points, about a third of
the calibration error. That is consistent with
:mod:`otcpricer.marketdata.forward`, which measured a 100 bp rate change moving
the implied forward by 0.003% and has a test pinning that insensitivity. In
this pipeline the rate's only first-order job is discounting. Getting it right
is a correctness matter, not an accuracy headline.

Conventions, and the one that is free to get wrong
--------------------------------------------------
The file quotes percentages (``1.9656`` means 1.9656%) of a *bond* yield, which
is annually compounded. Everything downstream is continuously compounded, so
``r = log(1 + y)``. The correction is 0.6 to 2.8 bp across the panel's range --
below the parity noise, and exactly the kind of silent convention bug that is
free to avoid and expensive to find later.

A government yield is also not a funding rate; an OTC desk discounts 20-60 bp
higher. This curve is labelled as what it is, and
:mod:`otcpricer.validation` measures the gap rather than papering over it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
from numpy.typing import NDArray

from otcpricer.core.daycount import DayCount
from otcpricer.core.exceptions import MarketDataError

__all__ = [
    "RATE_DAY_COUNT",
    "RateCurve",
    "rate_curve_from_panel",
]

#: ACT/365F throughout, matching the option slices' own day count so the two
#: sources' numbers stay comparable. See :mod:`otcpricer.core.daycount`.
RATE_DAY_COUNT = DayCount.ACT_365F

#: The tenors the panel actually supplies, in years.
PANEL_TENORS: tuple[float, ...] = (1.0, 10.0)


@dataclass(frozen=True, slots=True)
class RateCurve:
    """A zero curve through the tenors the data actually has.

    Attributes:
        as_of: The observation date.
        tenors: Maturities in years, ascending.
        zero_rates: Continuously compounded zero rates, as decimals.
        source: Where the quotes came from, for provenance.
    """

    as_of: date
    tenors: tuple[float, ...]
    zero_rates: tuple[float, ...]
    source: str = "unknown"

    def __post_init__(self) -> None:
        if len(self.tenors) != len(self.zero_rates):
            raise MarketDataError(
                f"a curve needs one rate per tenor, got {len(self.tenors)} tenors "
                f"and {len(self.zero_rates)} rates"
            )
        if not self.tenors:
            raise MarketDataError("a curve needs at least one tenor")
        if list(self.tenors) != sorted(self.tenors):
            raise MarketDataError(f"tenors must ascend, got {self.tenors}")
        if any(t <= 0.0 for t in self.tenors):
            raise MarketDataError(f"tenors must be positive, got {self.tenors}")
        if not all(np.isfinite(r) for r in self.zero_rates):
            raise MarketDataError(f"zero rates must be finite, got {self.zero_rates}")

    # ------------------------------------------------------------------ core

    def zero_rate(self, t: float | NDArray[np.float64]) -> Any:
        """Continuously compounded zero rate at maturity ``t``, in years.

        Linear in the zero rate between knots and flat outside them. ``np.interp``
        does exactly this, including the flat extrapolation, which is why it is
        used rather than written out.

        This is also the scalar rate to hand an engine for a product maturing
        at ``t``. There is no separate "flat equivalent" conversion, and a
        method offering one would be a second name for this one: since
        ``discount_factor`` is ``exp(-zero_rate(t) * t)`` by construction,
        ``-log(DF(T))/T`` returns ``zero_rate(T)`` identically. Exact for a
        cash flow at ``t``; a coupon stream needs the discount factors
        themselves, not one rate.
        """
        t_arr = np.asarray(t, dtype=float)
        if np.any(t_arr < 0.0):
            raise MarketDataError("maturities cannot be negative")
        rates = np.interp(t_arr, np.asarray(self.tenors), np.asarray(self.zero_rates))
        return float(rates) if np.isscalar(t) or t_arr.ndim == 0 else rates

    def discount_factor(self, t: float | NDArray[np.float64]) -> Any:
        """``exp(-r(t) * t)``, the present value of one unit paid at ``t``."""
        t_arr = np.asarray(t, dtype=float)
        factors = np.exp(-np.asarray(self.zero_rate(t_arr)) * t_arr)
        return float(factors) if np.isscalar(t) or t_arr.ndim == 0 else factors

    def forward_rate(self, t_from: float, t_to: float) -> float:
        """The continuously compounded forward rate over ``[t_from, t_to]``.

        What a float leg actually resets off, which is why ``/api/irs`` needs
        it: a swap whose floating leg pays the same constant every period is
        not a swap at all, it is the difference of two annuities.
        """
        if t_to <= t_from:
            raise MarketDataError(f"need t_to > t_from, got {t_from} and {t_to}")
        df_from = float(self.discount_factor(t_from))
        df_to = float(self.discount_factor(t_to))
        return float(np.log(df_from / df_to) / (t_to - t_from))

    # ------------------------------------------------------------- reporting

    @property
    def short_rate(self) -> float:
        """The flat level governing every listed option in the panel.

        Not a figure of speech: the longest tenor quoted is 1.008 years and the
        curve is flat below its first knot at 1 year, so this single number is
        what surface construction sees.
        """
        return float(self.zero_rates[0])

    @property
    def slope(self) -> float:
        """Zero-rate difference between the longest and shortest tenor."""
        return float(self.zero_rates[-1] - self.zero_rates[0])

    def as_dict(self) -> dict[str, object]:
        return {
            "as_of": self.as_of.isoformat(),
            "tenors": [round(t, 6) for t in self.tenors],
            "zero_rates": [round(r, 8) for r in self.zero_rates],
            "short_rate": round(self.short_rate, 8),
            "slope": round(self.slope, 8),
            "day_count": str(RATE_DAY_COUNT),
            "source": self.source,
            "note": (
                "Two tenors interpolated, not a bootstrap. Flat below the first "
                "knot, which is where every listed option in the panel sits."
            ),
        }


def _yield_columns(frame: Any) -> tuple[str, str]:
    """Locate the 1-year and 10-year columns by their embedded tenor.

    Matched on the digits rather than on the whole localised header: the
    column names came back mojibake under a GBK console, and
    :mod:`otcpricer.marketdata.sources` already records this lesson about
    comparing localised headers by literal equality.
    """
    import re

    found: dict[int, str] = {}
    for column in frame.columns:
        match = re.match(r"^\s*(\d+)\D", str(column))
        if match is not None:
            found[int(match.group(1))] = str(column)

    missing = [t for t in (1, 10) if t not in found]
    if missing:
        raise MarketDataError(
            f"the rate file is missing a {missing} year column; saw "
            f"{[str(c) for c in frame.columns]}"
        )
    return found[1], found[10]


def rate_curve_from_panel(panel: Any, as_of: date) -> RateCurve:
    """Build the curve for one date out of a :class:`~otcpricer.marketdata.wind.WindPanel`.

    The file covers every one of the panel's 725 trading days, plus 21
    interbank-only days -- a clean superset, so there is no carry-forward rule
    here and a missing date is an error rather than something to paper over.

    Raises:
        MarketDataError: if the date is absent, or duplicated with conflicting
            values.
    """
    import pandas as pd

    frame = panel.rates
    one_year, ten_year = _yield_columns(frame)

    stamp = pd.Timestamp(as_of)
    rows = frame[frame[frame.columns[0]] == stamp]
    if rows.empty:
        raise MarketDataError(f"the government yield file has no row for {as_of.isoformat()}")

    quotes = rows[[one_year, ten_year]].drop_duplicates()
    if len(quotes) > 1:
        # The file's tail repeats three dates, with identical values both
        # times. Identical repeats are an export artefact and are collapsed;
        # conflicting ones mean the file disagrees with itself, which is not
        # something to pick a winner from.
        raise MarketDataError(
            f"the government yield file gives {len(quotes)} different readings "
            f"for {as_of.isoformat()}; refusing to choose between them"
        )

    percent = quotes.iloc[0]
    # Percent to decimal, and an annually compounded bond yield to the
    # continuous compounding everything downstream assumes.
    rates = tuple(
        float(np.log1p(float(percent[column]) / 100.0)) for column in (one_year, ten_year)
    )

    return RateCurve(
        as_of=as_of,
        tenors=PANEL_TENORS,
        zero_rates=rates,
        source=f"wind-cgb:{as_of.isoformat()}",
    )
