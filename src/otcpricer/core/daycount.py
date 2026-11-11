"""Day-count conventions and tenor parsing.

The legacy code passed maturity around as a bare float year fraction, with the
unit conversion done ad hoc in the Flask layer (days vs years) and the
"252 trading days" assumption buried inside the FDM grid builder. Pricing a
snowball correctly needs both clocks at once:

* **calendar time** discounts cash flows and drives theta;
* **trading time** drives the observation schedule -- an A-share snowball
  observes knock-out monthly and knock-in on each *trading* day.

Keeping both explicit here stops the two from being silently conflated.
"""

from __future__ import annotations

import re
from datetime import date
from enum import StrEnum

from otcpricer.core.exceptions import ParameterError

__all__ = [
    "DayCount",
    "year_fraction",
    "parse_tenor",
    "TRADING_DAYS_PER_YEAR",
    "CALENDAR_DAYS_PER_YEAR",
]

#: Conventional number of A-share trading days in a year. Used for observation
#: schedules and for the default PDE time grid.
TRADING_DAYS_PER_YEAR = 252

#: Conventional calendar-day basis for ACT/365F discounting and daily theta.
CALENDAR_DAYS_PER_YEAR = 365


class DayCount(StrEnum):
    """Supported day-count bases.

    ``ACT_365F`` is the default throughout: it is the market convention for
    CNY-denominated OTC equity derivatives and matches the ``theta / 365``
    scaling the legacy Greeks already used.
    """

    ACT_365F = "ACT/365F"
    ACT_360 = "ACT/360"
    ACT_ACT = "ACT/ACT"
    THIRTY_360 = "30/360"

    def year_fraction(self, start: date, end: date) -> float:
        """Year fraction between two dates under this convention."""
        return year_fraction(start, end, self)


def _thirty_360_days(start: date, end: date) -> int:
    """Day count under the 30/360 (Bond Basis) convention."""
    d1 = min(start.day, 30)
    d2 = min(end.day, 30) if d1 == 30 else end.day
    return 360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)


def _act_act_fraction(start: date, end: date) -> float:
    """ACT/ACT (ISDA): split the period at year boundaries, each on its own basis."""
    if start.year == end.year:
        basis = 366.0 if _is_leap(start.year) else 365.0
        return (end - start).days / basis

    total = 0.0
    # Leading stub: start -> 1 Jan of the following year.
    stub_end = date(start.year + 1, 1, 1)
    total += (stub_end - start).days / (366.0 if _is_leap(start.year) else 365.0)
    # Whole years in between.
    total += float(end.year - start.year - 1)
    # Trailing stub: 1 Jan of the end year -> end.
    stub_start = date(end.year, 1, 1)
    total += (end - stub_start).days / (366.0 if _is_leap(end.year) else 365.0)
    return total


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def year_fraction(start: date, end: date, convention: DayCount = DayCount.ACT_365F) -> float:
    """Year fraction from ``start`` to ``end``.

    Args:
        start: Period start (typically the trade or valuation date).
        end: Period end (typically maturity).
        convention: Day-count basis.

    Returns:
        The accrued year fraction. Negative if ``end`` precedes ``start``,
        which callers should treat as an expired trade rather than an error.

    Raises:
        ParameterError: if the convention is not recognised.
    """
    if convention is DayCount.ACT_365F:
        return (end - start).days / 365.0
    if convention is DayCount.ACT_360:
        return (end - start).days / 360.0
    if convention is DayCount.THIRTY_360:
        return _thirty_360_days(start, end) / 360.0
    if convention is DayCount.ACT_ACT:
        return _act_act_fraction(start, end)
    raise ParameterError(f"unsupported day-count convention: {convention!r}")


_TENOR_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([dwmy])\s*$", re.IGNORECASE)

#: Year fraction of one unit of each tenor code, on an ACT/365F-style basis.
_TENOR_UNITS = {"d": 1.0 / 365.0, "w": 7.0 / 365.0, "m": 1.0 / 12.0, "y": 1.0}


def parse_tenor(tenor: str | float | int) -> float:
    """Convert a tenor to a year fraction.

    Accepts ``'3M'``, ``'90D'``, ``'1Y'``, ``'2W'`` (case-insensitive), or a
    number already expressed in years.

    Months and years use exact fractions (``1M`` is 1/12, not 30/365) because
    observation schedules are built on month counts; days and weeks use
    ACT/365F.

    Raises:
        ParameterError: if the string is not a recognisable tenor.
    """
    if isinstance(tenor, int | float):
        return float(tenor)

    match = _TENOR_RE.match(str(tenor))
    if match is None:
        raise ParameterError(
            f"cannot parse tenor {tenor!r}; expected forms like '3M', '90D', '1Y', '2W'"
        )
    magnitude, unit = match.groups()
    return float(magnitude) * _TENOR_UNITS[unit.lower()]
