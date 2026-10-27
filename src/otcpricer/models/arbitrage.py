"""Static-arbitrage checks on a fitted volatility surface.

Calibration already constrains the analytic no-arbitrage conditions, so this is
a *second, independent* check: evaluate the conditions pointwise on a dense grid
rather than trusting that the constraints were satisfied where it matters. The
two can disagree -- the fitted conditions hold at the quoted maturities, while
the surface is used at every maturity in between, where the interpolated
variance term structure governs.

The two conditions
------------------
**Butterfly.** Gatheral's ``g(k) > 0``. Equivalent to the risk-neutral density
being non-negative, and -- the reason it matters here -- identically the
denominator of the Dupire local-variance formula. Where it fails, local
volatility is undefined.

**Calendar.** ``dw/dT >= 0``: total implied variance cannot decrease with
maturity. This is Dupire's numerator, so a violation means a negative local
variance even with a healthy denominator.

Together they are exactly the requirement that Dupire yields a real, positive
local volatility. The check therefore doubles as a precondition for Phase 2b,
and :func:`local_variance_is_well_defined` says so directly.

A checker that cannot fail is worthless, so the test suite builds surfaces with
deliberate arbitrage and requires this to find them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from otcpricer.core.logging import get_logger
from otcpricer.models.svi import SSVISurface

__all__ = [
    "ArbitrageReport",
    "Violation",
    "check_surface",
    "local_variance_is_well_defined",
]

_log = get_logger(__name__)

#: How far out in log-moneyness to check. Wider than any sane quote range: the
#: surface will be extrapolated by a PDE grid that spans several standard
#: deviations, so it must behave out there too.
DEFAULT_MONEYNESS_RANGE = 0.8

#: Tolerance for treating a condition as satisfied. Small negative values at the
#: very edge are numerical, not economic.
DEFAULT_TOLERANCE = -1e-10


@dataclass(frozen=True, slots=True)
class Violation:
    """One point where a condition fails.

    Attributes:
        condition: ``"butterfly"`` or ``"calendar"``.
        log_moneyness: Where it failed.
        maturity: Maturity at which it failed.
        value: The condition's value; negative means violated.
    """

    condition: str
    log_moneyness: float
    maturity: float
    value: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "condition": self.condition,
            "log_moneyness": round(self.log_moneyness, 6),
            "maturity": round(self.maturity, 6),
            "value": float(self.value),
        }


@dataclass(frozen=True, slots=True)
class ArbitrageReport:
    """Outcome of checking a surface on a grid.

    Attributes:
        min_butterfly: Smallest ``g(k)`` found; must be positive.
        min_calendar: Smallest ``dw/dT`` found; must be non-negative.
        violations: Worst offending points, for locating the problem.
        n_checked: Grid points evaluated.
        moneyness_range: Half-width of the log-moneyness range checked.
        maturity_range: ``(min, max)`` maturities checked.
    """

    min_butterfly: float
    min_calendar: float
    violations: tuple[Violation, ...]
    n_checked: int
    moneyness_range: float
    maturity_range: tuple[float, float]

    @property
    def butterfly_free(self) -> bool:
        return self.min_butterfly > 0.0

    @property
    def calendar_free(self) -> bool:
        return self.min_calendar >= DEFAULT_TOLERANCE

    @property
    def arbitrage_free(self) -> bool:
        """Whether the surface is free of static arbitrage on the grid checked."""
        return self.butterfly_free and self.calendar_free

    def as_dict(self) -> dict[str, object]:
        """Flat dict for API responses and reports."""
        return {
            "arbitrage_free": self.arbitrage_free,
            "butterfly_free": self.butterfly_free,
            "calendar_free": self.calendar_free,
            "min_butterfly": float(self.min_butterfly),
            "min_calendar": float(self.min_calendar),
            "n_checked": self.n_checked,
            "moneyness_range": self.moneyness_range,
            "maturity_range": [round(t, 6) for t in self.maturity_range],
            "n_violations": len(self.violations),
            "worst_violations": [v.as_dict() for v in self.violations[:10]],
        }

    def summary(self) -> str:
        """One line, for logs and report headers."""
        if self.arbitrage_free:
            return (
                f"arbitrage-free on {self.n_checked} points "
                f"(min g={self.min_butterfly:.4f}, min dw/dT={self.min_calendar:.3e})"
            )
        failed = []
        if not self.butterfly_free:
            failed.append(f"butterfly (min g={self.min_butterfly:.3e})")
        if not self.calendar_free:
            failed.append(f"calendar (min dw/dT={self.min_calendar:.3e})")
        return f"ARBITRAGE: {', '.join(failed)}, {len(self.violations)} points"


def check_surface(
    surface: SSVISurface,
    *,
    moneyness_range: float = DEFAULT_MONEYNESS_RANGE,
    n_moneyness: int = 161,
    maturities: np.ndarray | list[float] | None = None,
    n_maturities: int = 60,
    tolerance: float = DEFAULT_TOLERANCE,
    max_violations: int = 50,
) -> ArbitrageReport:
    """Check a surface for static arbitrage on a dense grid.

    Args:
        surface: The fitted surface.
        moneyness_range: Half-width in log-moneyness to span.
        n_moneyness: Grid points across moneyness.
        maturities: Maturities to check. Defaults to a grid spanning from well
            inside the first quoted expiry to beyond the last, because the
            surface gets used at maturities between and past the quoted ones --
            which is exactly where the interpolated term structure, rather than
            the fitted constraints, decides the answer.
        n_maturities: Grid points across maturity when defaulted.
        tolerance: Threshold below which a value counts as a violation.
        max_violations: Cap on recorded violations.

    Returns:
        An :class:`ArbitrageReport`.
    """
    if maturities is None:
        quoted = np.asarray(surface.params.maturities, dtype=float)
        maturities = np.linspace(max(quoted[0] * 0.5, 1e-4), quoted[-1] * 1.5, n_maturities)
    maturity_grid = np.asarray(maturities, dtype=float)
    moneyness_grid = np.linspace(-moneyness_range, moneyness_range, n_moneyness)

    violations: list[Violation] = []
    min_butterfly = np.inf
    min_calendar = np.inf

    for maturity in maturity_grid:
        butterfly = np.asarray(surface.butterfly(moneyness_grid, maturity))
        calendar = np.asarray(surface.dw_dT(moneyness_grid, maturity))

        min_butterfly = min(min_butterfly, float(np.min(butterfly)))
        min_calendar = min(min_calendar, float(np.min(calendar)))

        for name, values, limit in (
            ("butterfly", butterfly, 0.0),
            ("calendar", calendar, tolerance),
        ):
            bad = np.flatnonzero(values < limit)
            for index in bad:
                if len(violations) >= max_violations:
                    break
                violations.append(
                    Violation(
                        condition=name,
                        log_moneyness=float(moneyness_grid[index]),
                        maturity=float(maturity),
                        value=float(values[index]),
                    )
                )

    # Worst first, so a truncated list is still the informative one.
    violations.sort(key=lambda v: v.value)

    report = ArbitrageReport(
        min_butterfly=float(min_butterfly),
        min_calendar=float(min_calendar),
        violations=tuple(violations),
        n_checked=int(maturity_grid.size * moneyness_grid.size),
        moneyness_range=moneyness_range,
        maturity_range=(float(maturity_grid[0]), float(maturity_grid[-1])),
    )
    _log.info("surface arbitrage check", extra={"result": report.summary()})
    return report


def local_variance_is_well_defined(report: ArbitrageReport) -> tuple[bool, str]:
    """Whether Dupire local variance exists and is non-negative on this surface.

    The Dupire denominator *is* the butterfly expression and its numerator *is*
    ``dw/dT``, so this follows directly from the two conditions. Stated as its
    own function because it is the precondition Phase 2b depends on, and
    phrasing it in those terms makes the dependency explicit rather than
    something a reader has to reconstruct.
    """
    if report.arbitrage_free:
        return True, (
            "local variance is well defined and non-negative: the butterfly "
            "condition keeps the Dupire denominator positive and the calendar "
            "condition keeps its numerator non-negative"
        )

    problems = []
    if not report.butterfly_free:
        problems.append(
            f"the Dupire denominator vanishes or changes sign (min g = {report.min_butterfly:.3e})"
        )
    if not report.calendar_free:
        problems.append(f"the Dupire numerator is negative (min dw/dT = {report.min_calendar:.3e})")
    return False, "local variance is not well defined: " + "; and ".join(problems)
