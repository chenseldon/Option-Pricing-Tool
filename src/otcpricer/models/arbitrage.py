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

A third thing worth reporting, which is not arbitrage
----------------------------------------------------
``dw/dT == 0`` satisfies the calendar condition and is still useless: a flat
total variance means a local variance of *zero*. This is not hypothetical --
it is how a real defect hid. PCHIP clamps its endpoint derivative to zero
whenever the variance term structure is concave enough at the end, so the
extrapolated tail went flat and local volatility vanished beyond the last
quoted expiry, while this checker reported the surface arbitrage-free, because
it was. See :class:`~otcpricer.models.svi.SSVISurface`.

So the calendar condition is also measured *relative* to the surface's own
average variance accumulation rate ``theta(T) / T``, which is scale-free and
separates the cases cleanly. Measured: 0.839 and 0.832 on the two fitted real
surfaces, 0.387 and 0.119 on synthetic surfaces with steeply downward-sloping
volatility term structures, and 0.000 on the degenerate flat tail. The floor
below which the surface is called degenerate is 1%, twelve times below the
tightest legitimate case.

A checker that cannot fail is worthless, so the test suite builds surfaces with
deliberate arbitrage and requires this to find them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from otcpricer.core.logging import get_logger
from otcpricer.models.svi import SSVISurface

__all__ = [
    "DEFAULT_DEGENERACY_FLOOR",
    "DEFAULT_MAX_MATURITY",
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

#: Longest maturity the default grid reaches, in years. Not derived from the
#: quoted expiries, deliberately: the listed ETF options stop at roughly five
#: months while the structured products priced off this surface run one to
#: three years, so a grid bounded by the data would leave the entire region of
#: use unchecked. That region is where the flat-tail defect lived.
DEFAULT_MAX_MATURITY = 3.0

#: Smallest ``dw/dT``, as a fraction of the surface's own average variance rate
#: ``theta(T) / T``, that still yields a usable local volatility. Satisfying the
#: calendar condition is not enough on its own; see the module docstring for the
#: measurements behind this value.
DEFAULT_DEGENERACY_FLOOR = 1e-2


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
        min_calendar_relative: Smallest ``dw/dT`` as a fraction of
            ``theta(T) / T``. Near zero means local variance collapses even
            though no arbitrage is present.
        violations: Worst offending points, for locating the problem.
            Capped, so this can be shorter than :attr:`n_violations`.
        n_violations: How many grid points violated a condition in total.
        n_checked: Grid points evaluated.
        moneyness_range: Half-width of the log-moneyness range checked.
        maturity_range: ``(min, max)`` maturities checked.
    """

    min_butterfly: float
    min_calendar: float
    min_calendar_relative: float
    violations: tuple[Violation, ...]
    n_violations: int
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
        """Whether the surface is free of static arbitrage on the grid checked.

        Deliberately independent of :attr:`calendar_degenerate`: a surface with
        a flat variance term structure genuinely admits no arbitrage. It is
        simply unusable for local volatility, which is a separate question.
        """
        return self.butterfly_free and self.calendar_free

    @property
    def calendar_degenerate(self) -> bool:
        """Whether ``dw/dT`` is so small that local variance collapses."""
        return self.min_calendar_relative < DEFAULT_DEGENERACY_FLOOR

    def as_dict(self) -> dict[str, object]:
        """Flat dict for API responses and reports."""
        return {
            "arbitrage_free": self.arbitrage_free,
            "butterfly_free": self.butterfly_free,
            "calendar_free": self.calendar_free,
            "calendar_degenerate": self.calendar_degenerate,
            "min_butterfly": float(self.min_butterfly),
            "min_calendar": float(self.min_calendar),
            "min_calendar_relative": float(self.min_calendar_relative),
            "n_checked": self.n_checked,
            "moneyness_range": self.moneyness_range,
            "maturity_range": [round(t, 6) for t in self.maturity_range],
            "n_violations": self.n_violations,
            "n_violations_retained": len(self.violations),
            "worst_violations": [v.as_dict() for v in self.violations[:10]],
        }

    def summary(self) -> str:
        """One line, for logs and report headers."""
        if self.arbitrage_free:
            head = (
                f"arbitrage-free on {self.n_checked} points "
                f"(min g={self.min_butterfly:.4f}, min dw/dT={self.min_calendar:.3e})"
            )
            if self.calendar_degenerate:
                return (
                    f"{head} but DEGENERATE: dw/dT falls to "
                    f"{self.min_calendar_relative:.2%} of theta/T, so local "
                    f"variance collapses"
                )
            return head
        failed = []
        if not self.butterfly_free:
            failed.append(f"butterfly (min g={self.min_butterfly:.3e})")
        if not self.calendar_free:
            failed.append(f"calendar (min dw/dT={self.min_calendar:.3e})")
        return f"ARBITRAGE: {', '.join(failed)}, {self.n_violations} points"


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
            inside the first quoted expiry out to :data:`DEFAULT_MAX_MATURITY`,
            because the surface gets used at maturities between and far past
            the quoted ones -- which is exactly where the interpolated term
            structure, rather than the fitted constraints, decides the answer.
        n_maturities: Grid points across maturity when defaulted.
        tolerance: Threshold below which a value counts as a violation.
        max_violations: Cap on *retained* violations, applied after sorting, so
            what survives is the worst points rather than the first found.

    Returns:
        An :class:`ArbitrageReport`.
    """
    if maturities is None:
        quoted = np.asarray(surface.params.maturities, dtype=float)
        maturities = np.linspace(
            max(quoted[0] * 0.5, 1e-4),
            max(quoted[-1] * 1.5, DEFAULT_MAX_MATURITY),
            n_maturities,
        )
    maturity_grid = np.asarray(maturities, dtype=float)
    moneyness_grid = np.linspace(-moneyness_range, moneyness_range, n_moneyness)

    violations: list[Violation] = []
    min_butterfly = np.inf
    min_calendar = np.inf
    min_calendar_relative = np.inf

    for maturity in maturity_grid:
        butterfly = np.asarray(surface.butterfly(moneyness_grid, maturity))
        calendar = np.asarray(surface.dw_dT(moneyness_grid, maturity))

        min_butterfly = min(min_butterfly, float(np.min(butterfly)))
        worst_calendar = float(np.min(calendar))
        min_calendar = min(min_calendar, worst_calendar)

        # Scale-free version: how dw/dT compares with the rate at which this
        # surface accumulates variance on average. Catches a flat term
        # structure, which passes the condition above while leaving local
        # variance at zero.
        average_rate = float(surface.theta(maturity)) / float(maturity)
        if average_rate > 0.0:
            min_calendar_relative = min(min_calendar_relative, worst_calendar / average_rate)

        for name, values, limit in (
            ("butterfly", butterfly, 0.0),
            ("calendar", calendar, tolerance),
        ):
            for index in np.flatnonzero(values < limit):
                violations.append(
                    Violation(
                        condition=name,
                        log_moneyness=float(moneyness_grid[index]),
                        maturity=float(maturity),
                        value=float(values[index]),
                    )
                )

    # Sort first, then truncate. Capping during collection keeps whichever
    # maturity was scanned first, which is not the same as keeping the worst:
    # on a badly arbitraged surface the retained worst was -0.939 while the
    # true minimum was -5.360, five times larger and at a maturity the cap had
    # already excluded. This list is what a reader uses to locate the problem,
    # so it has to hold the worst points, not the earliest ones.
    violations.sort(key=lambda v: v.value)
    n_violations = len(violations)
    violations = violations[:max_violations]

    report = ArbitrageReport(
        min_butterfly=float(min_butterfly),
        min_calendar=float(min_calendar),
        min_calendar_relative=float(min_calendar_relative),
        violations=tuple(violations),
        n_violations=n_violations,
        n_checked=int(maturity_grid.size * moneyness_grid.size),
        moneyness_range=moneyness_range,
        maturity_range=(float(maturity_grid[0]), float(maturity_grid[-1])),
    )
    _log.info("surface arbitrage check", extra={"result": report.summary()})
    return report


def local_variance_is_well_defined(report: ArbitrageReport) -> tuple[bool, str]:
    """Whether Dupire yields a *usable* local volatility on this surface.

    The Dupire denominator *is* the butterfly expression and its numerator *is*
    ``dw/dT``, so the two no-arbitrage conditions settle most of this. Stated as
    its own function because it is the precondition Phase 2b depends on, and
    phrasing it in those terms makes the dependency explicit rather than
    something a reader has to reconstruct.

    Degeneracy is reported as a failure even though it is not arbitrage. A
    surface whose total variance is flat in maturity has a local variance of
    exactly zero, which is non-negative, well defined, and useless. Calling
    that a pass would be answering a narrower question than the caller asked.
    """
    if report.arbitrage_free and not report.calendar_degenerate:
        return True, (
            "local variance is well defined and non-negative: the butterfly "
            "condition keeps the Dupire denominator positive and the calendar "
            "condition keeps its numerator non-negative"
        )

    if report.arbitrage_free:
        return False, (
            f"local variance is well defined but degenerate: dw/dT falls to "
            f"{report.min_calendar_relative:.2%} of the surface's own average "
            f"variance rate, so Dupire's numerator is effectively zero and the "
            f"local volatility it yields collapses towards zero. No arbitrage is "
            f"present -- the variance term structure is simply flat where it "
            f"should be rising."
        )

    problems = []
    if not report.butterfly_free:
        problems.append(
            f"the Dupire denominator vanishes or changes sign (min g = {report.min_butterfly:.3e})"
        )
    if not report.calendar_free:
        problems.append(f"the Dupire numerator is negative (min dw/dT = {report.min_calendar:.3e})")
    return False, "local variance is not well defined: " + "; and ".join(problems)
