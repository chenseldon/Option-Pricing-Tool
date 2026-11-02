"""Fitting Heston to a quoted surface, on the same objective as SSVI.

Why the objective matches SSVI's exactly
----------------------------------------
Weighted least squares on implied *volatility*, vega-weighted, with the weights
normalised per slice. Not because that is the only sensible choice, but because
it is the choice :mod:`otcpricer.models.svi_calibration` already made -- and the
only interesting question about a Heston fit here is how it compares to the
SSVI fit of the same quotes. Two models scored on two different objectives are
not comparable, and the comparison is the whole point.

Fitting on volatility rather than price matters for the same reason it did
there: price errors are dominated by the at-the-money options, which carry the
most premium, so a price objective quietly stops fitting the wings.

What the comparison showed: Heston fits worse, and predictably so
-----------------------------------------------------------------
Measured on the same quotes, same objective, same weights:

=====================  ========  =====  ============  =========  ===========  ======
chain                  expiries   pts   SSVI params   SSVI rms   Heston rms    ratio
=====================  ========  =====  ============  =========  ===========  ======
510050 (akshare)              3     35             6      0.587        0.644   1.10x
510300 (akshare)              3     42             6      0.654        0.730   1.12x
HO (CFFEX)                    6     93             9      0.460        0.648   1.41x
IO (CFFEX)                    6    121             9      0.670        0.924   1.38x
MO (CFFEX)                    6    130             9      1.384        1.447   1.05x
=====================  ========  =====  ============  =========  ===========  ======

Volatility points. The pattern is the parameter budget, not a defect: SSVI
carries three global parameters **plus one per expiry**, so a six-expiry chain
gives it nine against Heston's five, and its advantage grows exactly as the
term structure gets richer. On three expiries the two are within 12% of each
other; on six, Heston is 38-41% worse.

This is worth stating plainly because the temptation runs the other way.
**Heston is not bought here for accuracy -- it is strictly worse at describing
today's quotes.** What it provides is a *dynamic*: a forward smile that does
not flatten, and a variance that is a state variable rather than a function of
spot. Dupire reprices today's vanillas by construction and then implies a
forward smile known to be wrong; Heston gives up some of today's fit in
exchange for a forward smile that is at least a model's rather than an
artefact's. For a path-dependent product that is the trade worth making, and
for a vanilla it is not.

One consequence is practical: calibration takes 56-75 seconds per chain, so a
full 725-day panel sweep would be about 39 hours. Heston parameters are
therefore not swept across the panel the way the SSVI anchor is.

The Feller condition is not imposed
-----------------------------------
Both fits violate it, with margins of -3.36 and -1.52. Constraining it would
produce a worse fit to the same quotes and would not make the model more true;
what it means is that simulation needs the QE scheme. The margin is reported
alongside the fit so the consequence is visible rather than buried.

Why five parameters are harder to fit than they look
-----------------------------------------------------
``kappa`` and ``sigma`` trade off against each other almost freely -- a fast
mean reversion with a large vol-of-vol looks much like a slow one with a small
one over a year of quotes. So the objective has long flat valleys, and a
gradient method from a single start lands wherever it started. This fits from
several starting points and keeps the best, which is cheap here because one
objective evaluation is a few hundred COS prices.

How badly that bites was measured. On a synthetic surface generated from known
parameters, the best start recovered every one of them to four decimals at an
rms of 4.6e-06 volatility units -- and the next best start, which also reported
successful convergence, stopped at 4.2e-03, a thousand times worse, with
``kappa`` at 8.09 instead of 2.00. ``converged`` was True for both.

So this module reports three separate things rather than one: whether the
optimiser stopped cleanly, whether a *second* start corroborated the answer,
and whether comparable starts agreed on the parameters. Collapsing them would
hide exactly the failure that actually occurs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from otcpricer.core.exceptions import CalibrationError
from otcpricer.core.logging import get_logger
from otcpricer.engines.fourier import price_heston_european
from otcpricer.models.heston import HestonParams

__all__ = [
    "CALIBRATION_BOUNDS",
    "HestonCalibration",
    "calibrate_heston",
]

_log = get_logger(__name__)

#: Admissible ranges, as ``(low, high)`` per parameter.
#:
#: Deliberately wide on ``sigma``: the fits this data produces land near 2,
#: which a textbook bound of 1 would clip. Clipping it would improve the Feller
#: margin and worsen the fit, i.e. make the model look better behaved by making
#: it describe the market less well.
CALIBRATION_BOUNDS: dict[str, tuple[float, float]] = {
    "v0": (1e-4, 1.0),
    "kappa": (1e-2, 20.0),
    "theta": (1e-4, 1.0),
    "sigma": (1e-2, 5.0),
    "rho": (-0.95, 0.95),
}

#: Starting points, chosen to sit in different valleys rather than near each
#: other: slow/large-sigma, fast/small-sigma, and one in between.
_STARTS: tuple[tuple[float, float, float, float, float], ...] = (
    (0.04, 2.0, 0.04, 0.5, -0.5),
    (0.04, 8.0, 0.05, 1.8, -0.2),
    (0.06, 0.8, 0.08, 0.3, -0.7),
    (0.03, 4.0, 0.06, 1.2, 0.1),
)


@dataclass(frozen=True, slots=True)
class HestonCalibration:
    """A fitted Heston model and the evidence for trusting it.

    Attributes:
        params: The fitted parameters.
        rms_vol_error: Root-mean-square implied-volatility error.
        max_vol_error: Largest absolute implied-volatility error.
        n_points: Observations fitted.
        converged: Whether the optimiser reported success.
        message: The optimiser's own account.
        n_starts: How many starting points were tried.
        n_comparable_starts: Starts that reached within
            :data:`COMPARABLE_OBJECTIVE_FACTOR` of the best objective.
        parameter_spread: Largest relative disagreement in any parameter among
            those comparable starts. This is the identifiability measure, and
            it deliberately looks at *parameters* rather than at objective
            values: ``kappa`` and ``sigma`` trade off almost freely, so two
            starts can reach the same fit quality from quite different places,
            and that -- not a poor objective -- is the failure mode here.
        n_unidentifiable: Observations whose fitted model price could not be
            inverted to a volatility, excluded from ``rms_vol_error`` and
            counted here instead.
    """

    params: HestonParams
    rms_vol_error: float
    max_vol_error: float
    n_points: int
    converged: bool
    message: str
    n_starts: int
    n_comparable_starts: int
    parameter_spread: float
    n_unidentifiable: int = 0

    @property
    def rms_vol_points(self) -> float:
        """The error in volatility points, which is how it gets reported."""
        return self.rms_vol_error * 100.0

    @property
    def confirmed_by_multiple_starts(self) -> bool:
        """Whether more than one start reached a comparable fit.

        False does **not** mean the fit is wrong -- it means no second start
        corroborated it. Measured on synthetic data recovered to four decimals,
        one start of four reached rms 4.6e-06 while the next best managed
        4.2e-03, a thousand times worse: the fit was excellent and entirely
        unconfirmed. Those are different facts and are reported separately.
        """
        return self.n_comparable_starts > 1

    @property
    def ambiguous(self) -> bool:
        """Whether comparable starts disagreed on the parameters.

        This is the ``kappa``/``sigma`` trade-off showing up: a fast mean
        reversion with a large vol-of-vol looks much like a slow one with a
        small one over a year of quotes, so two starts can fit equally well
        from quite different places. Only meaningful when more than one start
        was comparable.
        """
        return self.confirmed_by_multiple_starts and self.parameter_spread >= 0.25

    @property
    def well_identified(self) -> bool:
        """Corroborated by a second start, and agreeing with it on parameters.

        Deliberately strict, and deliberately not the same as ``converged`` --
        which only says the optimiser stopped cleanly, and is true of a local
        minimum a thousand times worse than the global one.
        """
        return self.confirmed_by_multiple_starts and not self.ambiguous

    def as_dict(self) -> dict[str, object]:
        return {
            "params": self.params.as_dict(),
            "rms_vol_error": round(self.rms_vol_error, 8),
            "rms_vol_points": round(self.rms_vol_points, 5),
            "max_vol_error": round(self.max_vol_error, 8),
            "n_points": self.n_points,
            "converged": self.converged,
            "message": self.message,
            "n_starts": self.n_starts,
            "n_comparable_starts": self.n_comparable_starts,
            "parameter_spread": round(self.parameter_spread, 6),
            "confirmed_by_multiple_starts": self.confirmed_by_multiple_starts,
            "ambiguous": self.ambiguous,
            "well_identified": self.well_identified,
            "n_unidentifiable": self.n_unidentifiable,
            "note": (
                "Vega-weighted least squares on implied volatility, the same "
                "objective svi_calibration uses, so the two fits are directly "
                "comparable. Feller is not imposed: both calibrated fits "
                "violate it, and constraining it would fit the quotes worse "
                "without making the model more true."
            ),
        }


#: Residual charged, in volatility units, for a model price that cannot be
#: inverted to a volatility at all.
#:
#: One volatility point is a hundred times the fit error, so a parameter set
#: that breaks a point is firmly rejected -- but *proportionally*, point by
#: point. The alternative is what this module first did: return a single huge
#: objective value for the whole surface. That turns a locally bad trial into a
#: flat plateau with no gradient, and the optimiser simply stops. Penalising
#: per point keeps the descent direction informative.
UNIDENTIFIABLE_PENALTY = 1.0

#: How much worse than the best a start may be and still count as having
#: reached the same minimum, as a fraction of the best *volatility* error.
COMPARABLE_RELATIVE_SLACK = 0.10

#: And an absolute floor on that comparison, in volatility units.
#:
#: Both are needed. A purely relative bar is meaningless when the fit is
#: near-exact: on synthetic data recovered to four decimals the objective is
#: around 1e-12, and "within 10%" of that excludes a start which agreed to six
#: decimal places. 0.05 volatility points is twenty times below the fit error
#: on real quotes, so on real data the relative bar is the one that binds.
COMPARABLE_ABSOLUTE_SLACK = 5e-4


def _implied_vol_from_price(
    price: float, spot: float, strike: float, maturity: float, rate: float
) -> float:
    """Invert a call price to a Black-Scholes volatility.

    Uses the project's own solver, so a Heston fit is scored in the same units,
    by the same inversion, as the SSVI fit it is being compared against. That
    solver *refuses* to invert where vega underflows -- a deep out-of-the-money
    premium of 3e-08 does not pin down a volatility, and it says so rather than
    returning a number. ``nan`` here means exactly that, and the objective
    charges :data:`UNIDENTIFIABLE_PENALTY` for it.
    """
    from otcpricer.models.bsm import implied_volatility

    try:
        return float(implied_volatility("c", price, spot, strike, maturity, rate, q=0.0))
    except Exception:
        return float("nan")


def calibrate_heston(
    maturities: list[float],
    log_moneyness: list[NDArray[np.float64]],
    implied_vols: list[NDArray[np.float64]],
    vegas: list[NDArray[np.float64]] | None = None,
    *,
    forwards: list[float] | None = None,
    rate: float = 0.0,
    terms: int = 256,
    starts: int = len(_STARTS),
) -> HestonCalibration:
    """Fit Heston to a set of implied-volatility slices.

    Args:
        maturities: One year fraction per slice.
        log_moneyness: ``log(K / F)`` per slice.
        implied_vols: Observed Black volatilities per slice.
        vegas: Fit weights per slice; vega is the sensible choice and equal
            weights are used if omitted. Matches ``calibrate_ssvi``.
        forwards: Forward per slice. Defaults to 1.0, which makes the fit
            scale-free -- strikes are then ``exp(k)`` and spot is 1.
        rate: Discount rate used when pricing inside the objective. The fit is
            in volatility space, so this only has to be self-consistent
            between the pricing and the inversion, not correct.
        terms: COS terms inside the objective. Lower than the engine default
            because this runs thousands of times and the objective only needs
            to resolve volatility differences far below the fit error -- the
            final parameters are re-scored at full precision.
        starts: How many starting points to try.

    Returns:
        A :class:`HestonCalibration`.

    Raises:
        CalibrationError: on inconsistent inputs or an empty observation set.
    """
    from scipy.optimize import minimize

    if not (len(maturities) == len(log_moneyness) == len(implied_vols)):
        raise CalibrationError(
            f"got {len(maturities)} maturities, {len(log_moneyness)} moneyness "
            f"slices and {len(implied_vols)} volatility slices"
        )
    if not maturities:
        raise CalibrationError("no slices to fit")

    T = np.asarray(maturities, dtype=float)
    if np.any(T <= 0.0):
        raise CalibrationError(f"maturities must be positive, got {maturities}")

    k_slices = [np.asarray(k, dtype=float) for k in log_moneyness]
    vol_slices = [np.asarray(v, dtype=float) for v in implied_vols]
    for index, (k, v) in enumerate(zip(k_slices, vol_slices, strict=True)):
        if k.shape != v.shape:
            raise CalibrationError(f"slice {index}: {k.size} strikes, {v.size} volatilities")
        if k.size == 0:
            raise CalibrationError(f"slice {index} is empty")

    n_points = int(sum(v.size for v in vol_slices))

    if vegas is None:
        weight_slices = [np.ones_like(v) for v in vol_slices]
    else:
        weight_slices = [np.asarray(w, dtype=float) for w in vegas]
        for index, (w, v) in enumerate(zip(weight_slices, vol_slices, strict=True)):
            if w.shape != v.shape:
                raise CalibrationError(f"slice {index}: weights do not match observations")
    # Normalised per slice, so no expiry dominates by having more quotes.
    normalised = [w / max(float(np.sum(w)), 1e-30) for w in weight_slices]

    spots = [1.0] * len(T) if forwards is None else list(forwards)

    def model_vols(params: HestonParams, cos_terms: int) -> list[NDArray[np.float64]]:
        out = []
        for maturity, k, forward in zip(T, k_slices, spots, strict=True):
            # Priced off the forward, so the moneyness coordinate matches the
            # one the quotes were expressed in.
            spot = forward * float(np.exp(-rate * maturity))
            strikes = forward * np.exp(k)
            vols = np.empty(k.size, dtype=float)
            for index, strike in enumerate(strikes):
                try:
                    price = price_heston_european(
                        params,
                        spot=spot,
                        strike=float(strike),
                        maturity=float(maturity),
                        rate=rate,
                        flag="c",
                        terms=cos_terms,
                    ).price
                except Exception:
                    vols[index] = float("nan")
                    continue
                vols[index] = _implied_vol_from_price(
                    price, spot, float(strike), float(maturity), rate
                )
            out.append(vols)
        return out

    def objective(x: NDArray[np.float64], cos_terms: int = terms) -> float:
        try:
            params = HestonParams(v0=x[0], kappa=x[1], theta=x[2], sigma=x[3], rho=x[4])
        except Exception:
            return 1e6
        total = 0.0
        for observed, weight, modelled in zip(
            vol_slices, normalised, model_vols(params, cos_terms), strict=True
        ):
            residual = modelled - observed
            # Per point, not per surface. See UNIDENTIFIABLE_PENALTY.
            residual = np.where(np.isfinite(residual), residual, UNIDENTIFIABLE_PENALTY)
            total += float(np.sum(weight * residual * residual))
        return total / len(vol_slices)

    bounds = [CALIBRATION_BOUNDS[name] for name in ("v0", "kappa", "theta", "sigma", "rho")]

    results = []
    for start in _STARTS[: max(1, starts)]:
        outcome = minimize(
            objective,
            np.array(start, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-12},
        )
        results.append(outcome)

    results.sort(key=lambda r: float(r.fun))
    best = results[0]
    objectives = np.array([float(r.fun) for r in results])

    # Identifiability is measured on the parameters, among the starts that
    # actually reached a comparable fit. An earlier version compared objective
    # *values* relatively, which on a near-exact fit divides by ~1e-12 and
    # reported a spread of 850,000 for a fit that had recovered every parameter
    # to four decimals. The objective scale says nothing useful here.
    # Compared in volatility units -- the objective is a weighted mean squared
    # volatility error, so its square root is on the same scale as the reported
    # rms -- and with both a relative and an absolute slack. See the constants.
    best_vol_error = float(np.sqrt(max(objectives.min(), 0.0)))
    bar = max(
        best_vol_error * (1.0 + COMPARABLE_RELATIVE_SLACK),
        best_vol_error + COMPARABLE_ABSOLUTE_SLACK,
    )
    comparable = [r for r in results if float(np.sqrt(max(float(r.fun), 0.0))) <= bar]
    if len(comparable) > 1:
        stacked = np.array([np.asarray(r.x, dtype=float) for r in comparable])
        scale = np.maximum(np.abs(stacked).max(axis=0), 1e-8)
        spread = float(np.max((stacked.max(axis=0) - stacked.min(axis=0)) / scale))
    else:
        spread = 0.0

    params = HestonParams(
        v0=float(best.x[0]),
        kappa=float(best.x[1]),
        theta=float(best.x[2]),
        sigma=float(best.x[3]),
        rho=float(best.x[4]),
    )

    # Re-score at full COS precision, so the reported error is not the
    # objective's own cheaper approximation of itself.
    modelled = model_vols(params, 1024)
    residuals = np.concatenate([m - o for m, o in zip(modelled, vol_slices, strict=True)])
    finite = residuals[np.isfinite(residuals)]
    n_unidentifiable = int(residuals.size - finite.size)
    # Reported on the points that can be scored, with the count of those that
    # cannot carried alongside -- averaging a penalty into an error figure
    # would make the fit look worse than it is in a way nobody could unpick.
    rms = float(np.sqrt(np.mean(finite**2))) if finite.size else float("nan")
    worst = float(np.max(np.abs(finite))) if finite.size else float("nan")

    _log.info(
        "heston fit: rms %.4f vol points, max %.4f, feller %.4f, "
        "%d/%d comparable starts, parameter spread %.3f",
        rms * 100.0,
        worst * 100.0,
        params.feller_margin,
        len(comparable),
        len(results),
        spread,
    )

    return HestonCalibration(
        params=params,
        rms_vol_error=rms,
        max_vol_error=worst,
        n_points=n_points,
        converged=bool(best.success),
        message=str(best.message),
        n_starts=len(results),
        n_comparable_starts=len(comparable),
        parameter_spread=spread,
        n_unidentifiable=n_unidentifiable,
    )
