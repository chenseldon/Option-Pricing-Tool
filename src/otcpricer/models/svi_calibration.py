"""Constrained SSVI calibration.

The no-arbitrage conditions are handed to the optimiser as constraints rather
than checked after the fact. That distinction is the whole point of the module:
a surface that violates them is not a slightly worse fit, it is unusable for
the purpose, because the Dupire denominator it feeds *is* the butterfly
expression and would go negative. Fitting freely and rejecting afterwards would
mean discarding the work and having nothing to fall back on.

What is constrained
-------------------
*Butterfly*, from Gatheral and Jacquier's sufficient conditions, imposed at
every fitted maturity:

    theta * phi(theta) * (1 + |rho|) < 4
    theta * phi(theta)^2 * (1 + |rho|) <= 4

*Calendar*: ``theta`` non-decreasing in maturity. Expressed by optimising over
non-negative *increments* rather than the levels themselves, so monotonicity
holds identically at every point the optimiser visits -- it cannot be violated
even transiently, which keeps the search inside the admissible set throughout.

Fitting in volatility, not variance
-----------------------------------
The objective is weighted least squares on implied *volatility*. Volatility is
what the market quotes and what an error budget is expressed in, so a residual
of "half a vol point" is directly interpretable; total-variance residuals mix
maturities and are not. Weights are each quote's vega, so the fit is pulled
towards the quotes that actually carry volatility information rather than
treating a one-tick wing as equal to an at-the-money quote.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

from otcpricer.core.exceptions import CalibrationError
from otcpricer.core.logging import get_logger
from otcpricer.models.svi import MAX_ETA, SSVIParams, SSVISurface, power_law_phi

__all__ = ["CalibrationResult", "SliceFitQuality", "calibrate_ssvi"]

_log = get_logger(__name__)

#: Butterfly bound from the sufficient conditions, with a small safety margin so
#: the optimiser settles strictly inside rather than exactly on the boundary.
_BUTTERFLY_LIMIT = 4.0
_BUTTERFLY_SAFETY = 1e-6


@dataclass(frozen=True, slots=True)
class SliceFitQuality:
    """How well one expiry was fitted, in volatility points."""

    maturity: float
    n_points: int
    rms_vol_error: float
    max_vol_error: float
    atm_vol_fitted: float
    atm_vol_observed: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "maturity": round(self.maturity, 6),
            "n_points": self.n_points,
            "rms_vol_error": round(self.rms_vol_error, 8),
            "max_vol_error": round(self.max_vol_error, 8),
            "atm_vol_fitted": round(self.atm_vol_fitted, 6),
            "atm_vol_observed": round(self.atm_vol_observed, 6),
        }


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """A fitted surface together with the evidence for trusting it."""

    params: SSVIParams
    surface: SSVISurface
    rms_vol_error: float
    max_vol_error: float
    n_points: int
    slices: tuple[SliceFitQuality, ...]
    converged: bool
    message: str

    @property
    def butterfly_margin(self) -> float:
        """Headroom in the butterfly condition; must stay positive."""
        return self.params.butterfly_margin

    @property
    def calendar_margin(self) -> float:
        """Smallest increment in the variance term structure."""
        return self.params.calendar_margin

    def as_dict(self) -> dict[str, object]:
        """Flat dict for API responses and reports."""
        return {
            "params": self.params.as_dict(),
            "rms_vol_error": round(self.rms_vol_error, 8),
            "max_vol_error": round(self.max_vol_error, 8),
            "n_points": self.n_points,
            "converged": self.converged,
            "message": self.message,
            "butterfly_margin": round(self.butterfly_margin, 8),
            "calendar_margin": round(self.calendar_margin, 10),
            "slices": [s.as_dict() for s in self.slices],
        }


def _unpack(x: np.ndarray, n_maturities: int) -> tuple[float, float, float, np.ndarray]:
    """Decode the optimiser vector into ``(rho, eta, gamma, thetas)``.

    ``theta`` is carried as a first level plus non-negative increments, so the
    monotonicity that constitutes the calendar condition is structural rather
    than a constraint the optimiser could momentarily violate.
    """
    rho, eta, gamma = float(x[0]), float(x[1]), float(x[2])
    first = x[3]
    increments = x[4 : 3 + n_maturities]
    thetas = first + np.concatenate(([0.0], np.cumsum(increments)))
    return rho, eta, gamma, thetas


def _butterfly_values(x: np.ndarray, n_maturities: int) -> np.ndarray:
    """The two butterfly expressions at every maturity, for the constraint."""
    rho, eta, gamma, thetas = _unpack(x, n_maturities)
    phi = power_law_phi(thetas, eta, gamma)
    factor = 1.0 + abs(rho)
    return np.concatenate([thetas * phi * factor, thetas * phi**2 * factor])


def calibrate_ssvi(
    maturities: list[float] | np.ndarray,
    log_moneyness: list[np.ndarray],
    implied_vols: list[np.ndarray],
    weights: list[np.ndarray] | None = None,
    *,
    initial_rho: float = -0.3,
    initial_eta: float = 1.0,
    initial_gamma: float = 0.4,
    max_iterations: int = 400,
) -> CalibrationResult:
    """Fit an SSVI surface to implied volatilities, free of static arbitrage.

    Args:
        maturities: Expiry year fractions, increasing. One per slice.
        log_moneyness: Per slice, ``log(K / F)`` of each observation.
        implied_vols: Per slice, the observed implied volatilities.
        weights: Per slice, fit weights -- vega is the sensible choice. Equal
            weights are used if omitted.
        initial_rho: Starting slope asymmetry. Negative by default: equity
            index smiles skew to the downside, so this starts the search in the
            right half of the space.
        initial_eta: Starting curvature scale.
        initial_gamma: Starting curvature decay.
        max_iterations: Optimiser iteration cap.

    Returns:
        A :class:`CalibrationResult` with the surface, per-slice fit quality in
        volatility points, and the arbitrage margins.

    Raises:
        CalibrationError: if the inputs are inconsistent, there is too little
            data to identify the parameters, or the optimiser cannot find an
            arbitrage-free fit.
    """
    T = np.asarray(maturities, dtype=float)
    n = T.size
    if n == 0:
        raise CalibrationError("no maturities supplied")
    if not np.all(np.diff(T) > 0):
        raise CalibrationError(f"maturities must be strictly increasing, got {T.tolist()}")
    if len(log_moneyness) != n or len(implied_vols) != n:
        raise CalibrationError(
            f"got {n} maturities but {len(log_moneyness)} moneyness arrays and "
            f"{len(implied_vols)} volatility arrays"
        )

    k_slices = [np.asarray(k, dtype=float) for k in log_moneyness]
    vol_slices = [np.asarray(v, dtype=float) for v in implied_vols]
    for index, (k, v) in enumerate(zip(k_slices, vol_slices, strict=True)):
        if k.shape != v.shape:
            raise CalibrationError(
                f"slice {index}: {k.size} moneyness values but {v.size} volatilities"
            )
        if k.size == 0:
            raise CalibrationError(f"slice {index} has no observations")

    if weights is None:
        weight_slices = [np.ones_like(v) for v in vol_slices]
    else:
        weight_slices = [np.asarray(w, dtype=float) for w in weights]
        for index, (w, v) in enumerate(zip(weight_slices, vol_slices, strict=True)):
            if w.shape != v.shape:
                raise CalibrationError(f"slice {index}: weights do not match observations")

    total_points = sum(k.size for k in k_slices)
    n_parameters = 3 + n
    if total_points < n_parameters:
        raise CalibrationError(
            f"{total_points} observations cannot identify {n_parameters} parameters "
            f"(rho, eta, gamma and one at-the-money variance per expiry)"
        )

    # Normalise weights so the objective is comparable across inputs.
    normalised = [w / max(float(np.sum(w)), 1e-30) for w in weight_slices]

    # Start theta at the observed at-the-money variance, which is nearly right
    # by construction and leaves the optimiser only the shape to find.
    observed_theta = np.array(
        [
            float(np.interp(0.0, k[np.argsort(k)], (v * v * t)[np.argsort(k)]))
            for k, v, t in zip(k_slices, vol_slices, T, strict=True)
        ]
    )
    # Enforce a strictly increasing start so the increment parameterisation
    # begins inside the admissible set.
    for i in range(1, n):
        observed_theta[i] = max(observed_theta[i], observed_theta[i - 1] * 1.000001)

    x0 = np.concatenate(
        [
            [initial_rho, initial_eta, initial_gamma, observed_theta[0]],
            np.diff(observed_theta) if n > 1 else [],
        ]
    )

    bounds = [
        (-0.999, 0.999),  # rho
        (1e-4, MAX_ETA),  # eta
        (1e-3, 0.5),  # gamma, kept inside the analysed range
        (1e-8, 10.0),  # first theta
    ] + [(0.0, 10.0)] * max(n - 1, 0)  # non-negative increments: calendar condition

    def objective(x: np.ndarray) -> float:
        rho, eta, gamma, thetas = _unpack(x, n)
        try:
            params = SSVIParams(
                rho=rho,
                eta=eta,
                gamma=gamma,
                maturities=tuple(T.tolist()),
                thetas=tuple(thetas.tolist()),
            )
            surface = SSVISurface(params)
        except Exception:  # noqa: BLE001 - an inadmissible point, not a failure
            return 1e6

        total = 0.0
        for k, observed, weight, maturity in zip(k_slices, vol_slices, normalised, T, strict=True):
            fitted = surface.implied_vol(k, maturity)
            residual = fitted - observed
            total += float(np.sum(weight * residual * residual))
        return total / n

    constraints = [
        NonlinearConstraint(
            lambda x: _butterfly_values(x, n),
            -np.inf,
            _BUTTERFLY_LIMIT - _BUTTERFLY_SAFETY,
        )
    ]

    outcome = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": max_iterations, "ftol": 1e-12},
    )

    rho, eta, gamma, thetas = _unpack(np.asarray(outcome.x), n)
    params = SSVIParams(
        rho=rho,
        eta=eta,
        gamma=gamma,
        maturities=tuple(T.tolist()),
        thetas=tuple(thetas.tolist()),
    )
    surface = SSVISurface(params)

    if params.butterfly_margin <= 0.0:
        raise CalibrationError(
            f"calibration ended on a surface with butterfly arbitrage "
            f"(margin {params.butterfly_margin:.3e}). The Dupire denominator would "
            f"be non-positive, so this surface cannot produce a local volatility."
        )

    # Per-slice quality, in volatility points.
    slice_quality: list[SliceFitQuality] = []
    all_errors: list[np.ndarray] = []
    for k, observed, maturity in zip(k_slices, vol_slices, T, strict=True):
        fitted = surface.implied_vol(k, maturity)
        errors = np.abs(fitted - observed)
        all_errors.append(errors)
        order = np.argsort(k)
        slice_quality.append(
            SliceFitQuality(
                maturity=float(maturity),
                n_points=int(k.size),
                rms_vol_error=float(np.sqrt(np.mean(errors**2))),
                max_vol_error=float(np.max(errors)),
                atm_vol_fitted=float(surface.implied_vol(0.0, maturity)),
                atm_vol_observed=float(np.interp(0.0, k[order], observed[order])),
            )
        )

    pooled = np.concatenate(all_errors)
    result = CalibrationResult(
        params=params,
        surface=surface,
        rms_vol_error=float(np.sqrt(np.mean(pooled**2))),
        max_vol_error=float(np.max(pooled)),
        n_points=int(pooled.size),
        slices=tuple(slice_quality),
        converged=bool(outcome.success),
        message=str(outcome.message),
    )
    _log.info(
        "SSVI calibrated",
        extra={
            "rho": rho,
            "eta": eta,
            "gamma": gamma,
            "rms_vol_error": result.rms_vol_error,
            "butterfly_margin": result.butterfly_margin,
            "converged": result.converged,
        },
    )
    return result
