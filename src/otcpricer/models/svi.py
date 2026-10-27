"""SVI and SSVI volatility surfaces.

Why a parametric surface rather than interpolation
--------------------------------------------------
The purpose of this module is Dupire local volatility, and the Dupire
denominator is *identically* Gatheral's butterfly function ``g(k)`` -- verified
symbolically, see ``docs/models``. So:

* **butterfly-free surface** implies a positive Dupire denominator
* **calendar-free surface** implies a non-negative numerator ``dw/dT``

An arbitrage-free surface therefore yields a non-negative local variance *by
construction*, not by clamping a negative one. Feed the Dupire formula raw or
loosely interpolated quotes instead and it produces negative local variances
wherever arbitrage has crept in, which is why the usual advice is that building
the clean surface is most of the work.

SSVI (Gatheral and Jacquier, *Arbitrage-free SVI volatility surfaces*) is the
smallest parameterisation that makes those conditions checkable in closed form:

    w(k, theta) = (theta / 2) * [1 + rho * phi(theta) * k
                                 + sqrt((phi(theta) * k + rho)^2 + 1 - rho^2)]

with ``w`` the total implied variance ``sigma^2 * T``, ``k = log(K / F)``,
``theta`` the at-the-money total variance for that expiry, and ``phi`` a
curvature function. Three global shape parameters (``rho``, ``eta``, ``gamma``)
plus one ``theta`` per expiry.

The trade is deliberate: SSVI fits *less* tightly than an independent SVI per
slice, because far fewer parameters are available. What it buys is that calendar
arbitrage is structurally impossible rather than something to check for
afterwards. For a surface whose purpose is to be differentiated, that is the
right side of the trade.

Analytic derivatives
--------------------
``dw/dk``, ``d2w/dk2`` and ``dw/dT`` are all closed form here, verified against
sympy and against finite differences in the tests. Phase 2b's Dupire formula
consumes them directly -- the entire reason not to differentiate prices
numerically, which is unstable enough to be the standard warning in the
literature.

One structural result worth noting: ``d2w/dk2 = theta * psi^2 * (1 - rho^2) /
(2 * R^3)`` is strictly positive for ``|rho| < 1``, so an SSVI slice is always
convex in log-moneyness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

from otcpricer.core.exceptions import ParameterError
from otcpricer.core.validation import check_in_range, check_positive

__all__ = [
    "MAX_ETA",
    "SSVIParams",
    "SSVISurface",
    "SVISliceParams",
    "butterfly_g",
    "power_law_phi",
    "power_law_phi_derivative",
    "raw_svi_total_variance",
]

#: Upper bound on the curvature scale. The butterfly condition caps
#: ``eta * (1 + |rho|)`` at 4, so eta alone cannot usefully exceed it.
MAX_ETA = 8.0

#: Floor on total variance, to keep divisions finite at the front of the curve.
_MIN_TOTAL_VARIANCE = 1e-12


# ---------------------------------------------------------------------------
# Raw SVI, single slice
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SVISliceParams:
    """Raw SVI parameters for one expiry.

    ``w(k) = a + b * [rho * (k - m) + sqrt((k - m)^2 + sigma^2)]``

    Attributes:
        a: Vertical level of total variance.
        b: Slope scale; controls the wings jointly with ``rho``.
        rho: Slope asymmetry, in ``(-1, 1)``. Negative tilts the smile into a
            downside skew, which is the usual equity shape.
        m: Horizontal shift of the smile's minimum.
        sigma: Curvature at the minimum; larger is flatter.
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        check_in_range(self.rho, "rho", -1.0, 1.0)
        if self.b < 0.0:
            raise ParameterError(f"SVI b must be non-negative, got {self.b}")
        if self.sigma <= 0.0:
            raise ParameterError(f"SVI sigma must be positive, got {self.sigma}")


def raw_svi_total_variance(k: float | np.ndarray, params: SVISliceParams) -> float | np.ndarray:
    """Total implied variance of a raw SVI slice at log-moneyness ``k``."""
    shifted = np.asarray(k, dtype=float) - params.m
    return np.asarray(
        params.a + params.b * (params.rho * shifted + np.sqrt(shifted * shifted + params.sigma**2))
    )


# ---------------------------------------------------------------------------
# SSVI curvature function
# ---------------------------------------------------------------------------


def power_law_phi(theta: float | np.ndarray, eta: float, gamma: float) -> np.ndarray:
    """Power-law curvature function ``phi(theta) = eta / (theta^gamma * (1+theta)^(1-gamma))``.

    The choice Gatheral and Jacquier analyse, and the one with tractable
    no-arbitrage conditions. Note that ``theta * phi(theta)`` -- the quantity
    the butterfly condition bounds -- equals ``eta * (theta / (1 + theta))^(1 -
    gamma)``, which is increasing in ``theta`` and bounded above by ``eta``. So
    constraining ``eta * (1 + |rho|) <= 4`` makes the butterfly condition hold at
    *every* maturity rather than only at the fitted ones.
    """
    theta_safe = np.maximum(np.asarray(theta, dtype=float), _MIN_TOTAL_VARIANCE)
    return np.asarray(eta / (theta_safe**gamma * (1.0 + theta_safe) ** (1.0 - gamma)))


def power_law_phi_derivative(theta: float | np.ndarray, eta: float, gamma: float) -> np.ndarray:
    """``d phi / d theta`` for :func:`power_law_phi`.

    From ``log phi = log eta - gamma * log theta + (gamma - 1) * log(1 + theta)``,
    so ``phi' = phi * [-gamma / theta + (gamma - 1) / (1 + theta)]``. Needed for
    ``dw/dT``, and therefore for Dupire's numerator.
    """
    theta_safe = np.maximum(np.asarray(theta, dtype=float), _MIN_TOTAL_VARIANCE)
    phi = power_law_phi(theta_safe, eta, gamma)
    return np.asarray(phi * (-gamma / theta_safe + (gamma - 1.0) / (1.0 + theta_safe)))


# ---------------------------------------------------------------------------
# Arbitrage diagnostics
# ---------------------------------------------------------------------------


def butterfly_g(
    w: float | np.ndarray,
    dw_dk: float | np.ndarray,
    d2w_dk2: float | np.ndarray,
    k: float | np.ndarray,
) -> np.ndarray:
    """Gatheral's butterfly function ``g(k)``; positive means no butterfly arbitrage.

    ``g = (1 - k * w' / (2w))^2 - (w' / 2)^2 * (1 / w + 1 / 4) + w'' / 2``

    This same expression is the denominator of the Dupire local-variance
    formula, which is why enforcing ``g > 0`` is what guarantees a well-defined
    local volatility. Verified symbolically; see ``docs/models``.
    """
    w_arr = np.maximum(np.asarray(w, dtype=float), _MIN_TOTAL_VARIANCE)
    first = np.asarray(dw_dk, dtype=float)
    second = np.asarray(d2w_dk2, dtype=float)
    k_arr = np.asarray(k, dtype=float)

    term = 1.0 - k_arr * first / (2.0 * w_arr)
    return np.asarray(term * term - (first * first / 4.0) * (1.0 / w_arr + 0.25) + second / 2.0)


# ---------------------------------------------------------------------------
# SSVI surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SSVIParams:
    """Fitted SSVI parameters.

    Attributes:
        rho: Global slope asymmetry, in ``(-1, 1)``.
        eta: Curvature scale.
        gamma: Curvature decay exponent, in ``(0, 0.5]``.
        maturities: Expiry year fractions, increasing.
        thetas: At-the-money total variance at each maturity, non-decreasing.
            Monotonicity here *is* the calendar-arbitrage condition.
    """

    rho: float
    eta: float
    gamma: float
    maturities: tuple[float, ...]
    thetas: tuple[float, ...]

    def __post_init__(self) -> None:
        check_in_range(self.rho, "rho", -1.0, 1.0)
        check_positive(self.eta, "eta")
        check_in_range(self.gamma, "gamma", 0.0, 0.5)
        if len(self.maturities) != len(self.thetas):
            raise ParameterError(
                f"got {len(self.maturities)} maturities but {len(self.thetas)} thetas"
            )
        if not self.maturities:
            raise ParameterError("SSVI needs at least one maturity")
        if any(t <= 0.0 for t in self.maturities):
            raise ParameterError("maturities must be positive")
        if list(self.maturities) != sorted(self.maturities):
            raise ParameterError("maturities must be increasing")
        if any(t < 0.0 for t in self.thetas):
            raise ParameterError("at-the-money total variance cannot be negative")

    @property
    def butterfly_margin(self) -> float:
        """Headroom in the butterfly condition; positive means satisfied.

        Both sufficient conditions of Gatheral and Jacquier are checked at every
        fitted maturity: ``theta * phi * (1 + |rho|) < 4`` and
        ``theta * phi^2 * (1 + |rho|) <= 4``. The returned value is the smallest
        slack across both.
        """
        theta = np.array(self.thetas, dtype=float)
        phi = power_law_phi(theta, self.eta, self.gamma)
        factor = 1.0 + abs(self.rho)
        return float(min(np.min(4.0 - theta * phi * factor), np.min(4.0 - theta * phi**2 * factor)))

    @property
    def calendar_margin(self) -> float:
        """Smallest increment in ``theta``; non-negative means no calendar arbitrage."""
        if len(self.thetas) < 2:
            return float("inf")
        return float(np.min(np.diff(np.array(self.thetas, dtype=float))))

    def as_dict(self) -> dict[str, object]:
        """Flat dict for API responses and reports."""
        return {
            "rho": round(self.rho, 8),
            "eta": round(self.eta, 8),
            "gamma": round(self.gamma, 8),
            "maturities": [round(t, 8) for t in self.maturities],
            "thetas": [round(t, 10) for t in self.thetas],
            "atm_vols": [
                round(float(np.sqrt(theta / t)), 6)
                for t, theta in zip(self.maturities, self.thetas, strict=True)
            ],
            "butterfly_margin": round(self.butterfly_margin, 8),
            "calendar_margin": round(self.calendar_margin, 10),
        }


class SSVISurface:
    """An SSVI surface, with analytic derivatives.

    The at-the-money variance term structure is interpolated with a monotone
    cubic (PCHIP) through the fitted points and the origin, where total variance
    must vanish. Monotone cubic rather than linear for a specific reason: Dupire
    needs ``dw/dT``, and piecewise-linear ``theta`` would make it piecewise
    *constant*, putting a jump in local volatility at every expiry boundary.
    PCHIP is C-1 and preserves the monotonicity that is the calendar-arbitrage
    condition, so it keeps both properties at once.
    """

    def __init__(self, params: SSVIParams) -> None:
        self.params = params
        maturities = np.asarray(params.maturities, dtype=float)
        thetas = np.asarray(params.thetas, dtype=float)

        # Anchor at the origin: no time, no variance.
        knots_t = np.concatenate(([0.0], maturities))
        knots_theta = np.concatenate(([0.0], thetas))
        self._last_maturity = float(maturities[-1])

        # Beyond the last quoted expiry, extend theta linearly rather than
        # letting a cubic wander off, at the rate variance actually accumulated
        # over the last quoted interval.
        #
        # The rate has to be chosen carefully, because dw/dT is Dupire's
        # numerator and two obvious choices are both wrong.
        #
        # The average rate theta/T leaves a kink: the analytic derivative
        # disagreed with a central finite difference by 1.04e-03 exactly at the
        # last maturity while matching to 1e-9 everywhere else -- a jump in
        # local volatility at the edge of the quoted surface.
        #
        # PCHIP's own slope at that knot is continuous but can be *zero*.
        # PCHIP computes an endpoint derivative from a one-sided three-point
        # formula and clamps it to zero whenever its sign disagrees with the
        # adjacent secant, which happens whenever the variance term structure
        # is concave enough at the end -- a downward-sloping volatility term
        # structure, which is ordinary in a stressed market. Measured on
        # maturities (0.25, 1.0) with thetas (0.05, 0.10): PCHIP gives exactly
        # 0.000000 where the realised rate is 0.066667. Total variance then
        # goes flat forever past the last expiry, so implied vol decays as
        # 1/sqrt(T) and local variance is identically zero out there. Nothing
        # flags it, because flat total variance is calendar-arbitrage-*free*.
        # For a snowball running one to three years off a surface whose last
        # expiry is five months, that is the entire pricing range.
        #
        # So the rate is the last quoted secant, and the interpolant's final
        # derivative is overridden to match it, which keeps theta continuously
        # differentiable across the boundary. The segment stays monotone -- and
        # so stays free of calendar arbitrage -- because a cubic Hermite
        # segment is monotone when both end derivatives lie in [0, 3s], and
        # here one of them *is* s while PCHIP guarantees the other.
        slopes = PchipInterpolator(knots_t, knots_theta).derivative()(knots_t)
        self._tail_rate = float((knots_theta[-1] - knots_theta[-2]) / (knots_t[-1] - knots_t[-2]))
        slopes[-1] = self._tail_rate
        self._theta_curve = CubicHermiteSpline(knots_t, knots_theta, slopes, extrapolate=True)
        self._theta_slope = self._theta_curve.derivative()

    # -- term structure ---------------------------------------------------

    def theta(self, T: float | np.ndarray) -> np.ndarray:
        """At-the-money total variance at maturity ``T``."""
        T_arr = np.asarray(T, dtype=float)
        interior = np.asarray(self._theta_curve(np.minimum(T_arr, self._last_maturity)))
        beyond = T_arr > self._last_maturity
        if np.any(beyond):
            extra = (T_arr - self._last_maturity) * self._tail_rate
            interior = np.where(beyond, interior + extra, interior)
        return np.asarray(np.maximum(interior, _MIN_TOTAL_VARIANCE))

    def dtheta_dT(self, T: float | np.ndarray) -> np.ndarray:  # noqa: N802
        """``d theta / dT``, the term-structure slope."""
        T_arr = np.asarray(T, dtype=float)
        slope = np.asarray(self._theta_slope(np.minimum(T_arr, self._last_maturity)))
        return np.asarray(np.where(T_arr > self._last_maturity, self._tail_rate, slope))

    # -- surface and its derivatives --------------------------------------

    def _pieces(
        self, k: float | np.ndarray, T: float | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(theta, psi, u, R)``, the shared intermediates."""
        k_arr = np.asarray(k, dtype=float)
        theta = self.theta(T)
        psi = power_law_phi(theta, self.params.eta, self.params.gamma)
        u = psi * k_arr + self.params.rho
        R = np.sqrt(u * u + 1.0 - self.params.rho**2)
        return theta, psi, u, R

    def total_variance(self, k: float | np.ndarray, T: float | np.ndarray) -> np.ndarray:
        """Total implied variance ``w = sigma^2 * T``."""
        theta, psi, u, R = self._pieces(k, T)
        return np.asarray(
            (theta / 2.0) * (1.0 + self.params.rho * psi * np.asarray(k, dtype=float) + R)
        )

    def dw_dk(self, k: float | np.ndarray, T: float | np.ndarray) -> np.ndarray:
        """``dw/dk = (theta * psi / 2) * (rho + u / R)``."""
        theta, psi, u, R = self._pieces(k, T)
        return np.asarray((theta * psi / 2.0) * (self.params.rho + u / R))

    def d2w_dk2(self, k: float | np.ndarray, T: float | np.ndarray) -> np.ndarray:
        """``d2w/dk2 = theta * psi^2 * (1 - rho^2) / (2 R^3)``.

        Strictly positive for ``|rho| < 1``: an SSVI slice is always convex.
        """
        theta, psi, _, R = self._pieces(k, T)
        return np.asarray(theta * psi**2 * (1.0 - self.params.rho**2) / (2.0 * R**3))

    def dw_dT(self, k: float | np.ndarray, T: float | np.ndarray) -> np.ndarray:  # noqa: N802
        """``dw/dT``, by the chain rule through ``theta(T)``.

        ``dw/dtheta = (1/2)(1 + rho psi k + R) + (theta/2) k psi' (rho + u/R)``,
        then multiplied by ``dtheta/dT``. This is Dupire's numerator, so its
        sign is the calendar-arbitrage condition.
        """
        k_arr = np.asarray(k, dtype=float)
        theta, psi, u, R = self._pieces(k, T)
        psi_prime = power_law_phi_derivative(theta, self.params.eta, self.params.gamma)

        dw_dtheta = 0.5 * (1.0 + self.params.rho * psi * k_arr + R) + (theta / 2.0) * (
            k_arr * psi_prime
        ) * (self.params.rho + u / R)
        return np.asarray(dw_dtheta * self.dtheta_dT(T))

    # -- quoted forms -----------------------------------------------------

    def implied_vol(self, k: float | np.ndarray, T: float | np.ndarray) -> np.ndarray:
        """Black implied volatility at log-moneyness ``k`` and maturity ``T``."""
        T_arr = np.maximum(np.asarray(T, dtype=float), _MIN_TOTAL_VARIANCE)
        variance = np.maximum(self.total_variance(k, T), 0.0)
        return np.asarray(np.sqrt(variance / T_arr))

    def implied_vol_at_strike(
        self, strike: float | np.ndarray, forward: float, T: float | np.ndarray
    ) -> np.ndarray:
        """Implied volatility for a strike, given that expiry's forward."""
        k = np.log(np.asarray(strike, dtype=float) / forward)
        return self.implied_vol(k, T)

    def butterfly(self, k: float | np.ndarray, T: float | np.ndarray) -> np.ndarray:
        """Gatheral's ``g(k)`` on this surface; positive means no butterfly arbitrage."""
        return butterfly_g(self.total_variance(k, T), self.dw_dk(k, T), self.d2w_dk2(k, T), k)

    def __repr__(self) -> str:
        p = self.params
        return (
            f"SSVISurface(rho={p.rho:.4f}, eta={p.eta:.4f}, gamma={p.gamma:.4f}, "
            f"maturities={len(p.maturities)})"
        )
