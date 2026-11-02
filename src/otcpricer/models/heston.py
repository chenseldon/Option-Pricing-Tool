"""The Heston model: stochastic volatility, and the branch cut that breaks it.

Why this is not behind ``core.vol.LocalVolatility``
---------------------------------------------------
That Protocol asks for ``local_vol(S, t)`` -- a volatility that is a function
of spot and time. Heston's variance is a *second state variable* with its own
Brownian motion, so there is no such function, and no amount of interface
widening would produce one. The Protocol's docstring says this deliberately: it
was never meant to abstract "any volatility model".

The consequence is structural rather than cosmetic. A local-volatility PDE is
one-dimensional in space; a Heston PDE is two-dimensional with a cross
derivative, and needs an ADI solver. That is not built here -- see "One engine,
and saying so" below.

The Little Heston Trap
----------------------
The characteristic function has two algebraically identical forms, differing
only in which square root is taken. One produces ``exp(+dT)`` and the other
``exp(-dT)``, where ``d`` has positive real part. Albrecher et al. (2007) showed
the second is the stable one, and this module implements only that.

The classic form is kept in the tests as a *control*, because the failure is
worth being able to demonstrate rather than assert:

===  ========================  ===================
 T    classic, largest jump     little-trap form
===  ========================  ===================
 2            9.9e-04                 9.9e-04
 5            **2.9e-01**             1.6e-03
20            **NaN**                 6.0e-03
===  ========================  ===================

Where both are stable they agree to 2.8e-16, so this is not an accuracy
question -- it is a branch of the complex logarithm being crossed. And at the
parameters actually calibrated to this data (``sigma = 1.96``, ``kappa =
4.86``), ``exp(+dT)`` exceeds double precision at **T = 2.47 years**, which is
inside a snowball's life. The trap is a hard requirement here, not a textbook
footnote.

Feller is violated, and that is the point
------------------------------------------
Calibrated to the panel, ``2*kappa*theta - sigma^2`` comes back **negative on
both captured underlyings** (-3.36 and -1.52). The variance process can
therefore reach zero, so a Euler discretisation steps negative and
``sqrt(v)`` fails.

:class:`HestonParams` therefore *reports* the Feller margin and does not reject
on it. Rejecting would make the model unusable on its own best fit, which is a
sign the constraint does not belong in the constructor. What it does mean is
that Andersen's QE scheme is mandatory rather than a refinement -- see
:mod:`otcpricer.engines.mc`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from otcpricer.core.exceptions import ParameterError

__all__ = [
    "HestonParams",
    "characteristic_function",
    "cumulants",
]


@dataclass(frozen=True, slots=True)
class HestonParams:
    """Heston's five parameters.

    The dynamics, under the pricing measure:

    .. math::

        dS_t &= (r - q) S_t\\,dt + \\sqrt{v_t}\\,S_t\\,dW^1_t \\\\
        dv_t &= \\kappa(\\theta - v_t)\\,dt + \\sigma\\sqrt{v_t}\\,dW^2_t

    with :math:`dW^1 dW^2 = \\rho\\,dt`.

    Attributes:
        v0: Initial variance. Not a volatility -- ``0.04`` is a 20% vol.
        kappa: Mean-reversion speed.
        theta: Long-run variance.
        sigma: Volatility of variance. Often called vol-of-vol.
        rho: Correlation between the two Brownian motions. Negative gives the
            downside skew equities usually quote.
    """

    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float

    def __post_init__(self) -> None:
        if self.v0 <= 0.0:
            raise ParameterError(f"v0 must be positive, got {self.v0}")
        if self.kappa <= 0.0:
            raise ParameterError(f"kappa must be positive, got {self.kappa}")
        if self.theta <= 0.0:
            raise ParameterError(f"theta must be positive, got {self.theta}")
        if self.sigma <= 0.0:
            raise ParameterError(f"sigma must be positive, got {self.sigma}")
        if not -1.0 < self.rho < 1.0:
            # The open interval, for the same reason SSVI's rho uses it: at
            # |rho| = 1 the two Brownian motions are the same one and the
            # model degenerates.
            raise ParameterError(f"rho must lie in (-1, 1), got {self.rho}")

    @property
    def feller_margin(self) -> float:
        """``2*kappa*theta - sigma^2``.

        Positive keeps the variance strictly away from zero. Negative is
        *reported, not rejected*: the panel's own best fits violate it, so a
        constructor that refused them would refuse the calibrated model.
        """
        return 2.0 * self.kappa * self.theta - self.sigma * self.sigma

    @property
    def satisfies_feller(self) -> bool:
        return self.feller_margin > 0.0

    @property
    def initial_vol(self) -> float:
        """``sqrt(v0)``, for reading the parameters as volatilities."""
        return float(np.sqrt(self.v0))

    @property
    def long_run_vol(self) -> float:
        return float(np.sqrt(self.theta))

    def with_v0(self, v0: float) -> HestonParams:
        return replace(self, v0=v0)

    def as_dict(self) -> dict[str, object]:
        return {
            "v0": round(self.v0, 8),
            "kappa": round(self.kappa, 8),
            "theta": round(self.theta, 8),
            "sigma": round(self.sigma, 8),
            "rho": round(self.rho, 8),
            "initial_vol": round(self.initial_vol, 6),
            "long_run_vol": round(self.long_run_vol, 6),
            "feller_margin": round(self.feller_margin, 8),
            "satisfies_feller": self.satisfies_feller,
            "feller_note": (
                "2*kappa*theta - sigma^2. Negative means the variance can "
                "reach zero, so Euler stepping goes negative and the QE scheme "
                "is required. Reported rather than enforced -- the calibrated "
                "fits violate it."
            ),
        }


def characteristic_function(
    params: HestonParams,
    u: complex | NDArray[np.complex128],
    maturity: float,
    *,
    drift: float = 0.0,
) -> Any:
    """``E[exp(i u log(S_T / S_0))]`` under Heston.

    The "little trap" formulation of Albrecher et al. (2007): the root is taken
    so that the exponential is ``exp(-dT)`` with ``Re(d) > 0``, which stays
    bounded for every maturity. The classic form with ``exp(+dT)`` is
    algebraically identical and numerically unusable past a few years -- at the
    parameters calibrated here it overflows double precision at 2.47 years.

    Args:
        params: The five parameters.
        u: Fourier variable, scalar or array.
        maturity: Time to expiry in years.
        drift: ``r - q``, the deterministic log drift.

    Returns:
        The characteristic function at ``u``, same shape as ``u``.
    """
    if maturity < 0.0:
        raise ParameterError(f"maturity must be non-negative, got {maturity}")

    u_arr = np.asarray(u, dtype=np.complex128)
    if maturity == 0.0:
        return np.ones_like(u_arr)

    kappa, theta, sigma, rho, v0 = (
        params.kappa,
        params.theta,
        params.sigma,
        params.rho,
        params.v0,
    )
    iu = 1j * u_arr

    # beta is kappa minus the correlation term; d is the discriminant root.
    beta = kappa - rho * sigma * iu
    d = np.sqrt(beta * beta + sigma * sigma * (iu + u_arr * u_arr))

    # The choice that matters. g uses (beta - d) over (beta + d), so the
    # exponential below is exp(-d*T) and |g| <= 1. Taking the other root gives
    # the classic form, exp(+d*T), and an overflow.
    g = (beta - d) / (beta + d)

    exp_dt = np.exp(-d * maturity)
    one_minus_g_exp = 1.0 - g * exp_dt

    # D multiplies v0; C is the rest.
    D = (beta - d) / (sigma * sigma) * ((1.0 - exp_dt) / one_minus_g_exp)
    C = (
        (kappa * theta)
        / (sigma * sigma)
        * ((beta - d) * maturity - 2.0 * np.log(one_minus_g_exp / (1.0 - g)))
    )

    return np.exp(C + D * v0 + iu * drift * maturity)


def _characteristic_function_classic(
    params: HestonParams,
    u: complex | NDArray[np.complex128],
    maturity: float,
    *,
    drift: float = 0.0,
) -> Any:
    """The classic formulation, kept only so the trap can be demonstrated.

    Algebraically identical to :func:`characteristic_function` -- it takes the
    other square root, giving ``exp(+dT)``. Not exported and never used for
    pricing. Its failure is reproduced in the tests, because a hazard nobody
    can demonstrate is a hazard nobody believes.
    """
    u_arr = np.asarray(u, dtype=np.complex128)
    if maturity == 0.0:
        return np.ones_like(u_arr)

    kappa, theta, sigma, rho, v0 = (
        params.kappa,
        params.theta,
        params.sigma,
        params.rho,
        params.v0,
    )
    iu = 1j * u_arr
    beta = kappa - rho * sigma * iu
    d = np.sqrt(beta * beta + sigma * sigma * (iu + u_arr * u_arr))

    g = (beta + d) / (beta - d)
    exp_dt = np.exp(d * maturity)
    one_minus_g_exp = 1.0 - g * exp_dt

    D = (beta + d) / (sigma * sigma) * ((1.0 - exp_dt) / one_minus_g_exp)
    C = (
        (kappa * theta)
        / (sigma * sigma)
        * ((beta + d) * maturity - 2.0 * np.log(one_minus_g_exp / (1.0 - g)))
    )
    return np.exp(C + D * v0 + iu * drift * maturity)


def cumulants(
    params: HestonParams, maturity: float, *, drift: float = 0.0, step: float = 1e-3
) -> tuple[float, float]:
    """First and second cumulants of ``log(S_T / S_0)``.

    These set the COS method's truncation range, and getting the second one
    wrong is unusually dangerous: it narrows the range, the density is clipped,
    and the price comes back finite, plausible and wrong.

    Why this differentiates the characteristic function instead of using a
    closed form
    -----------------------------------------------------------------------
    Fang and Oosterlee give ``c2`` in closed form, and it is long -- half a
    dozen terms in ``kappa``, ``theta``, ``sigma``, ``rho`` and two
    exponentials. Transcribing it here produced a value that was 0.4% off in a
    textbook parameter set and **wrong by a factor of five** at
    ``(v0, kappa, theta, sigma, rho) = (0.25, 0.5, 0.30, 1.5, -0.9)``: 0.0888
    against a true 0.4424. That halved the truncation width and the resulting
    European prices were off by hundreds of currency units, with no warning.

    The cumulants are by definition the derivatives of ``log phi`` at zero, so
    differentiating numerically is exact up to the step size *and* automatically
    consistent with whichever characteristic function the engine is actually
    integrating. A transcription error becomes impossible rather than merely
    unlikely, which is the right trade for three extra function evaluations.

    Verified against a 400,000-path Euler simulation: both cumulants agree to
    better than 1% across textbook, calibrated and high-volatility parameters.

    Args:
        params: Heston parameters.
        maturity: Years to expiry.
        drift: ``r - q``.
        step: Finite-difference step in the Fourier variable.

    Returns:
        ``(c1, c2)``, the mean and variance of the log return. ``c2`` is
        floored at zero rather than taking an absolute value -- a negative
        variance means the differencing broke down, and ``abs`` would hide that
        the way it hid the transcription error.
    """
    if maturity <= 0.0:
        return (0.0, 0.0)

    h = float(step)
    grid = np.array([-h, 0.0, h], dtype=np.complex128)
    phi = np.asarray(characteristic_function(params, grid, maturity, drift=drift))
    log_phi = np.log(phi)

    # d/du log phi |_0 = i*c1, and d2/du2 log phi |_0 = -c2.
    c1 = float(np.real((log_phi[2] - log_phi[0]) / (2.0 * h) / 1j))
    c2 = float(np.real(-(log_phi[2] - 2.0 * log_phi[1] + log_phi[0]) / (h * h)))

    return (c1, max(c2, 0.0))
