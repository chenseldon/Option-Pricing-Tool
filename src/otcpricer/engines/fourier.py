"""COS: European prices from a characteristic function.

Fang and Oosterlee (2008). The density is recovered from its Fourier-cosine
coefficients on a truncated interval, and the payoff's own cosine coefficients
are known in closed form, so a European price is a single inner product:

.. math::

    v \\approx e^{-rT} \\sum_{k=0}^{N-1} {}' \\Re\\!\\left[
        \\varphi\\!\\left(\\frac{k\\pi}{b-a}\\right)
        e^{-i k \\pi \\frac{a}{b-a}} \\right] V_k

The prime halves the first term. Convergence is exponential in ``N`` for a
smooth density, so a thousand terms reach machine precision where a
finite-difference grid would need many thousands of steps -- one European price
costs 265 microseconds here.

Three things govern whether it works, and all three were measured
-----------------------------------------------------------------
**The truncation range**, set from the cumulants as
``[c1 - L*sqrt(c2), c1 + L*sqrt(c2)]``. Too narrow and the density is clipped,
which no number of terms repairs: at ``L = 12`` the error on a calibrated
one-year price *floors* at 3.8e-05 from ``N = 256`` all the way to 2048. See
:data:`DEFAULT_L`.

**The second cumulant**, because it sets that range. A wrong ``c2`` narrows the
window and the price comes back finite, plausible and wrong -- which is exactly
what a transcribed closed form did here, by a factor of five in one regime. See
:func:`~otcpricer.models.heston.cumulants`.

**Which payoff is integrated.** The call's cosine coefficients contain
``exp(upper)``, so a wide range overflows them: on a high-volatility three-year
density at ``L = 20`` the bound reaches 30.8 and the engine returned 3.9e13 as a
price. Puts are integrated instead -- their coefficients stay O(1) -- and calls
come from put-call parity. That single change made accuracy *monotone* in both
``L`` and ``N``; the apparent degradation at wide ranges had been the overflow
all along.

**The characteristic function's branch.** COS evaluates ``phi`` at
``k*pi/(b-a)`` for every ``k``, so a formulation that loses its branch at large
argument poisons the whole sum rather than one term. This is why
:mod:`otcpricer.models.heston` implements only the stable form.

This is a Fourier engine, which ``engines/__init__.py`` has listed among this
project's methods since Phase 1 without one existing. It does now.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from otcpricer.core.exceptions import EngineError, ParameterError
from otcpricer.core.types import OptionType
from otcpricer.models.heston import HestonParams, characteristic_function, cumulants

__all__ = [
    "DEFAULT_L",
    "DEFAULT_TERMS",
    "CosResult",
    "price_european_cos",
    "price_heston_european",
]

#: Cosine terms. Measured against QuantLib over five parameter sets, five
#: strikes and five maturities, worst absolute error:
#:
#:     L=12 N=128    L=20 N=512    L=30 N=1024   L=40 N=2048
#:     3.6e-04       9.6e-07       6.2e-10       7.7e-13
#:
#: 1024 at L=30 costs 265 microseconds and leaves the engine four orders of
#: magnitude inside any calibration tolerance, so that is the default. A
#: calibration loop may lower it; nothing else should need to.
DEFAULT_TERMS = 1024

#: Truncation width in standard deviations of the log return.
#:
#: 30, not the 10 Fang-Oosterlee suggest, because the fits this data produces
#: have ``sigma`` near 2 and a density far wider than a lognormal of the same
#: variance. The narrow settings do not merely converge more slowly -- they
#: converge to the wrong answer: at L=8 the error floors at 1.3e-03 and at L=12
#: at 3.8e-05, unchanged from 256 terms to 2048, because the clipped tail is
#: simply absent from the integral.
#:
#: Once calls stopped being integrated directly (see :func:`_put_coefficients`)
#: accuracy became monotone in ``L``, so this is a measured floor rather than an
#: interior optimum. Wider still costs terms without buying anything.
DEFAULT_L = 30.0


@dataclass(frozen=True, slots=True)
class CosResult:
    """A COS price and the numerics behind it.

    Attributes:
        price: The option price.
        terms: Cosine terms used.
        lower: Lower end of the truncation range, in log return.
        upper: Upper end.
        method: Engine tag, for provenance in a report.
    """

    price: float
    terms: int
    lower: float
    upper: float
    method: str = "cos-fourier"

    @property
    def width_in_sigmas(self) -> float:
        return self.upper - self.lower

    def as_dict(self) -> dict[str, object]:
        return {
            "price": round(self.price, 10),
            "terms": self.terms,
            "truncation": [round(self.lower, 6), round(self.upper, 6)],
            "method": self.method,
        }


def _put_coefficients(
    a: float, b: float, k: NDArray[np.float64], x_strike: float
) -> NDArray[np.float64]:
    """Cosine coefficients of a **put** payoff on ``[a, b]``.

    ``chi`` and ``psi`` are Fang-Oosterlee's equations 22 and 23, written out
    rather than integrated because they are exact -- a quadrature here would put
    a second error source inside a method whose only error should be truncation.

    Only the put, deliberately. ``chi`` contains ``exp(d)``, and for a call
    ``d`` is the *upper* truncation bound, so a wide range overflows it: at
    ``L = 20`` on a high-volatility three-year density the bound reaches 30.8
    and ``exp(30.8)`` is 2.4e13. That was measured -- the call route returned
    3.9e13 as a price. For a put ``d = min(x_strike, b)``, which sits near
    ``log(K/S)``, so the coefficients stay O(1) however wide the range gets.

    Calls therefore come from put-call parity, which is exact.
    """
    width = b - a
    c, d = a, min(x_strike, b)

    if d <= c:
        # The payoff is identically zero on the truncation range.
        return np.zeros_like(k)

    k_pi = k * np.pi / width
    cos_d, cos_c = np.cos(k_pi * (d - a)), np.cos(k_pi * (c - a))
    sin_d, sin_c = np.sin(k_pi * (d - a)), np.sin(k_pi * (c - a))

    # chi: integral of exp(y) cos(k pi (y-a)/(b-a)) over [c, d].
    denominator = 1.0 + k_pi * k_pi
    chi = (
        cos_d * np.exp(d) - cos_c * np.exp(c) + k_pi * sin_d * np.exp(d) - k_pi * sin_c * np.exp(c)
    ) / denominator

    # psi: integral of cos(k pi (y-a)/(b-a)) over [c, d].
    psi = np.empty_like(k, dtype=float)
    psi[0] = d - c
    nonzero = k[1:] * np.pi / width
    psi[1:] = (np.sin(nonzero * (d - a)) - np.sin(nonzero * (c - a))) / nonzero

    # The put's payoff is K - S, so the sign is negative on (chi - e^k psi).
    coefficients: NDArray[np.float64] = -2.0 / width * (chi - np.exp(x_strike) * psi)
    return coefficients


def price_european_cos(
    char_fn: Callable[[NDArray[np.complex128]], NDArray[np.complex128]],
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    flag: OptionType | str = OptionType.CALL,
    lower: float,
    upper: float,
    terms: int = DEFAULT_TERMS,
) -> CosResult:
    """Price a European option from any characteristic function.

    Model-agnostic on purpose: ``char_fn`` is the only thing that knows which
    model this is, so the same engine prices Heston, Black-Scholes or anything
    else with a tractable transform. The Black-Scholes case is used in the
    tests as an analytic anchor, which is the cheapest possible check that the
    quadrature and the payoff coefficients are right before any stochastic
    volatility is involved.

    Args:
        char_fn: ``u -> E[exp(i u log(S_T/S_0))]``, vectorised over ``u``.
        spot: Spot price.
        strike: Strike price.
        maturity: Years to expiry.
        rate: Continuously compounded discount rate.
        flag: Call or put.
        lower: Lower truncation bound, in log return.
        upper: Upper truncation bound.
        terms: Number of cosine terms.

    Raises:
        ParameterError: on a non-positive spot, strike, maturity or term count.
        EngineError: if the truncation range is degenerate, or the result is
            not finite -- which is how a characteristic function that has lost
            its branch announces itself.
    """
    if spot <= 0.0 or strike <= 0.0:
        raise ParameterError(f"spot and strike must be positive, got {spot} and {strike}")
    if maturity <= 0.0:
        raise ParameterError(f"maturity must be positive, got {maturity}")
    if terms < 2:
        raise ParameterError(f"terms must be at least 2, got {terms}")
    if not upper > lower:
        raise EngineError(f"degenerate truncation range [{lower}, {upper}]")

    option_type = OptionType.parse(flag)
    width = upper - lower
    k = np.arange(terms, dtype=float)
    u = k * np.pi / width

    phi = np.asarray(char_fn(u.astype(np.complex128)), dtype=np.complex128)
    # The density's cosine coefficients, shifted to the truncation range.
    coefficients = np.real(phi * np.exp(-1j * u * lower))
    coefficients[0] *= 0.5  # the primed sum halves the first term

    x_strike = float(np.log(strike / spot))
    payoff = _put_coefficients(lower, upper, k, x_strike)

    discount = float(np.exp(-rate * maturity))
    put = float(discount * spot * np.sum(coefficients * payoff))

    if option_type is OptionType.CALL:
        # Parity, rather than the call's own coefficients, which overflow on a
        # wide range. The forward comes from the characteristic function itself
        # -- phi(-i) = E[S_T/S_0] -- so this stays consistent with whatever
        # drift the model was handed instead of assuming exp((r-q)T).
        forward_ratio = float(np.real(char_fn(np.array([-1j], dtype=np.complex128))[0]))
        price = put + discount * (spot * forward_ratio - strike)
    else:
        price = put

    if not np.isfinite(price):
        raise EngineError(
            "COS produced a non-finite price. The usual cause is a "
            "characteristic function losing its branch at large argument -- "
            "see the Little Heston Trap in otcpricer.models.heston."
        )

    # A price below intrinsic is arbitrage, and here it means the truncation
    # range clipped the density rather than that the model says so.
    forward = spot * float(np.real(char_fn(np.array([-1j], dtype=np.complex128))[0]))
    intrinsic = discount * max(
        forward - strike if option_type is OptionType.CALL else strike - forward, 0.0
    )
    if price < intrinsic - 1e-8 * max(spot, 1.0):
        raise EngineError(
            f"COS price {price:.8f} is below intrinsic {intrinsic:.8f} on "
            f"[{lower:.3f}, {upper:.3f}] with {terms} terms; the truncation "
            f"range is too narrow for this density"
        )

    return CosResult(price=max(price, 0.0), terms=terms, lower=lower, upper=upper)


def price_heston_european(
    params: HestonParams,
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float = 0.0,
    flag: OptionType | str = OptionType.CALL,
    terms: int = DEFAULT_TERMS,
    width: float = DEFAULT_L,
) -> CosResult:
    """Price a European option under Heston by COS.

    The truncation range comes from the model's own cumulants, so it adapts to
    the parameters instead of assuming a scale. That matters at the fits this
    data produces, where ``sigma`` near 2 makes the density far wider than a
    lognormal of the same variance.

    Args:
        params: Heston parameters.
        spot: Spot price.
        strike: Strike price.
        maturity: Years to expiry.
        rate: Continuously compounded rate.
        dividend: Continuous dividend yield.
        flag: Call or put.
        terms: Cosine terms.
        width: Truncation half-width in standard deviations.
    """
    drift = rate - dividend
    c1, c2 = cumulants(params, maturity, drift=drift)
    half = width * float(np.sqrt(max(c2, 1e-12)))

    def char_fn(u: NDArray[np.complex128]) -> Any:
        return characteristic_function(params, u, maturity, drift=drift)

    return price_european_cos(
        char_fn,
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        flag=flag,
        lower=c1 - half,
        upper=c1 + half,
        terms=terms,
    )
