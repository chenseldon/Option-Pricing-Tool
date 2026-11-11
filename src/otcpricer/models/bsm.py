"""Black-Scholes-Merton: price, analytic Greeks, and implied volatility.

This replaces the project's former dependency on ``py_vollib``, which is now
only a deprecation shim in front of ``vollib`` and emits a warning on import.
Writing the formulae out has two concrete benefits beyond removing a dead
dependency: every term is visible and reviewable in place, and the whole module
is vectorised, so the price/Greeks *curves* the web UI draws come from one
array call instead of a Python loop over several hundred spots.

Greek units follow desk convention, matching the previous behaviour exactly so
that no downstream number moves:

===========  ==========================================
``delta``    per 1 unit of spot
``gamma``    per 1 unit of spot, squared
``vega``     per **1 percentage point** of volatility
``theta``    per **calendar day**
``rho``      per **1 percentage point** of rate
===========  ==========================================

The normal CDF comes from :func:`scipy.special.ndtr` rather than
``scipy.stats.norm.cdf``: identical values, materially faster, and it accepts
arrays without constructing a frozen distribution object.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from otcpricer.core.exceptions import ArbitrageError, ConvergenceError, ParameterError
from otcpricer.core.types import Greeks, OptionType
from otcpricer.core.validation import validate_vanilla

__all__ = [
    "RHO_SCALE",
    "THETA_SCALE",
    "VEGA_SCALE",
    "d1_d2",
    "delta",
    "forward_price",
    "gamma",
    "greeks",
    "implied_volatility",
    "no_arbitrage_bounds",
    "price",
    "rho",
    "theta",
    "vega",
    "vol_uncertainty",
]

#: Vega is quoted per 1 percentage point of vol, not per unit.
VEGA_SCALE = 0.01
#: Theta is quoted per calendar day, not per year.
THETA_SCALE = 1.0 / 365.0
#: Rho is quoted per 1 percentage point of rate, not per unit.
RHO_SCALE = 0.01

#: Below this total standard deviation (``sigma * sqrt(T)``) the lognormal
#: density collapses and ``d1``/``d2`` overflow, so we switch to the
#: deterministic forward limit instead of dividing by (almost) zero.
_DEGENERATE_VOL = 1e-12

#: Hard cap on the volatility the implied-vol solver will search.
_MAX_SEARCH_VOL = 5.0


#: 1 / sqrt(2 * pi), the standard normal density's normalising constant.
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    """Standard normal density, written out to avoid a ``scipy.stats`` call."""
    return np.asarray(np.exp(-0.5 * x * x) * _INV_SQRT_2PI)


def _as_array(*values: float | np.ndarray) -> tuple[np.ndarray, ...]:
    """Broadcast inputs to a common shape as float arrays."""
    arrays = [np.asarray(v, dtype=float) for v in values]
    if len(arrays) == 1:
        return (arrays[0],)
    return tuple(np.broadcast_arrays(*arrays))


def _scalar_if_scalar(result: np.ndarray, *inputs: float | np.ndarray) -> float | np.ndarray:
    """Return a plain float when every input was scalar.

    Keeps single-quote call sites returning ``float`` while curve call sites
    get arrays back.
    """
    if all(np.ndim(i) == 0 for i in inputs):
        return float(np.asarray(result).reshape(()).item())
    return result


def d1_d2(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    q: float | np.ndarray = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the BSM ``d1`` and ``d2`` terms.

    ``d1 = [ln(S/K) + (r - q + sigma^2 / 2) * T] / (sigma * sqrt(T))`` and
    ``d2 = d1 - sigma * sqrt(T)``.

    Where ``sigma * sqrt(T)`` underflows, both are set to +/- infinity
    according to the sign of ``ln(F/K)``. That makes the normal CDFs collapse
    to 0 or 1 and recovers the correct deterministic limit, rather than
    producing NaN.
    """
    S_a, K_a, T_a, r_a, sigma_a, q_a = _as_array(S, K, T, r, sigma, q)
    sqrt_T = np.sqrt(T_a)
    total_vol = sigma_a * sqrt_T

    log_moneyness = np.log(S_a / K_a)
    degenerate = total_vol < _DEGENERATE_VOL

    # Guard the divisor so degenerate entries do not emit a warning; their
    # values are overwritten immediately below.
    safe_vol = np.where(degenerate, 1.0, total_vol)
    d1 = (log_moneyness + (r_a - q_a + 0.5 * sigma_a * sigma_a) * T_a) / safe_vol
    d2 = d1 - safe_vol

    if np.any(degenerate):
        forward_log = log_moneyness + (r_a - q_a) * T_a
        limit = np.where(forward_log > 0.0, np.inf, np.where(forward_log < 0.0, -np.inf, 0.0))
        d1 = np.where(degenerate, limit, d1)
        d2 = np.where(degenerate, limit, d2)

    return d1, d2


def forward_price(
    S: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    q: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """Cost-of-carry forward, ``F = S * exp((r - q) * T)``."""
    S_a, T_a, r_a, q_a = _as_array(S, T, r, q)
    return _scalar_if_scalar(S_a * np.exp((r_a - q_a) * T_a), S, T, r, q)


def price(
    option_type: OptionType | str,
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    q: float | np.ndarray = 0.0,
    *,
    validate: bool = True,
) -> float | np.ndarray:
    """Black-Scholes-Merton price of a European option.

    ``C = S * e^{-qT} * N(d1) - K * e^{-rT} * N(d2)`` and
    ``P = K * e^{-rT} * N(-d2) - S * e^{-qT} * N(-d1)``.

    Args:
        option_type: Call or put; the legacy ``'c'``/``'p'`` strings are accepted.
        S: Spot price of the underlying.
        K: Strike.
        T: Time to maturity in years.
        r: Continuously compounded risk-free rate.
        sigma: Annualised volatility.
        q: Continuous dividend yield.
        validate: Check input domains. Pass ``False`` on hot paths such as
            finite-difference loops where the caller has already validated.

    Returns:
        The option premium: a ``float`` for scalar inputs, an array otherwise.
    """
    opt = OptionType.parse(option_type)
    if validate:
        validate_vanilla(
            float(np.min(S)),
            float(np.min(K)),
            float(np.min(T)),
            0.0,
            float(np.max(sigma)),
            float(np.min(q)),
        )

    S_a, K_a, T_a, r_a, sigma_a, q_a = _as_array(S, K, T, r, sigma, q)
    d1, d2 = d1_d2(S_a, K_a, T_a, r_a, sigma_a, q_a)

    discount = np.exp(-r_a * T_a)
    carry = np.exp(-q_a * T_a)

    if opt is OptionType.CALL:
        result = S_a * carry * ndtr(d1) - K_a * discount * ndtr(d2)
    else:
        result = K_a * discount * ndtr(-d2) - S_a * carry * ndtr(-d1)

    # The degenerate-vol branch yields exactly the discounted forward
    # intrinsic; clamp the tiny negatives that catastrophic cancellation can
    # leave behind for deep out-of-the-money strikes.
    return _scalar_if_scalar(np.maximum(result, 0.0), S, K, T, r, sigma, q)


def delta(
    option_type: OptionType | str,
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    q: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """``dV/dS``: ``e^{-qT} * N(d1)`` for a call, ``-e^{-qT} * N(-d1)`` for a put."""
    opt = OptionType.parse(option_type)
    S_a, K_a, T_a, r_a, sigma_a, q_a = _as_array(S, K, T, r, sigma, q)
    d1, _ = d1_d2(S_a, K_a, T_a, r_a, sigma_a, q_a)
    carry = np.exp(-q_a * T_a)
    result = carry * ndtr(d1) if opt is OptionType.CALL else -carry * ndtr(-d1)
    return _scalar_if_scalar(result, S, K, T, r, sigma, q)


def gamma(
    option_type: OptionType | str,
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    q: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """``d2V/dS2 = e^{-qT} * n(d1) / (S * sigma * sqrt(T))``.

    Identical for calls and puts, which is why put-call parity has zero
    curvature.
    """
    OptionType.parse(option_type)  # validated even though gamma is type-independent
    S_a, K_a, T_a, r_a, sigma_a, q_a = _as_array(S, K, T, r, sigma, q)
    d1, _ = d1_d2(S_a, K_a, T_a, r_a, sigma_a, q_a)
    denominator = S_a * sigma_a * np.sqrt(T_a)
    safe = np.where(denominator < _DEGENERATE_VOL, 1.0, denominator)
    result = np.where(denominator < _DEGENERATE_VOL, 0.0, np.exp(-q_a * T_a) * _norm_pdf(d1) / safe)
    return _scalar_if_scalar(result, S, K, T, r, sigma, q)


def vega(
    option_type: OptionType | str,
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    q: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """``dV/dsigma`` per percentage point: ``S * e^{-qT} * n(d1) * sqrt(T) / 100``.

    Identical for calls and puts.
    """
    OptionType.parse(option_type)
    S_a, K_a, T_a, r_a, sigma_a, q_a = _as_array(S, K, T, r, sigma, q)
    d1, _ = d1_d2(S_a, K_a, T_a, r_a, sigma_a, q_a)
    result = S_a * np.exp(-q_a * T_a) * _norm_pdf(d1) * np.sqrt(T_a) * VEGA_SCALE
    return _scalar_if_scalar(result, S, K, T, r, sigma, q)


def theta(
    option_type: OptionType | str,
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    q: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """``dV/dt`` per calendar day (negative for a long option, normally).

    Three economic pieces: decay of optionality, carry on the dividend stream,
    and interest on the deferred strike payment.
    """
    opt = OptionType.parse(option_type)
    S_a, K_a, T_a, r_a, sigma_a, q_a = _as_array(S, K, T, r, sigma, q)
    d1, d2 = d1_d2(S_a, K_a, T_a, r_a, sigma_a, q_a)

    sqrt_T = np.sqrt(T_a)
    safe_sqrt_T = np.where(sqrt_T < _DEGENERATE_VOL, 1.0, sqrt_T)
    carry = np.exp(-q_a * T_a)
    discount = np.exp(-r_a * T_a)

    decay = (S_a * carry * _norm_pdf(d1) * sigma_a) / (2.0 * safe_sqrt_T)
    decay = np.where(sqrt_T < _DEGENERATE_VOL, 0.0, decay)

    if opt is OptionType.CALL:
        dividend = -q_a * S_a * carry * ndtr(d1)
        interest = r_a * K_a * discount * ndtr(d2)
        result = -(decay + dividend + interest) * THETA_SCALE
    else:
        dividend = -q_a * S_a * carry * ndtr(-d1)
        interest = r_a * K_a * discount * ndtr(-d2)
        result = (-decay + dividend + interest) * THETA_SCALE

    return _scalar_if_scalar(result, S, K, T, r, sigma, q)


def rho(
    option_type: OptionType | str,
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    sigma: float | np.ndarray,
    q: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """``dV/dr`` per percentage point: ``T * K * e^{-rT} * N(d2) / 100`` for a call."""
    opt = OptionType.parse(option_type)
    S_a, K_a, T_a, r_a, sigma_a, q_a = _as_array(S, K, T, r, sigma, q)
    _, d2 = d1_d2(S_a, K_a, T_a, r_a, sigma_a, q_a)
    discount = np.exp(-r_a * T_a)
    if opt is OptionType.CALL:
        result = T_a * K_a * discount * ndtr(d2) * RHO_SCALE
    else:
        result = -T_a * K_a * discount * ndtr(-d2) * RHO_SCALE
    return _scalar_if_scalar(result, S, K, T, r, sigma, q)


def greeks(
    option_type: OptionType | str,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> Greeks:
    """All five Greeks at one point, as a :class:`~otcpricer.core.types.Greeks`.

    Shares a single ``d1``/``d2`` evaluation across all five, instead of
    recomputing them per Greek as the legacy per-function wrappers did.
    """
    opt = OptionType.parse(option_type)
    validate_vanilla(S, K, T, r, sigma, q)

    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    sqrt_T = float(np.sqrt(T))
    carry = float(np.exp(-q * T))
    discount = float(np.exp(-r * T))
    pdf_d1 = float(_norm_pdf(d1))
    cdf_d1 = float(ndtr(d1))
    cdf_d2 = float(ndtr(d2))

    decay = (S * carry * pdf_d1 * sigma) / (2.0 * sqrt_T)

    if opt is OptionType.CALL:
        delta_v = carry * cdf_d1
        rho_v = T * K * discount * cdf_d2 * RHO_SCALE
        theta_v = -(decay - q * S * carry * cdf_d1 + r * K * discount * cdf_d2) * THETA_SCALE
    else:
        # N(-x) == 1 - N(x); reusing the CDFs already computed above keeps this
        # to one ndtr evaluation per term.
        delta_v = -carry * (1.0 - cdf_d1)
        rho_v = -T * K * discount * (1.0 - cdf_d2) * RHO_SCALE
        theta_v = (
            -decay - q * S * carry * (1.0 - cdf_d1) + r * K * discount * (1.0 - cdf_d2)
        ) * THETA_SCALE

    return Greeks(
        delta=delta_v,
        gamma=carry * pdf_d1 / (S * sigma * sqrt_T),
        vega=S * carry * pdf_d1 * sqrt_T * VEGA_SCALE,
        theta=theta_v,
        rho=rho_v,
    )


def no_arbitrage_bounds(
    option_type: OptionType | str,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
) -> tuple[float, float]:
    """Return the ``(lower, upper)`` no-arbitrage premium bounds.

    A European call is worth at least its discounted forward intrinsic and at
    most the discounted forward itself; a put is bounded above by the
    discounted strike. A quote outside this band cannot be matched by any
    volatility, so :func:`implied_volatility` rejects it up front rather than
    failing to converge later with an opaque error.
    """
    opt = OptionType.parse(option_type)
    discounted_forward = S * float(np.exp(-q * T))
    discounted_strike = K * float(np.exp(-r * T))
    if opt is OptionType.CALL:
        return max(0.0, discounted_forward - discounted_strike), discounted_forward
    return max(0.0, discounted_strike - discounted_forward), discounted_strike


def _brenner_subrahmanyam(market_price: float, S: float, T: float) -> float:
    """Closed-form starting volatility, exact for an at-the-money forward."""
    if S <= 0.0 or T <= 0.0:
        return 0.2
    return float(np.sqrt(2.0 * np.pi / T) * market_price / S)


def _brent_implied_vol(
    opt: OptionType,
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
) -> float:
    """Bracketed bisection fallback for the implied-vol solve."""

    def objective(vol: float) -> float:
        return float(price(opt, S, K, T, r, vol, q, validate=False)) - market_price

    low, high = 1e-6, _MAX_SEARCH_VOL
    f_low, f_high = objective(low), objective(high)

    # Price is monotonically increasing in vol, so a same-sign bracket means the
    # quote is unreachable anywhere in the admissible range.
    if f_low > 0.0:
        raise ConvergenceError(
            f"premium {market_price:.6f} is below the zero-volatility value "
            f"{market_price + f_low:.6f}; implied volatility would be negative"
        )
    if f_high < 0.0:
        raise ConvergenceError(
            f"premium {market_price:.6f} exceeds the value at "
            f"{_MAX_SEARCH_VOL:.0%} volatility {market_price + f_high:.6f}; "
            f"implied volatility is above the cap"
        )

    try:
        return float(brentq(objective, low, high, xtol=1e-12, rtol=8.9e-16, maxiter=200))
    except (RuntimeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ConvergenceError(
            f"implied volatility solve failed for premium {market_price} "
            f"(S={S}, K={K}, T={T}, r={r}, q={q}): {exc}"
        ) from exc


def vol_uncertainty(
    option_type: OptionType | str,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    price_tolerance: float = 1e-8,
) -> float:
    """How precisely a quote can pin down volatility, in vol units.

    Inverting the local linearisation ``d(price) = vega * d(sigma)`` gives
    ``d(sigma) = price_tolerance / vega``: the width of the volatility band
    that reproduces the quote to within the price tolerance. Returns infinity
    once vega underflows, meaning the quote constrains volatility not at all.

    Use this to screen a quote file before fitting a surface: a strike whose
    uncertainty exceeds the width of the smile feature being fitted carries no
    usable information, however precisely its implied vol is reported.
    """
    opt = OptionType.parse(option_type)
    # vega is quoted per percentage point; undo that for d(price)/d(sigma).
    sensitivity = abs(float(vega(opt, S, K, T, r, sigma, q))) / VEGA_SCALE
    if sensitivity <= 0.0:
        return float("inf")
    return price_tolerance / sensitivity


def _assert_identifiable(
    opt: OptionType,
    solved_vol: float,
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    tolerance: float,
    max_vol_uncertainty: float,
) -> None:
    """Reject a solve whose volatility the quote cannot actually pin down.

    See the note on :func:`implied_volatility` for why this matters.
    """
    if max_vol_uncertainty <= 0.0 or not np.isfinite(max_vol_uncertainty):
        return
    uncertainty = vol_uncertainty(opt, S, K, T, r, solved_vol, q, price_tolerance=tolerance)
    if uncertainty <= max_vol_uncertainty:
        return

    lower, _ = no_arbitrage_bounds(opt, S, K, T, r, q)
    reported = "infinite" if not np.isfinite(uncertainty) else f"{uncertainty:.3e}"
    raise ConvergenceError(
        f"premium {market_price:.8f} does not pin down a volatility: the recovered "
        f"{solved_vol:.4%} is only resolvable to +/-{reported} in vol units, against a "
        f"{max_vol_uncertainty:.1e} limit. The quote sits at (or within rounding of) its "
        f"discounted forward intrinsic {lower:.8f}, where vega underflows and a wide range "
        f"of volatilities reproduces the same price to machine precision "
        f"(S={S}, K={K}, T={T}). Exclude this strike from a surface fit, or pass "
        f"max_vol_uncertainty=float('inf') to accept an unidentified value."
    )


def implied_volatility(
    option_type: OptionType | str,
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
    initial_guess: float | None = None,
    max_vol_uncertainty: float = 1e-3,
) -> float:
    """Solve BSM for the volatility that reproduces ``market_price``.

    Newton-Raphson on price using the analytic vega, falling back to Brent
    bisection when Newton leaves the admissible bracket or stalls on a
    vanishing vega. That fallback is what makes the solver usable on real
    quote files: for deep in- or out-of-the-money strikes the price is nearly
    flat in volatility, so the Newton step explodes and the naive solver that
    this replaces simply raised.

    The initial guess uses the Brenner-Subrahmanyam approximation
    ``sigma ~= sqrt(2 * pi / T) * price / S``, which is exact for an
    at-the-money forward option and a sound starting point elsewhere.

    Args:
        option_type: Call or put.
        market_price: Observed premium.
        S: Spot.
        K: Strike.
        T: Time to maturity in years.
        r: Risk-free rate.
        q: Dividend yield.
        tolerance: Absolute price tolerance for convergence.
        max_iterations: Newton iteration cap before handing over to Brent.
        initial_guess: Override the starting volatility.
        max_vol_uncertainty: Reject the solve when the quote cannot pin the
            volatility to within this many vol units (default 0.001, i.e. a
            tenth of a vol point). See the note below. Pass
            ``float('inf')`` to disable the check.

    Returns:
        The implied volatility as a decimal (``0.20`` means 20%).

    Raises:
        ParameterError: If the premium or maturity is non-positive.
        ArbitrageError: If the premium sits outside the no-arbitrage band.
        ConvergenceError: If neither solver reaches the tolerance, or if the
            quote carries no volatility information.

    Note:
        Deep in-the-money and very short-dated options price at their
        discounted forward intrinsic to full double precision: vega underflows
        to exactly zero, and *every* volatility over a wide range reproduces
        the quote. The implied volatility is then not identifiable, and a
        solver that returns the first value it lands on will hand back a
        plausible-looking number with no content -- which then pollutes any
        surface fitted through it. ``max_vol_uncertainty`` turns that silent
        failure into an explicit one.

        The screen is expressed in vol units rather than as a raw vega floor
        because that is the quantity with meaning: inverting
        ``d(price) = vega * d(sigma)`` gives the width of the volatility band
        consistent with the quote. A bare vega threshold has to be retuned
        whenever the price tolerance, the notional, or the underlying's scale
        changes; this does not. See :func:`vol_uncertainty`.
    """
    opt = OptionType.parse(option_type)
    if market_price <= 0.0:
        raise ParameterError(f"market price must be greater than 0, got {market_price!r}")
    if T <= 0.0:
        raise ParameterError(f"maturity T must be greater than 0, got {T!r}")

    lower, upper = no_arbitrage_bounds(opt, S, K, T, r, q)
    # A hair of slack absorbs rounding in a quote printed to two decimals.
    slack = max(1e-10, 1e-9 * max(1.0, upper))
    if market_price < lower - slack or market_price > upper + slack:
        raise ArbitrageError(
            f"{opt.name.lower()} premium {market_price:.6f} lies outside the no-arbitrage "
            f"band [{lower:.6f}, {upper:.6f}] for S={S}, K={K}, T={T}, r={r}, q={q}; "
            f"no volatility can reproduce it"
        )

    start = (
        initial_guess if initial_guess is not None else _brenner_subrahmanyam(market_price, S, T)
    )
    sigma = float(np.clip(start, 1e-4, _MAX_SEARCH_VOL))

    solved: float | None = None
    for _ in range(max_iterations):
        model_price = float(price(opt, S, K, T, r, sigma, q, validate=False))
        diff = model_price - market_price
        if abs(diff) < tolerance:
            solved = sigma
            break
        # vega is returned per percentage point; undo that to get d(price)/d(sigma).
        slope = float(vega(opt, S, K, T, r, sigma, q)) / VEGA_SCALE
        if slope < 1e-10:
            break  # price is flat in vol -- hand over to Brent
        step = diff / slope
        next_sigma = sigma - step
        if not np.isfinite(next_sigma) or not 0.0 < next_sigma < _MAX_SEARCH_VOL:
            break
        if abs(step) < 1e-14:
            solved = next_sigma
            break
        sigma = next_sigma

    if solved is None:
        solved = _brent_implied_vol(opt, market_price, S, K, T, r, q)

    _assert_identifiable(opt, solved, market_price, S, K, T, r, q, tolerance, max_vol_uncertainty)
    return solved
