"""Input validation shared by every product and engine.

The legacy code repeated the same ad-hoc parameter checks in three places:
``bs_pricing.validate_params`` (returning a ``(bool, str)`` tuple), a second
copy of the bounds inside each ``greeks_calc`` function, and a third set of
``float(...)`` casts with try/except in every Flask endpoint. This module is
the single source of truth.

Validators raise :class:`~otcpricer.core.exceptions.ParameterError` instead of
returning a flag, so a mistake cannot be ignored by forgetting to inspect a
return value.
"""

from __future__ import annotations

from collections.abc import Iterable

from otcpricer.core.exceptions import ParameterError

__all__ = [
    "MAX_ABS_CARRY",
    "MAX_VOLATILITY",
    "check_positive",
    "check_non_negative",
    "check_in_range",
    "validate_market_params",
    "validate_vanilla",
    "validate_barrier",
    "validate_snowball",
]

#: Upper bound on annualised volatility. 500% is absurd for a listed index but
#: can legitimately appear mid-solve in an implied-vol search, so the solver
#: clamps rather than relying on this bound.
MAX_VOLATILITY = 5.0

#: Bound on the magnitude of an annualised carry or basis, either sign.
#:
#: The sign is *not* constrained, and that is the point of this constant
#: existing. ``basis_pa`` was required to be non-negative, which assumed the
#: futures can only be at a discount. Measured over the 725-day CFFEX panel at
#: the panel's own zero rate, the spot-to-farthest-contract carry is negative
#: on **8.3%** of days for IF and **10.3%** for IH; the underlying premium -- a
#: future trading above its index -- occurs on **40.9%** of IH days. Premium is
#: a real market state, and clamping it to zero would be editing the data to
#: fit the assumption rather than the reverse.
#:
#: What remains worth refusing is a unit error. The measured range across the
#: panel is -6.9% to +19.1%, so 100% a year is two decades of headroom and
#: still catches a basis handed over in percent (``6.0`` meaning 6%), which is
#: the mistake this bound is actually for.
MAX_ABS_CARRY = 1.0


def check_positive(value: float, name: str) -> float:
    """Require ``value > 0``."""
    v = float(value)
    if not v > 0.0:
        raise ParameterError(f"{name} must be greater than 0, got {v!r}")
    return v


def check_non_negative(value: float, name: str) -> float:
    """Require ``value >= 0``."""
    v = float(value)
    if v < 0.0:
        raise ParameterError(f"{name} must be greater than or equal to 0, got {v!r}")
    return v


def check_in_range(
    value: float,
    name: str,
    low: float,
    high: float,
    *,
    include_low: bool = False,
    include_high: bool = True,
) -> float:
    """Require ``value`` to sit inside the given interval."""
    v = float(value)
    low_ok = v >= low if include_low else v > low
    high_ok = v <= high if include_high else v < high
    if not (low_ok and high_ok):
        lb = "[" if include_low else "("
        rb = "]" if include_high else ")"
        raise ParameterError(f"{name} must lie in {lb}{low}, {high}{rb}, got {v!r}")
    return v


def validate_market_params(
    S: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> None:
    """Validate the market inputs common to every equity product.

    ``r`` is deliberately unconstrained in sign: negative risk-free rates are
    real, and a CNY desk routinely books a negative *net* carry once the index
    futures basis is folded in.

    Raises:
        ParameterError: on the first input that fails.
    """
    check_positive(S, "spot S")
    check_positive(T, "maturity T (years)")
    check_in_range(sigma, "volatility sigma", 0.0, MAX_VOLATILITY)
    check_non_negative(q, "dividend yield q")
    float(r)  # must be numeric, but any sign is admissible


def validate_vanilla(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> None:
    """Validate a European vanilla option's inputs."""
    validate_market_params(S, T, r, sigma, q)
    check_positive(K, "strike K")


def validate_barrier(
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> None:
    """Validate a single-barrier option's inputs.

    Does not require the barrier to sit on a particular side of spot: an
    already-breached barrier is a legitimate state to value (the option is
    simply worth its rebate or its vanilla continuation), and refusing it here
    would make live revaluation impossible.
    """
    validate_vanilla(S, K, T, r, sigma, q)
    check_positive(H, "barrier H")


def validate_snowball(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    ko_pct: float,
    ki_pct: float,
    coupon_pa: float,
    obs_freq: int,
    ki_obs_freq: int,
    lockup_months: int,
) -> None:
    """Validate a snowball autocallable's inputs.

    Beyond the usual bounds this enforces the structural relations that make
    the product a snowball at all -- knock-in strictly below knock-out, and a
    lock-up shorter than the trade itself. A lock-up that swallowed the whole
    tenor would silently turn the note into a plain short put.

    Raises:
        ParameterError: on the first input that fails.
    """
    validate_market_params(S0, T, r, sigma, q)
    check_positive(ko_pct, "knock-out level KO_pct")
    check_positive(ki_pct, "knock-in level KI_pct")
    check_non_negative(coupon_pa, "coupon_pa")

    if ki_pct >= ko_pct:
        raise ParameterError(
            f"knock-in level ({ki_pct:.4f}) must sit strictly below the knock-out level "
            f"({ko_pct:.4f}); otherwise the two barriers cross and the payoff is undefined"
        )

    if obs_freq < 1:
        raise ParameterError(
            f"obs_freq (knock-out observations per year) must be >= 1, got {obs_freq}"
        )
    if ki_obs_freq < 1:
        raise ParameterError(
            f"ki_obs_freq (knock-in observations per year) must be >= 1, got {ki_obs_freq}"
        )

    if lockup_months < 0:
        raise ParameterError(f"lockup_months must be >= 0, got {lockup_months}")
    lockup_years = lockup_months / 12.0
    if lockup_years >= T:
        raise ParameterError(
            f"lockup of {lockup_months} months ({lockup_years:.4f}y) must be shorter than the "
            f"tenor T={T:.4f}y, otherwise no knock-out observation can ever occur"
        )


def ensure_finite(values: Iterable[float], name: str) -> None:
    """Require every element to be a finite number.

    Guards engine boundaries: a NaN that reaches a PDE grid or an MC path array
    propagates silently and surfaces as a plausible-looking price.
    """
    import math

    for i, v in enumerate(values):
        if not math.isfinite(float(v)):
            raise ParameterError(f"{name}[{i}] must be finite, got {v!r}")
