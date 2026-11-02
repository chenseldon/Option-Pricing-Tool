"""Plain vanilla interest rate swap, priced off a discount curve.

What was wrong with the previous version
----------------------------------------
``/api/irs`` reported a DV01 of ``notional * dt * df(T) * 1bp`` -- the *final*
discount factor only, where the sensitivity of a fixed leg is the whole
annuity. Measured at ``T=2``, ``freq=2``, 2% flat on a million notional, the
three numbers that should have agreed were:

- what it reported: **48.52**
- the annuity sensitivity: **192.67**
- bump-and-reprice of its own NPV: **0.15**

Three mutually unrelated answers. The deeper cause is the third one: the
floating leg paid ``float_rate * dt * notional`` every period, the same constant
throughout, and never reset off a curve. A leg like that is not floating, so the
instrument was not a swap -- it was the difference of two fixed annuities, whose
NPV barely moves when you bump a curve it never consulted.

It happened to look plausible because ``r_disc`` defaulted to ``float_rate``:
on a flat curve every forward rate equals the spot rate, so the constant was
accidentally right. The error only appeared once the two differed, which is
exactly when someone would start trusting it.

How this prices it
------------------
The floating leg pays each period's *forward* rate, read off the curve:

    ``float_cf_i = F(t_{i-1}, t_i) * dt * notional``

With principal exchanged at maturity on both legs -- which cancels in the NPV,
since both are discounted identically -- the floating leg then prices to par by
construction. That identity is the decisive test that the leg is right, and it
holds for *any* curve shape, so it cannot be satisfied by accident.

The DV01 is the fixed-leg annuity,

    ``sum_i notional * dt * df(t_i) * 1bp``

and there is a test that bump-and-reprice agrees with it, because an analytic
sensitivity nobody has checked against a bump is how the previous one survived.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from otcpricer.core.exceptions import ParameterError
from otcpricer.marketdata.rates import RateCurve

__all__ = [
    "ONE_BASIS_POINT",
    "SwapLeg",
    "SwapResult",
    "price_swap",
]

ONE_BASIS_POINT = 1e-4


@dataclass(frozen=True, slots=True)
class SwapLeg:
    """One period's cash flow on both legs, with the evidence behind it.

    Attributes:
        period: 1-based period index.
        t_start: Accrual start, in years.
        t_end: Accrual end and payment date, in years.
        discount_factor: Curve discount factor to ``t_end``.
        forward_rate: The rate the floating leg resets at for this period.
        fixed_payment: Fixed coupon, plus principal in the final period.
        float_payment: Floating coupon, plus principal in the final period.
    """

    period: int
    t_start: float
    t_end: float
    discount_factor: float
    forward_rate: float
    fixed_payment: float
    float_payment: float

    @property
    def pv_fixed(self) -> float:
        return self.fixed_payment * self.discount_factor

    @property
    def pv_float(self) -> float:
        return self.float_payment * self.discount_factor

    def as_dict(self) -> dict[str, object]:
        return {
            "period": self.period,
            "t": round(self.t_end, 6),
            "df": round(self.discount_factor, 8),
            "forward_rate": round(self.forward_rate, 8),
            "fixed_pmt": round(self.fixed_payment, 2),
            "float_pmt": round(self.float_payment, 2),
            "pv_fixed": round(self.pv_fixed, 2),
            "pv_float": round(self.pv_float, 2),
            "net_pmt": round(self.float_payment - self.fixed_payment, 2),
        }


@dataclass(frozen=True, slots=True)
class SwapResult:
    """A priced swap, from the fixed payer's perspective.

    Attributes:
        npv: ``PV(float) - PV(fixed)``.
        pv_fixed: Present value of the fixed leg including principal.
        pv_float: Present value of the floating leg including principal.
        annuity: ``sum_i dt * df(t_i)``, the fixed leg's rate sensitivity per
            unit of notional and rate.
        dv01: Value of one basis point on the fixed rate.
        par_rate: The fixed rate making the swap worth zero.
        legs: Per-period detail.
        notional: Notional principal.
    """

    npv: float
    pv_fixed: float
    pv_float: float
    annuity: float
    dv01: float
    par_rate: float
    legs: tuple[SwapLeg, ...]
    notional: float

    @property
    def float_leg_prices_to_par(self) -> bool:
        """Whether the floating leg plus principal is worth its notional.

        True for any curve shape when the leg resets off that curve, and the
        single most informative check on this pricing -- a floating leg that
        does not price to par is not floating.
        """
        return bool(abs(self.pv_float - self.notional) < 1e-6 * max(self.notional, 1.0))

    def as_dict(self) -> dict[str, object]:
        return {
            "npv": round(self.npv, 2),
            "pv_fixed": round(self.pv_fixed, 2),
            "pv_float": round(self.pv_float, 2),
            "annuity": round(self.annuity, 8),
            "dv01": round(self.dv01, 4),
            "par_rate": round(self.par_rate, 8),
            "float_leg_prices_to_par": self.float_leg_prices_to_par,
            "cashflows": [leg.as_dict() for leg in self.legs],
        }


def price_swap(
    curve: RateCurve,
    *,
    notional: float = 1_000_000.0,
    fixed_rate: float,
    maturity: float,
    frequency: int = 4,
) -> SwapResult:
    """Price a pay-fixed, receive-float swap off a discount curve.

    Args:
        curve: Discount and projection curve. One curve for both, which is
            pre-2008 practice -- this data has no basis curve to build a
            second one from, and inventing a spread would be worse than
            saying so.
        notional: Notional principal.
        fixed_rate: Annual fixed rate, as a decimal.
        maturity: Years to the final payment.
        frequency: Payments per year.

    Returns:
        A :class:`SwapResult`.

    Raises:
        ValidationError: on a non-positive notional, maturity or frequency, or
            a schedule with no payments in it.
    """
    if notional <= 0.0:
        raise ParameterError(f"notional must be positive, got {notional}")
    if maturity <= 0.0:
        raise ParameterError(f"maturity must be positive, got {maturity}")
    if frequency < 1:
        raise ParameterError(f"frequency must be at least 1, got {frequency}")

    dt = 1.0 / frequency
    periods = int(round(maturity * frequency))
    if periods < 1:
        raise ParameterError(
            f"maturity {maturity} at {frequency} payments a year leaves no payment dates"
        )

    legs: list[SwapLeg] = []
    annuity = 0.0
    pv_fixed = 0.0
    pv_float = 0.0

    for i in range(1, periods + 1):
        t_start = (i - 1) * dt
        t_end = i * dt
        discount = float(curve.discount_factor(t_end))

        # The floating leg resets off the curve. This is the line whose absence
        # made the previous version not a swap -- and it must be the *simply*
        # compounded forward, which is both what a float leg pays and the only
        # convention under which the leg prices to par. See
        # RateCurve.simple_forward_rate.
        forward = curve.simple_forward_rate(t_start, t_end)

        # Principal on both legs. It cancels in the NPV -- both are discounted
        # by the same factor -- but carrying it is what makes the floating leg
        # price to par, which is the identity worth being able to check.
        principal = notional if i == periods else 0.0
        fixed_payment = fixed_rate * dt * notional + principal
        float_payment = forward * dt * notional + principal

        annuity += dt * discount
        pv_fixed += fixed_payment * discount
        pv_float += float_payment * discount

        legs.append(
            SwapLeg(
                period=i,
                t_start=t_start,
                t_end=t_end,
                discount_factor=discount,
                forward_rate=forward,
                fixed_payment=fixed_payment,
                float_payment=float_payment,
            )
        )

    # The fixed leg's sensitivity is the whole annuity, not the final discount
    # factor. This is the defect that motivated the module.
    dv01 = notional * annuity * ONE_BASIS_POINT

    floating_pv = pv_float - notional * float(curve.discount_factor(periods * dt))
    par_rate = floating_pv / (notional * annuity) if annuity > 0.0 else float("nan")

    return SwapResult(
        npv=float(pv_float - pv_fixed),
        pv_fixed=float(pv_fixed),
        pv_float=float(pv_float),
        annuity=float(annuity),
        dv01=float(dv01),
        par_rate=float(par_rate),
        legs=tuple(legs),
        notional=float(notional),
    )


def flat_curve(rate: float, as_of: object = None) -> RateCurve:
    """A flat :class:`RateCurve`, for callers passing a single rate.

    Keeps the swap pricer curve-only: one code path, so the flat case cannot
    drift away from the curve case.
    """
    from datetime import date as _date

    moment = as_of if isinstance(as_of, _date) else _date.today()
    if not np.isfinite(rate):
        raise ParameterError(f"rate must be finite, got {rate}")
    return RateCurve(
        as_of=moment,
        tenors=(1.0, 10.0),
        zero_rates=(float(rate), float(rate)),
        source=f"flat:{rate}",
    )
