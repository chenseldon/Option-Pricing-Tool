"""Turn an option chain into the implied-volatility points a surface fits to.

Pricing in forward terms, with no new code
------------------------------------------
Once the forward is known, the natural formula is Black-76 on the forward
rather than Black-Scholes on spot. No separate implementation is needed:
Black-76 is exactly the existing :mod:`otcpricer.models.bsm` with ``S = F`` and
``q = r``. Substituting those, ``S e^{-qT}`` becomes ``F D``, ``K e^{-rT}``
becomes ``K D``, and the drift term in ``d1`` cancels to leave
``[ln(F/K) + sigma^2 T / 2] / (sigma sqrt(T))`` -- Black's formula. So the
whole Phase 1 pricing and implied-vol machinery, including its Newton/Brent
solver, is reused unchanged.

Which quotes are used, and why the rest are thrown away
-------------------------------------------------------
Screening is the substance of this module. The board gives last traded prices
with no bid/ask, so a slice's quotes are non-synchronous and the illiquid ones
can be badly stale. Fitting a surface through all of them indiscriminately
produces a surface that fits noise.

Four filters, in order:

1. **Traded today.** A zero or absent last price means the contract did not
   trade; its "price" is yesterday's settlement and says nothing about today.
2. **Out-of-the-money only.** Calls above the forward, puts below. Standard
   desk practice: an out-of-the-money option is nearly all optionality, so its
   price is sensitive to volatility, while an in-the-money one is mostly
   intrinsic value and its implied vol is dominated by quote error.
3. **Inside the no-arbitrage band.** A price outside it has no implied
   volatility at all, so the solver would simply fail.
4. **Volatility identifiable.** Phase 1's
   :func:`~otcpricer.models.bsm.vol_uncertainty` screen. Deep in-the-money and
   very short-dated quotes price at intrinsic to full double precision: vega
   underflows and every volatility reproduces the quote, so the "implied vol"
   is an artefact of where the solver happened to stop.
5. **Resolvable against the tick.** The same screen again, with the price
   tolerance set to the *minimum tick* rather than the solver's numerical
   tolerance. This catches a different failure, and the sharpest one in this
   data.

Why the tick screen matters
---------------------------
A quote resting on the minimum tick is a floor, not a price. On the captured
one-day expiry, the far wing priced at 0.0001 -- a single tick -- and inverting
it returned an implied volatility of **129.65%** at twenty standard deviations
out of the money. The option is worth essentially nothing; the tick is all
that is being measured.

The solver's own identifiability screen does not catch this, because a
volatility *does* exist that reproduces 0.0001. The quote is identifiable and
meaningless at the same time.

The principled test is how much volatility one tick of price is worth:
``tick / vega``. Where that exceeds a couple of vol points, the quote cannot
resolve volatility to any useful precision. On the captured data this rejects
nine of the one-day expiry's thirteen points while keeping every point of the
one-month expiry, whose worst case sits six times inside the limit -- so it
separates on the strength of the quote rather than on a threshold tuned to the
sample.

A sixth filter runs afterwards, once the slice's own at-the-money level is
known: **moneyness within a few standard deviations**.

Why the far wing has to go
--------------------------
Beyond roughly three standard deviations an option's vega is negligible against
quote noise, and its price stops being a volatility and becomes the market
maker's floor. The captured 300ETF 0.156-year slice shows it plainly: across
``k`` from +0.249 to +0.382 -- a 14% move in strike -- the price barely moves,
0.0022, 0.0020, 0.0019, 0.0018. A real smile would fall far faster. Because the
price is nearly flat while the strike rises, the implied volatility is forced
upward, reaching 0.395.

Those quotes pass the tick screen (18-22 ticks) and violate no arbitrage bound,
so nothing else catches them. But they were distorting the fit, and not only
locally: measured against the cap,

======  ======  =========  =========  ======
cap     points  rms (vp)   max (vp)   gamma
======  ======  =========  =========  ======
none    52      1.779      7.906      0.486
4.0     50      1.194      4.003      0.500
3.5     49      1.061      2.974      0.500
3.0     42      0.654      1.977      0.216
======  ======  =========  =========  ======

``gamma`` sits pinned against its upper bound until the far wing is removed and
then falls to 0.216, which says those quotes were bending the *global* shape
parameters rather than just fitting badly in place.

Three standard deviations is the default, and is generous rather than
aggressive: desks commonly fit the 5-delta to 95-delta range, around 1.65
standard deviations. It is configurable, and the sensitivity above is recorded
so the choice can be re-examined rather than taken on trust.

Every rejection is recorded with its reason and reported. A pipeline that
silently discards half its inputs is worse than one that fails.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from otcpricer.core.exceptions import OtcPricerError
from otcpricer.core.logging import get_logger
from otcpricer.core.types import OptionType
from otcpricer.marketdata.forward import DEFAULT_RATE, ForwardResult, imply_forward
from otcpricer.marketdata.quotes import OptionChain, OptionSlice
from otcpricer.models import bsm

__all__ = [
    "MAX_MONEYNESS_SD",
    "MAX_TICK_VOL_UNCERTAINTY",
    "MAX_VOL_UNCERTAINTY",
    "MIN_POINTS_PER_SLICE",
    "SliceInput",
    "SurfaceInput",
    "VolPoint",
    "extract_surface_input",
]

#: Vol-resolution limit for the solver's own identifiability screen, in vol
#: units. Looser than the solver's default: a hundredth is plenty for a surface
#: fit, and tightening it discards usable wings.
MAX_VOL_UNCERTAINTY = 1e-2

#: Limit on the volatility uncertainty implied by one tick of price, in vol
#: units. Two vol points: beyond that a quote cannot resolve volatility usefully
#: and is reporting the tick rather than the market. See the module docstring
#: for the measured separation this achieves.
MAX_TICK_VOL_UNCERTAINTY = 2e-2

#: Largest |log-moneyness| to accept, in units of the slice's own standard
#: deviation ``sigma * sqrt(T)``. Beyond this an option's vega is negligible
#: against quote noise and its price reflects a market maker's floor rather
#: than a volatility. See the module docstring for the measured effect.
MAX_MONEYNESS_SD = 3.0

#: Fewest surviving points for a slice to be worth fitting. Below this the
#: smile is not determined and including the slice only adds noise.
MIN_POINTS_PER_SLICE = 4


@dataclass(frozen=True, slots=True)
class VolPoint:
    """One implied-volatility observation.

    Attributes:
        strike: Strike price.
        option_type: Which leg supplied it.
        price: The quote used.
        implied_vol: Solved Black-76 implied volatility.
        log_moneyness: ``log(K / F)``, the surface's native coordinate.
        total_variance: ``sigma^2 * T``, which is what SVI parameterises.
        vega: Black-76 vega per percentage point, used as a fit weight.
        year_fraction: Time to expiry in years.
        tick_vol_uncertainty: Volatility one tick of price is worth here. The
            precision of this observation, carried alongside it so a fit can
            weight by it and a report can show it.
    """

    strike: float
    option_type: OptionType
    price: float
    implied_vol: float
    log_moneyness: float
    total_variance: float
    vega: float
    year_fraction: float
    tick_vol_uncertainty: float = 0.0


@dataclass(frozen=True, slots=True)
class SliceInput:
    """Surface inputs for one expiry, plus what was rejected.

    Attributes:
        expiry_tag: Feed expiry key, e.g. ``"2611"``.
        year_fraction: Time to expiry in years.
        forward: The implied forward and its quality evidence.
        points: Surviving implied-volatility observations.
        rejections: Reason counts, so the screening is auditable.
        usable: Whether this slice should enter the fit.
        reason: Why it should not, when it should not.
    """

    expiry_tag: str
    year_fraction: float
    forward: ForwardResult
    points: tuple[VolPoint, ...]
    rejections: dict[str, int] = field(default_factory=dict)
    usable: bool = True
    reason: str = ""

    @property
    def atm_total_variance(self) -> float:
        """Total variance at the money, by interpolation in log-moneyness.

        This is SSVI's ``theta`` for this expiry, and the anchor its term
        structure is built on.
        """
        if not self.points:
            return float("nan")
        moneyness = np.array([p.log_moneyness for p in self.points])
        variance = np.array([p.total_variance for p in self.points])
        order = np.argsort(moneyness)
        return float(np.interp(0.0, moneyness[order], variance[order]))

    def as_dict(self) -> dict[str, object]:
        """Flat summary for reports."""
        return {
            "expiry_tag": self.expiry_tag,
            "year_fraction": round(self.year_fraction, 6),
            "forward": round(self.forward.forward, 6),
            "n_points": len(self.points),
            "atm_total_variance": round(self.atm_total_variance, 8),
            "atm_vol": round(
                float(np.sqrt(max(self.atm_total_variance, 0.0) / self.year_fraction)), 6
            )
            if self.year_fraction > 0
            else None,
            "rejections": dict(self.rejections),
            "usable": self.usable,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class SurfaceInput:
    """Everything a surface fit needs, for one underlying on one date."""

    underlying_code: str
    as_of: str
    source: str
    rate: float
    slices: tuple[SliceInput, ...]

    @property
    def usable_slices(self) -> tuple[SliceInput, ...]:
        return tuple(s for s in self.slices if s.usable)

    @property
    def n_points(self) -> int:
        return sum(len(s.points) for s in self.usable_slices)

    @property
    def rejections(self) -> dict[str, int]:
        """Discarded quotes by reason, pooled across expiries, worst first.

        Every quote in the chain is either kept or counted here exactly once,
        so this and :attr:`n_points` reconcile against the raw quote count.
        That identity is what makes the screening auditable instead of merely
        documented, and it is asserted in the test suite.
        """
        pooled: Counter[str] = Counter()
        for slice_input in self.slices:
            pooled.update(slice_input.rejections)
        return dict(pooled.most_common())

    @property
    def n_rejected(self) -> int:
        """How many quotes were discarded in total."""
        return sum(self.rejections.values())

    def as_dict(self) -> dict[str, object]:
        """Flat summary for reports and API responses."""
        return {
            "underlying": self.underlying_code,
            "as_of": self.as_of,
            "source": self.source,
            "rate": self.rate,
            "n_usable_slices": len(self.usable_slices),
            "n_points": self.n_points,
            "n_rejected": self.n_rejected,
            "rejections": self.rejections,
            "slices": [s.as_dict() for s in self.slices],
        }


def _extract_slice(
    option_slice: OptionSlice,
    *,
    rate: float,
    max_vol_uncertainty: float,
    max_tick_vol_uncertainty: float,
    max_moneyness_sd: float,
    min_points: int,
) -> SliceInput:
    """Screen one expiry's quotes and solve for implied volatilities."""
    forward = imply_forward(option_slice, rate=rate)
    year_fraction = option_slice.year_fraction
    rejections: Counter[str] = Counter()

    if not forward.usable:
        return SliceInput(
            expiry_tag=option_slice.expiry_tag,
            year_fraction=year_fraction,
            forward=forward,
            points=(),
            rejections=dict(rejections),
            usable=False,
            reason=f"forward unusable: {forward.reason}",
        )

    F = forward.forward
    # Black-76 via the existing BSM implementation: S = F and q = r.
    pricing_rate = rate
    points: list[VolPoint] = []

    for quote in option_slice.quotes:
        if not quote.has_traded:
            rejections["not traded today"] += 1
            continue

        is_call = quote.option_type is OptionType.CALL
        # Out-of-the-money only: calls above the forward, puts below.
        if (is_call and quote.strike < F) or (not is_call and quote.strike > F):
            rejections["in the money"] += 1
            continue

        lower, upper = bsm.no_arbitrage_bounds(
            quote.option_type, F, quote.strike, year_fraction, pricing_rate, pricing_rate
        )
        if not lower - 1e-12 <= quote.price <= upper + 1e-12:
            rejections["outside no-arbitrage bounds"] += 1
            continue

        try:
            implied = bsm.implied_volatility(
                quote.option_type,
                quote.price,
                F,
                quote.strike,
                year_fraction,
                pricing_rate,
                pricing_rate,
                max_vol_uncertainty=max_vol_uncertainty,
            )
        except OtcPricerError as exc:
            label = (
                "volatility not identifiable"
                if "does not pin down" in str(exc)
                else "implied vol solve failed"
            )
            rejections[label] += 1
            continue

        # How much volatility is one tick of price worth here? Where that is
        # large, the quote is reporting the tick rather than the market.
        tick_uncertainty = bsm.vol_uncertainty(
            quote.option_type,
            F,
            quote.strike,
            year_fraction,
            pricing_rate,
            implied,
            pricing_rate,
            price_tolerance=option_slice.underlying.tick_size,
        )
        if tick_uncertainty > max_tick_vol_uncertainty:
            rejections["price at tick resolution"] += 1
            continue

        vega = float(
            bsm.vega(
                quote.option_type,
                F,
                quote.strike,
                year_fraction,
                pricing_rate,
                implied,
                pricing_rate,
            )
        )
        points.append(
            VolPoint(
                strike=quote.strike,
                option_type=quote.option_type,
                price=quote.price,
                implied_vol=implied,
                log_moneyness=float(np.log(quote.strike / F)),
                total_variance=implied * implied * year_fraction,
                vega=vega,
                year_fraction=year_fraction,
                tick_vol_uncertainty=float(tick_uncertainty),
            )
        )

    points.sort(key=lambda p: p.log_moneyness)

    # Moneyness cap, applied last because it needs the slice's own at-the-money
    # level, which is only known once the implied vols exist. Total variance at
    # the money is interpolated from the points just collected, and its square
    # root is this expiry's standard deviation of log-return.
    if points and np.isfinite(max_moneyness_sd):
        variance = np.array([p.total_variance for p in points])
        moneyness = np.array([p.log_moneyness for p in points])
        atm_variance = float(np.interp(0.0, moneyness, variance))
        sigma_sqrt_t = float(np.sqrt(max(atm_variance, 1e-12)))
        kept = [p for p in points if abs(p.log_moneyness) <= max_moneyness_sd * sigma_sqrt_t]
        rejections["beyond moneyness cap"] = len(points) - len(kept)
        if not rejections["beyond moneyness cap"]:
            del rejections["beyond moneyness cap"]
        points = kept

    usable = len(points) >= min_points
    reason = (
        "" if usable else f"only {len(points)} usable quotes survived screening, need {min_points}"
    )
    return SliceInput(
        expiry_tag=option_slice.expiry_tag,
        year_fraction=year_fraction,
        forward=forward,
        points=tuple(points),
        rejections=dict(rejections),
        usable=usable,
        reason=reason,
    )


def extract_surface_input(
    chain: OptionChain,
    *,
    rate: float = DEFAULT_RATE,
    max_vol_uncertainty: float = MAX_VOL_UNCERTAINTY,
    max_tick_vol_uncertainty: float = MAX_TICK_VOL_UNCERTAINTY,
    max_moneyness_sd: float = MAX_MONEYNESS_SD,
    min_points: int = MIN_POINTS_PER_SLICE,
) -> SurfaceInput:
    """Extract implied-volatility points from a chain, with screening.

    Args:
        chain: The option chain.
        rate: Continuously compounded rate, used for both the forward and the
            Black-76 discounting.
        max_vol_uncertainty: Identifiability limit for accepting a quote.
        max_tick_vol_uncertainty: Limit on the volatility one tick of price is
            worth. Quotes above it are reporting the tick, not the market.
        max_moneyness_sd: Largest |log-moneyness| to accept, in slice standard
            deviations. Pass ``float('inf')`` to keep the whole wing.
        min_points: Fewest surviving points for a slice to be usable.

    Returns:
        A :class:`SurfaceInput` carrying the surviving points per expiry and a
        full account of what was rejected and why.
    """
    slices = tuple(
        _extract_slice(
            option_slice,
            rate=rate,
            max_vol_uncertainty=max_vol_uncertainty,
            max_tick_vol_uncertainty=max_tick_vol_uncertainty,
            max_moneyness_sd=max_moneyness_sd,
            min_points=min_points,
        )
        for option_slice in chain.slices
    )

    result = SurfaceInput(
        underlying_code=chain.underlying.code,
        as_of=chain.as_of.isoformat(),
        source=chain.source,
        rate=rate,
        slices=slices,
    )
    get_logger(__name__).info(
        "surface input extracted",
        extra={
            "underlying": chain.underlying.code,
            "usable_slices": len(result.usable_slices),
            "points": result.n_points,
        },
    )
    return result
