"""Checking the parity forward against the index future.

Two forwards that must agree
----------------------------
Put-call parity implies a forward from the option quotes. The index future
quotes one directly. Verified across all 36 expiry tags in the panel with zero
exceptions, the future's ``S_INFO_DELISTDATE`` equals the option's
``S_INFO_MATURITYDATE`` -- the two instruments expire on the same day, so the
futures settlement *is* the forward to the option's own expiry. No stitching,
no interpolation between futures maturities, no basis adjustment, and therefore
no modelling choice hiding inside the comparison.

That exactness is what makes this worth building. The forward sets the
moneyness coordinate the whole surface is parameterised in, and until now
nothing independent could confirm it.

What this check does **not** measure: the discount rate
-------------------------------------------------------
This module was designed on the premise that a forward gap growing with tenor
is the signature of a mis-set rate, and that the gap could therefore be read as
a funding spread. **That premise is wrong, and the data says so unambiguously.**

Measured on IO at 2024-12-25, the rate's leverage on the parity forward:

======  ============  =======================  ====================
 tau     dF / 100bp    as a fraction of F       gap vs the future
======  ============  =======================  ====================
0.063      -0.013            -0.0003%                +0.103%
0.159      -0.018            -0.0005%                +0.162%
0.236      +0.274            +0.0069%                +0.298%
0.485      +0.724            +0.0182%                +0.457%
======  ============  =======================  ====================

The gaps are ten to a thousand times larger than a full percentage point of
rate can move the forward. Closing them with the rate would require changes of
**+2,508 bp, +4,346 bp, -30,940 bp and -35,654 bp** -- and on the two short
expiries the leverage has the *opposite sign* to the gap, so no rate closes
them at all. Solving for the reconciling rate numerically returns -44.9%,
-25.8%, +40.1% or no root in ±50%.

The reason is structural. With ``D`` fixed, parity gives
``F = mean[(C - P)/D + K]``, so ``dF/dr = T * mean(C - P)/D``. Near the money
``mean(C - P)`` is small, and its *sign* depends on whether the traded strikes
happen to sit above or below the forward -- which is why the leverage flips
between the short and long expiries above. The rate simply has no purchase on
this estimator.

This reconfirms and extends Phase 2a's finding. Phase 2a measured that implying
a rate from options alone is ill-posed (it returned 12.48%) and concluded the
rate must be an input. Options *plus futures* do not rescue it: the rate's
leverage is far below the gap, so the joint system is no better identified for
this purpose. The government curve stays an input, and it stays unadjusted by
anything measured here.

What the gap does measure
-------------------------
Total forward disagreement, 0.08% to 0.46% of forward on the measured dates --
and every gap exceeds its slice's own parity residual (0.03% to 0.24%), so this
is signal rather than quote noise. The residual candidates are a systematic
call-versus-put settlement bias, which maps straight into
``mean[(C - P)/D + K]``, and whatever asynchrony remains between the option and
futures settlement conventions. Attributing it further would need data this
panel does not contain, so the check reports the disagreement and bounds it
instead of explaining it.

That is still worth having. A 0.4% forward error is a 0.4% shift in the
moneyness coordinate, which lands directly on the fitted smile.

Coverage, stated rather than implied
------------------------------------
Only four futures list at a time against six option expiries, so **the two
longest expiries are never checked** -- including the one that matters most for
a one-to-three-year snowball. Coverage is reported per date.

Validation output only. Nothing here feeds a price, and the moment the gap were
used to adjust a rate or a forward, the independence that makes it evidence
would be gone.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from otcpricer.marketdata.forward import imply_forward
from otcpricer.marketdata.quotes import OptionChain, OptionSlice
from otcpricer.marketdata.rates import RateCurve

__all__ = [
    "RATE_BUMP",
    "ForwardCheck",
    "ForwardCheckReport",
    "check_forwards",
]

#: The bump used to measure the rate's leverage on the parity forward, in
#: absolute terms: 100 bp. Large enough that the answer is not noise, and the
#: unit a reader thinks in.
RATE_BUMP = 0.01


@dataclass(frozen=True, slots=True)
class ForwardCheck:
    """One expiry's parity forward against its matching future.

    Attributes:
        expiry: The shared expiry date, the same for both instruments.
        expiry_tag: The feed's expiry key, e.g. ``"2501"``.
        year_fraction: Time to expiry in years.
        n_paired: Paired strikes behind the parity forward.
        forward_parity: Forward implied by put-call parity.
        forward_futures: The future's settlement.
        rate_used: The rate the parity forward was computed at.
        parity_residual_relative: Parity's own fit quality, as a fraction of
            the forward. Bounds how much of the gap can be read as signal.
        forward_sensitivity: Change in the parity forward per
            :data:`RATE_BUMP` of rate, in price units. Signed, and its sign is
            not fixed -- it follows whether the traded strikes sit above or
            below the forward.
    """

    expiry: date
    expiry_tag: str
    year_fraction: float
    n_paired: int
    forward_parity: float
    forward_futures: float
    rate_used: float
    parity_residual_relative: float
    forward_sensitivity: float

    @property
    def gap_points(self) -> float:
        """Parity minus futures, in price units."""
        return self.forward_parity - self.forward_futures

    @property
    def gap_relative(self) -> float:
        """The gap as a fraction of the futures forward.

        The headline number, and the one that matters downstream: it is also
        the error in the moneyness coordinate the surface is fitted in.
        """
        return self.gap_points / self.forward_futures

    @property
    def rate_change_to_close(self) -> float:
        """The rate move that would close this gap, in absolute rate terms.

        Kept precisely because the answer is absurd. It is the quantitative
        demonstration that this gap is not a statement about the rate: on the
        measured dates it returns tens of thousands of basis points, and when
        the sensitivity has the opposite sign to the gap it returns a value
        pointing the wrong way entirely.

        ``nan`` when the forward does not respond to the rate at all.
        """
        if self.forward_sensitivity == 0.0:
            return float("nan")
        return float(self.gap_points / self.forward_sensitivity * RATE_BUMP)

    @property
    def rate_can_explain_the_gap(self) -> bool:
        """Whether a plausible rate error could account for this gap.

        "Plausible" is 200 bp, which is already wider than the panel's entire
        1-year yield range. On the measured dates this is false everywhere,
        which is the point.
        """
        needed = self.rate_change_to_close
        return bool(np.isfinite(needed) and abs(needed) <= 0.02)

    @property
    def explained_by_parity_noise(self) -> bool:
        """Whether the gap is smaller than parity's own residual.

        When true there is nothing here to interpret: the two forwards agree to
        within the precision of the quotes behind one of them.
        """
        return abs(self.gap_relative) <= self.parity_residual_relative

    def as_dict(self) -> dict[str, object]:
        return {
            "expiry": self.expiry.isoformat(),
            "expiry_tag": self.expiry_tag,
            "year_fraction": round(self.year_fraction, 6),
            "n_paired": self.n_paired,
            "forward_parity": round(self.forward_parity, 4),
            "forward_futures": round(self.forward_futures, 4),
            "gap_points": round(self.gap_points, 4),
            "gap_relative": round(self.gap_relative, 8),
            "rate_used": round(self.rate_used, 8),
            "parity_residual_relative": round(self.parity_residual_relative, 8),
            "forward_sensitivity_per_100bp": round(self.forward_sensitivity, 6),
            "rate_change_to_close_bp": (
                None
                if not np.isfinite(self.rate_change_to_close)
                else round(self.rate_change_to_close * 1e4, 1)
            ),
            "rate_can_explain_the_gap": self.rate_can_explain_the_gap,
            "explained_by_parity_noise": self.explained_by_parity_noise,
        }


@dataclass(frozen=True, slots=True)
class ForwardCheckReport:
    """Every checkable expiry on one date, and what could not be checked.

    Attributes:
        underlying_code: The option product.
        as_of: Observation date.
        checks: One row per expiry that had a matching future, by tenor.
        n_option_expiries: How many expiries the chain carried.
        unchecked_expiries: Expiry tags with no listed future, longest last.
    """

    underlying_code: str
    as_of: date
    checks: tuple[ForwardCheck, ...]
    n_option_expiries: int
    unchecked_expiries: tuple[str, ...]

    @property
    def coverage(self) -> float:
        """Fraction of the chain's expiries that had a matching future."""
        if self.n_option_expiries == 0:
            return 0.0
        return len(self.checks) / self.n_option_expiries

    @property
    def worst_gap_relative(self) -> float:
        """Largest absolute forward disagreement, as a fraction of forward."""
        if not self.checks:
            return float("nan")
        return max(abs(c.gap_relative) for c in self.checks)

    @property
    def median_gap_relative(self) -> float:
        """Typical disagreement. A median, so one stale slice cannot set it."""
        if not self.checks:
            return float("nan")
        return float(np.median([c.gap_relative for c in self.checks]))

    @property
    def gap_grows_with_tenor(self) -> bool:
        """Whether the disagreement increases monotonically in tenor.

        True on 5 of the 6 measured root-date combinations, rising from 0.02%
        at a fortnight to 0.46% at six months; HO on 2024-12-25 dips at its
        third expiry and so reads False. This is the pattern that made a rate
        story look plausible, so it is recorded as the observation it is rather
        than left to be rediscovered and misread the same way.

        The rate is ruled out (see the module docstring). The leading remaining
        candidate is liquidity: the share of untraded strikes rises with tenor,
        the exchange marks those theoretically, and a theoretical mark carrying
        a slightly different forward than the futures market would produce
        exactly this shape. Not claimed -- separating it from a call-versus-put
        settlement bias needs data this panel does not have.
        """
        if len(self.checks) < 3:
            return False
        gaps = [abs(c.gap_relative) for c in self.checks]
        # strict=False is deliberate: the slice is one shorter, pairwise.
        return all(b >= a for a, b in zip(gaps, gaps[1:], strict=False))

    @property
    def rate_explains_any_gap(self) -> bool:
        """Whether a plausible rate error accounts for *any* expiry's gap.

        False on all measured dates. Reported because it is the conclusion that
        keeps the government curve an input rather than something this check
        quietly tunes.
        """
        return any(c.rate_can_explain_the_gap for c in self.checks)

    @property
    def all_gaps_exceed_parity_noise(self) -> bool:
        """Whether every gap is bigger than its own slice's parity residual.

        When true the disagreement is signal, not the precision limit of the
        quotes behind it -- which is what gives the check resolving power.
        """
        return bool(self.checks) and not any(c.explained_by_parity_noise for c in self.checks)

    def as_dict(self) -> dict[str, object]:
        return {
            "underlying_code": self.underlying_code,
            "as_of": self.as_of.isoformat(),
            "n_option_expiries": self.n_option_expiries,
            "n_checked": len(self.checks),
            "coverage": round(self.coverage, 4),
            "unchecked_expiries": list(self.unchecked_expiries),
            "median_gap_relative": round(self.median_gap_relative, 8),
            "worst_gap_relative": round(self.worst_gap_relative, 8),
            "gap_grows_with_tenor": self.gap_grows_with_tenor,
            "rate_explains_any_gap": self.rate_explains_any_gap,
            "all_gaps_exceed_parity_noise": self.all_gaps_exceed_parity_noise,
            "checks": [c.as_dict() for c in self.checks],
            "note": (
                "Validation output only; nothing here feeds a price. The gap is "
                "a forward consistency check, NOT a reading on the discount "
                "rate: the rate moves this forward by under 0.05% per 100bp, "
                "far less than the gaps, so rate_change_to_close comes back "
                "absurd by construction. See the module docstring."
            ),
        }


def _sensitivity(option_slice: OptionSlice, rate: float) -> float:
    """Change in the parity forward per :data:`RATE_BUMP`, by central difference.

    Measured rather than derived, so it reflects whatever the estimator and the
    quote screening actually do -- including the sign flip between expiries,
    which an analytic formula assuming a symmetric strike grid would miss.
    """
    up = imply_forward(option_slice, rate=rate + RATE_BUMP / 2.0)
    down = imply_forward(option_slice, rate=rate - RATE_BUMP / 2.0)
    if not (np.isfinite(up.forward) and np.isfinite(down.forward)):
        return 0.0
    return float(up.forward - down.forward)


def check_forwards(
    chain: OptionChain,
    futures: tuple[tuple[date, float], ...],
    *,
    rate: float = 0.02,
    rate_curve: RateCurve | None = None,
) -> ForwardCheckReport:
    """Compare each expiry's parity forward with its matching future.

    Futures are matched on the **exact expiry date**, never on a nearest
    neighbour. A nearest-neighbour match would reintroduce the interpolation
    this check exists to avoid, and would silently compare a forward against
    the wrong maturity whenever a future is missing. Expiries with no future
    are reported as unchecked instead.

    Args:
        chain: The option chain, which supplies the parity forwards.
        futures: ``(expiry, settlement)`` pairs, as
            :meth:`~otcpricer.marketdata.wind.WindPanel.futures` returns.
        rate: Flat rate for the parity forward.
        rate_curve: A curve to read each slice's rate from instead.

    Returns:
        A :class:`ForwardCheckReport`.
    """
    futures_by_expiry = dict(futures)

    checks: list[ForwardCheck] = []
    unchecked: list[str] = []

    for option_slice in chain.slices:
        slice_rate = (
            rate if rate_curve is None else float(rate_curve.zero_rate(option_slice.year_fraction))
        )
        forward = imply_forward(option_slice, rate=slice_rate)
        if not np.isfinite(forward.forward):
            unchecked.append(option_slice.expiry_tag)
            continue

        settlement = futures_by_expiry.get(option_slice.expiry)
        if settlement is None:
            unchecked.append(option_slice.expiry_tag)
            continue

        checks.append(
            ForwardCheck(
                expiry=option_slice.expiry,
                expiry_tag=option_slice.expiry_tag,
                year_fraction=option_slice.year_fraction,
                n_paired=len(option_slice.paired_strikes()),
                forward_parity=float(forward.forward),
                forward_futures=float(settlement),
                rate_used=slice_rate,
                parity_residual_relative=float(forward.residual_relative),
                forward_sensitivity=_sensitivity(option_slice, slice_rate),
            )
        )

    return ForwardCheckReport(
        underlying_code=chain.underlying.code,
        as_of=chain.as_of,
        checks=tuple(sorted(checks, key=lambda c: c.year_fraction)),
        n_option_expiries=len(chain.slices),
        unchecked_expiries=tuple(unchecked),
    )
