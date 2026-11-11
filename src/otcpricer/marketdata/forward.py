"""Forward and discount factor from put-call parity.

A volatility surface is parameterised in log-moneyness ``k = log(K / F)``, so
the forward is the one market input it cannot do without. Parity supplies it
from the option board alone:

    C - P = (F - K) * D

Why the rate is an input and the forward is not
-----------------------------------------------
Fitting both ``F`` and ``D`` as free parameters is tempting -- the regression
supports it -- and wrong. Measured on a live 50ETF slice with 29 days to run:

==================  ==========  =========  ==========
Method              implied r   implied F  residual rms
==================  ==========  =========  ==========
r fixed at 1.5%     1.500%      2.85426    0.00257
r fixed at 2.0%     2.000%      2.85421    0.00250
r fixed at 2.5%     2.500%      2.85417    0.00243
free two-parameter  **12.480%** 2.85327    0.00162
==================  ==========  =========  ==========

The forward is essentially invariant: a 100bp change in the assumed rate moves
it by 0.003%. The implied *rate* is not identified at all -- the free fit
returns 12.48%, which is impossible for CNY, while improving the residual only
from 0.0025 to 0.0016. That extra parameter is fitting quote noise, because over
0.08 years the discount signal (``D`` around 0.990) is barely larger than the
parity residuals themselves.

So: take the rate as given, imply only the forward, and report the residual as
a data-quality measure. A proper discount curve is Phase 3's business.

The residual is not decoration. The board carries last traded prices with no
bid/ask, so quotes across a slice are non-synchronous; the residual is the most
direct measure of how badly, and a slice whose residual is too large should not
be used to build a surface at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from otcpricer.core.exceptions import MarketDataError
from otcpricer.core.logging import get_logger
from otcpricer.core.types import OptionType
from otcpricer.marketdata.quotes import OptionSlice

__all__ = [
    "DEFAULT_RATE",
    "MAX_PARITY_RESIDUAL_REL",
    "MIN_PAIRED_STRIKES",
    "ForwardCurve",
    "ForwardResult",
    "imply_forward",
]

#: Rate assumed when none is supplied. Roughly the 1-year CNY government yield.
#: Only the forward depends on it, and barely; see the module docstring.
DEFAULT_RATE = 0.02

#: Fewest paired strikes worth running a regression on. Below this the forward
#: is driven by two or three possibly stale quotes.
MIN_PAIRED_STRIKES = 4

#: Parity residual, relative to the forward, above which a slice is unusable.
#: 1% of forward is already loose for a liquid ETF; beyond it the quotes are
#: too far out of line with each other to carry volatility information.
MAX_PARITY_RESIDUAL_REL = 0.01


@dataclass(frozen=True, slots=True)
class ForwardResult:
    """Forward and discount factor for one expiry, with its quality evidence.

    Attributes:
        forward: Implied forward price of the underlying.
        discount_factor: ``exp(-r * T)`` for the rate used.
        rate: The rate assumed, echoed back so the result is self-describing.
        year_fraction: Time to expiry in years.
        parity_residual_rms: Root-mean-square parity residual, in price units.
        parity_residual_max: Largest absolute residual.
        n_strikes: Paired strikes used.
        usable: Whether the slice passed the quality screen.
        reason: Why it failed, when it did.
    """

    forward: float
    discount_factor: float
    rate: float
    year_fraction: float
    parity_residual_rms: float
    parity_residual_max: float
    n_strikes: int
    usable: bool
    reason: str = ""

    @property
    def residual_relative(self) -> float:
        """RMS residual as a fraction of the forward."""
        return self.parity_residual_rms / self.forward if self.forward else float("inf")

    @property
    def implied_carry(self) -> float:
        """Continuous carry rate implied by the forward, ``log(F/S)/T``.

        Requires a spot price to be meaningful, so it is not computed here;
        exposed as a reminder that ``F`` already embeds dividends and the
        futures basis, which is why nothing downstream needs a separate ``q``.
        """
        raise NotImplementedError(
            "carry needs a spot price; the surface works in forward terms and does not require it"
        )

    def as_dict(self) -> dict[str, float | int | bool | str]:
        """Flat dict for API responses and reports."""
        return {
            "forward": round(self.forward, 6),
            "discount_factor": round(self.discount_factor, 8),
            "rate": self.rate,
            "year_fraction": round(self.year_fraction, 6),
            "parity_residual_rms": round(self.parity_residual_rms, 8),
            "parity_residual_max": round(self.parity_residual_max, 8),
            "parity_residual_relative": round(self.residual_relative, 8),
            "n_strikes": self.n_strikes,
            "usable": self.usable,
            "reason": self.reason,
        }


def imply_forward(
    option_slice: OptionSlice,
    *,
    rate: float = DEFAULT_RATE,
    min_strikes: int = MIN_PAIRED_STRIKES,
    max_residual_relative: float = MAX_PARITY_RESIDUAL_REL,
    require_traded: bool = True,
) -> ForwardResult:
    """Imply the forward for one expiry from put-call parity.

    With ``D = exp(-r * T)`` fixed, parity rearranges to

        ``F = mean_over_strikes[ (C - P) / D + K ]``

    which is the least-squares solution for the single free parameter. Averaging
    over strikes is what makes it robust: an individual stale quote shifts the
    estimate by ``1 / n`` of its error instead of all of it.

    Args:
        option_slice: One expiry's quotes.
        rate: Continuously compounded rate for the discount factor.
        min_strikes: Fewest paired strikes to attempt a fit.
        max_residual_relative: Quality threshold, relative to the forward.
        require_traded: Use only strikes where both legs actually traded today.
            A previous settlement is a price from yesterday's close and carries
            no information about today's volatility.

    Returns:
        A :class:`ForwardResult`. A slice that fails the screen comes back with
        ``usable=False`` and a reason rather than raising, so a chain with one
        bad expiry still yields a surface from the rest.

    Raises:
        MarketDataError: only if the slice has no paired strikes at all, which
            means the data is structurally wrong rather than merely poor.
    """
    year_fraction = option_slice.year_fraction
    if year_fraction <= 0.0:
        return ForwardResult(
            forward=float("nan"),
            discount_factor=float("nan"),
            rate=rate,
            year_fraction=year_fraction,
            parity_residual_rms=float("inf"),
            parity_residual_max=float("inf"),
            n_strikes=0,
            usable=False,
            reason=f"expiry has {option_slice.days_to_expiry} days to run",
        )

    calls = option_slice.by_type(OptionType.CALL)
    puts = option_slice.by_type(OptionType.PUT)
    paired = option_slice.paired_strikes()
    if not paired:
        raise MarketDataError(
            f"{option_slice.underlying.code} expiry {option_slice.expiry_tag} has no "
            f"strike quoted as both a call and a put; parity cannot be applied"
        )

    usable_strikes = [
        strike
        for strike in paired
        if not require_traded or (calls[strike].has_traded and puts[strike].has_traded)
    ]

    if len(usable_strikes) < min_strikes:
        return ForwardResult(
            forward=float("nan"),
            discount_factor=float(np.exp(-rate * year_fraction)),
            rate=rate,
            year_fraction=year_fraction,
            parity_residual_rms=float("inf"),
            parity_residual_max=float("inf"),
            n_strikes=len(usable_strikes),
            usable=False,
            reason=(f"only {len(usable_strikes)} paired strikes traded today, need {min_strikes}"),
        )

    strikes = np.array(usable_strikes, dtype=float)
    spreads = np.array([calls[k].price - puts[k].price for k in usable_strikes], dtype=float)
    discount = float(np.exp(-rate * year_fraction))

    # F = mean[(C - P)/D + K], the one-parameter least-squares solution.
    forward = float(np.mean(spreads / discount + strikes))

    fitted = (forward - strikes) * discount
    residuals = spreads - fitted
    rms = float(np.sqrt(np.mean(residuals**2)))
    worst = float(np.max(np.abs(residuals)))

    relative = rms / forward if forward > 0 else float("inf")
    usable = forward > 0.0 and relative <= max_residual_relative
    reason = (
        ""
        if usable
        else (
            f"parity residual {relative:.3%} of forward exceeds the "
            f"{max_residual_relative:.1%} limit; quotes are too far out of line "
            f"with each other to carry volatility information"
        )
        if forward > 0.0
        else "implied forward is not positive"
    )

    result = ForwardResult(
        forward=forward,
        discount_factor=discount,
        rate=rate,
        year_fraction=year_fraction,
        parity_residual_rms=rms,
        parity_residual_max=worst,
        n_strikes=len(usable_strikes),
        usable=usable,
        reason=reason,
    )
    get_logger(__name__).debug(
        "forward implied",
        extra={
            "underlying": option_slice.underlying.code,
            "expiry": option_slice.expiry_tag,
            "forward": forward,
            "residual_rel": relative,
            "usable": usable,
        },
    )
    return result


# ---------------------------------------------------------------------------
# Term structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ForwardCurve:
    """The forward term structure implied by a chain, as a drift curve.

    A volatility surface is parameterised in ``k = log(K / F)``, so anything
    that prices off it has to use *the same* forward curve. Otherwise the
    moneyness at which the local volatility is evaluated is simply wrong, and
    nothing downstream reports it.

    The gap is not academic. The implied carry measured on the committed
    chains is -4.16% a year on 510050 and -5.92% on 510300, while a typical
    ``r - q`` input of +2% would put the one-year forward out by 6.4% and 8.2%
    respectively. Against a snowball with barriers at 103% and 80% that error
    swamps every other modelling choice in the engine.

    Why a drift curve rather than a price curve
    -------------------------------------------
    What is stored is the *shape*: the instantaneous drift ``d log F / dt``,
    piecewise constant between quoted expiries. Levels come from whatever spot
    the caller is pricing at, via :meth:`forward_from`. That keeps the curve
    scale-free, so a surface fitted on an ETF trading at 2.85 can price a
    snowball quoted on an index at 5000 -- the moneyness coordinate only cares
    about ratios, and a structured note's barriers are percentages.

    Beyond the last quoted expiry
    -----------------------------
    The drift reverts to ``tail_drift``, which the caller supplies from its own
    ``r - q - basis``. This is a deliberate choice over extrapolating the
    implied carry: the listed options stop at roughly five months while the
    products run one to three years, and carrying a -5% carry out to three
    years would drop the forward 14% on the strength of five-month quotes.

    The price of that choice is a kink in the drift at the last quoted expiry.
    :attr:`kink` measures it, and it is reported rather than smoothed, because
    on real data it is large.

    Attributes:
        maturities: Quoted expiry year fractions, strictly increasing.
        forwards: Parity-implied forward at each quoted expiry.
        tail_drift: Drift used beyond the last quoted expiry.
        rate: The rate used for discounting, kept so :meth:`carry` can report
            the yield deducted from the drift.
    """

    maturities: tuple[float, ...]
    forwards: tuple[float, ...]
    tail_drift: float
    rate: float = DEFAULT_RATE

    def __post_init__(self) -> None:
        if len(self.maturities) != len(self.forwards):
            raise MarketDataError(
                f"got {len(self.maturities)} maturities but {len(self.forwards)} forwards"
            )
        if not self.maturities:
            raise MarketDataError("a forward curve needs at least one quoted expiry")
        T = np.asarray(self.maturities, dtype=float)
        F = np.asarray(self.forwards, dtype=float)
        if np.any(T <= 0.0):
            raise MarketDataError("quoted maturities must be positive")
        if len(T) > 1 and not np.all(np.diff(T) > 0):
            raise MarketDataError(f"maturities must be strictly increasing, got {T.tolist()}")
        if np.any(F <= 0.0):
            raise MarketDataError("implied forwards must be positive")

    # -- internals --------------------------------------------------------

    @property
    def _knots(self) -> tuple[np.ndarray, np.ndarray]:
        """Knot times and log-forwards, with ``t = 0`` prepended.

        The value at the origin is back-extrapolated at the first segment's own
        drift, so the curve is continuous there and ``log_forward_ratio(0)`` is
        exactly zero.
        """
        T = np.asarray(self.maturities, dtype=float)
        log_F = np.log(np.asarray(self.forwards, dtype=float))
        first_drift = self._segment_drifts[0]
        return (
            np.concatenate(([0.0], T)),
            np.concatenate(([log_F[0] - first_drift * T[0]], log_F)),
        )

    @property
    def segment_drifts(self) -> np.ndarray:
        """Drift on each quoted segment, for reporting the curve's shape."""
        return self._segment_drifts

    @property
    def _segment_drifts(self) -> np.ndarray:
        """Drift on each quoted segment; the tail drift when there is only one."""
        T = np.asarray(self.maturities, dtype=float)
        log_F = np.log(np.asarray(self.forwards, dtype=float))
        if T.size < 2:
            return np.array([self.tail_drift])
        return np.asarray(np.diff(log_F) / np.diff(T))

    # -- the curve --------------------------------------------------------

    def drift(self, t: float | np.ndarray) -> np.ndarray:
        """Instantaneous drift ``d log F / dt`` at time ``t``.

        Piecewise constant: the first segment's drift before the first quoted
        expiry, the enclosing segment's drift between them, and
        :attr:`tail_drift` beyond the last.
        """
        t_arr = np.asarray(t, dtype=float)
        T = np.asarray(self.maturities, dtype=float)
        segments = self._segment_drifts

        # searchsorted returns the KNOT index; the segment index is one lower,
        # because segment i spans (T_i, T_{i+1}]. Without the -1 the curve
        # returns the next segment's drift, which integrates to a log-forward
        # that disagrees with log_forward_ratio by a constant -- measured at
        # 1.62e-03 before this was fixed. The clip at 0 is what makes the drift
        # before the first quoted expiry equal the first segment's, as
        # documented, and side="left" makes the drift left-continuous at a
        # knot, matching a piecewise-constant drift defined on (T_i, T_{i+1}].
        index = np.clip(np.searchsorted(T, t_arr, side="left") - 1, 0, segments.size - 1)
        value = segments[index]
        return np.asarray(np.where(t_arr > T[-1], self.tail_drift, value))

    def log_forward_ratio(self, t: float | np.ndarray) -> np.ndarray:
        """``log(F(t) / F(0))`` -- the integrated drift.

        Exact rather than a quadrature: the drift is piecewise constant, so the
        integral is the piecewise-linear interpolant of the log-forwards.
        """
        t_arr = np.asarray(t, dtype=float)
        knot_t, knot_log_f = self._knots
        inside = np.interp(np.minimum(t_arr, knot_t[-1]), knot_t, knot_log_f)
        beyond = np.maximum(t_arr - knot_t[-1], 0.0) * self.tail_drift
        return np.asarray(inside + beyond - knot_log_f[0])

    def forward_from(self, spot: float, t: float | np.ndarray) -> np.ndarray:
        """Forward to ``t`` for a given spot, following this curve's shape."""
        return np.asarray(float(spot) * np.exp(self.log_forward_ratio(t)))

    def carry(self, t: float | np.ndarray) -> np.ndarray:
        """Yield deducted from the drift, ``rate - drift(t)``.

        Matches :attr:`~otcpricer.core.market.EquityMarket.carry`, so a
        boundary condition written in terms of carry can take either.
        """
        return np.asarray(self.rate - self.drift(t))

    @property
    def spot_anchor(self) -> float:
        """``F(0)``, back-extrapolated from the first quoted expiry.

        Price a European at this spot and the curve reproduces every quoted
        forward exactly, which is what makes the vanilla reprice test a test of
        the local volatility rather than of a spot mismatch.
        """
        return float(np.exp(self._knots[1][0]))

    @property
    def implied_carry(self) -> float:
        """Average implied carry across the quoted range, ``-d log F / dt``.

        The headline number for how far the market's forward curve sits from a
        naive ``r - q``.
        """
        T = np.asarray(self.maturities, dtype=float)
        if T.size < 2:
            return float(self.rate - self.tail_drift)
        log_F = np.log(np.asarray(self.forwards, dtype=float))
        return float(-(log_F[-1] - log_F[0]) / (T[-1] - T[0]))

    @property
    def kink(self) -> float:
        """Jump in drift at the last quoted expiry, ``tail_drift - last implied``.

        Zero only if the caller's ``r - q - basis`` happens to match what the
        options imply. On the committed chains it does not, by several
        percentage points, which is why it is reported.
        """
        return float(self.tail_drift - self._segment_drifts[-1])

    def as_dict(self) -> dict[str, object]:
        """Flat dict for API responses and reports."""
        segments = self._segment_drifts
        return {
            "maturities": [round(t, 6) for t in self.maturities],
            "forwards": [round(f, 6) for f in self.forwards],
            "segment_drifts": [round(float(d), 8) for d in segments],
            "implied_carry": round(self.implied_carry, 8),
            "tail_drift": round(self.tail_drift, 8),
            "kink": round(self.kink, 8),
            "spot_anchor": round(self.spot_anchor, 6),
            "rate": self.rate,
        }
