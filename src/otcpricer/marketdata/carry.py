"""Carry measured from listed index futures, and the reach of that measurement.

Why this is carry and not basis
-------------------------------
A snowball's coupon is paid for by volatility, the futures discount and cash
management, and the market view is that the deeper the discount the higher the
coupon. Until now ``basis_pa`` was a user input defaulting to zero, so the
second of those three was simply absent from every price this project produced.

The futures panel can supply it -- but what the board shows is the **drift**,
and three steps separate that from a basis. A future's settlement is the
forward, so

.. code-block:: text

    log(F / S) / tau  =  drift  =  r - q - basis

is one observable against three unknowns. Nothing in the futures board
separates the dividend yield from the financing spread from the discount: the
same forward is consistent with ``q = 3%, basis = 0`` and with ``q = 0,
basis = 3%``. So this module measures the drift, which is observed, and reports
``carry = r - drift = q + basis``, which is the single quantity both engines
actually read -- ``dV/dq`` and ``dV/dbasis_pa`` are the same derivative to
1.5e-15. It is fed in as ``q=0, basis_pa=carry``. Splitting it further would be
inventing the split.

The rate is a label, not an input to the price
----------------------------------------------
``carry = r - drift`` needs a rate, and I got this wrong on the first pass:
I measured ``-log(F/S)/tau`` and called it the carry, which is the carry at a
zero rate. The offset is the whole rate -- 2% a year, against measured carries
of 2% to 10%.

What saves it from mattering is that the rate cancels. Feed ``q=0`` and
``basis_pa = r - drift`` into an :class:`~otcpricer.core.market.EquityMarket`
carrying that same ``r`` and its drift comes back as ``r - (r - drift)``, the
measured drift exactly, whatever ``r`` was. So under
:attr:`TailCarry.HOLD_LAST` every forward this module produces is invariant to
the rate, and there is a test for that. The rate changes what the number is
*called*, not what it prices.

The exception is :attr:`TailCarry.ZERO`, which is a statement about carry
rather than about the board, and so is the one tail rule whose forwards do move
with the rate. That is not a flaw in it -- "no carry" has no meaning without a
rate to be no carry against.

The front contract cannot be annualised
---------------------------------------
Annualising a small basis over a small tenor divides noise by a small number.
Measured across all 725 panel days, the front contract's implied drift by its
own time to expiry (the rate offset is constant, so it moves the mean and
leaves every spread below unchanged):

=====================  ======  ===========  ==================
front contract tenor   n       sd           range
=====================  ======  ===========  ==================
tau <= 0.03y           264     **0.473**    -5.63 to **+1.76**
0.03 < tau <= 0.06y    222     0.109        -0.97 to +0.52
0.06 < tau <= 0.10y    198     0.060        -0.18 to +0.11
=====================  ======  ===========  ==================

A 563% annualised drift is one index point of basis divided by seven days. The
spot-to-farthest-contract drift over the same panel has an sd of **0.026** --
eighteen times tighter. So contracts inside :data:`MIN_FUTURES_TENOR` are
rejected and counted, rather than left to dominate a curve by being first.

The carry term structure is not flat
------------------------------------
``|carry(far) - carry(near)|`` exceeds 2% a year on **59.8%** of days for IF,
**54.9%** for IH and **70.1%** for IM. A single scalar is therefore a real
approximation rather than a formality, which is why the shape is kept as a
:class:`~otcpricer.marketdata.forward.ForwardCurve` and why
:meth:`CarryMeasurement.average_carry_to` integrates it over the note's own
life instead of quoting a level.

That class is reused rather than reimplemented. It already holds a
piecewise-constant drift with exact integration, an explicit tail and a
reported kink -- which is precisely this problem, and it was validated against
the option-implied forwards in Phase 2.

Carry is negative often enough that rejecting it is wrong
---------------------------------------------------------
Spot-to-farthest-contract carry over the panel, at the panel's own measured
zero rate (mean 1.57%):

======  =======  ======  ======================  =============
root    mean     sd      range                   negative
======  =======  ======  ======================  =============
IF      +3.70%   2.48%   -4.68% to +12.73%       8.3% of days
IH      +2.34%   2.10%   -6.91% to +12.15%       **10.3%**
IM      +10.35%  3.70%   +0.19% to +19.08%       0.0%
======  =======  ======  ======================  =============

``core/market.py`` rejected ``basis_pa < 0`` outright, which refuses roughly
**one day in ten** of measured IF and IH data. Premium is a real market state
-- the SSE 50's dividend structure differs from the CSI 1000's, and its futures
trade above the index on 40.9% of panel days -- and clamping it to zero would
be editing the data to fit the assumption rather than the reverse.

The CSI 1000 is the exception, and it is the interesting one: its carry is
positive on every single panel day, averaging 10.35%. That is why snowballs are
written on it.

Half the note's life is not measured at all
-------------------------------------------
The listed contracts reach **0.4984 years on average** (min 0.019, max 0.663).
Against the products this project prices:

=============  ===================
snowball tenor  fraction measured
=============  ===================
1 year          **48.5%**
2 years         24.2%
3 years         16.2%
=============  ===================

So for a three-year note, **84% of the carry is an assumption**, and it is the
single largest one in the quote. :class:`TailCarry` is therefore a required
choice with no default, for the same reason ``SharkFinTerms.monitoring`` is:
picking one silently is how a model comes to hold a number nobody chose.
:meth:`CarryMeasurement.measured_fraction` is what a report leads with.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Protocol

import numpy as np

from otcpricer.core.exceptions import MarketDataError
from otcpricer.core.logging import get_logger
from otcpricer.marketdata.forward import DEFAULT_RATE, ForwardCurve

__all__ = [
    "MIN_CONTRACTS",
    "MIN_FUTURES_TENOR",
    "CarryMeasurement",
    "TailCarry",
    "measure_carry",
]

_log = get_logger(__name__)

#: Shortest futures tenor whose basis is worth annualising.
#:
#: Set from the measurement in the module docstring: below 0.03 years the
#: implied carry has a standard deviation of 0.47 and reaches +5.63, which is
#: one index point of basis divided by a week. At 0.05 the amplification is
#: bounded by 20x rather than by 50x, and no panel day loses its curve -- four
#: contracts list at a time and the rejected one is replaced by the next.
MIN_FUTURES_TENOR = 0.05

#: Fewest usable contracts for a measurement.
#:
#: Two, because one contract gives a level but no shape, and the shape is what
#: distinguishes this from the scalar input it replaces. A single contract is
#: reported as unusable rather than extrapolated flat.
MIN_CONTRACTS = 2


class TailCarry(Enum):
    """What carry applies beyond the farthest listed future.

    A required choice, not a default. The listed contracts reach about half a
    year while the notes run one to three, so this governs 48% to 84% of the
    carry in a quote -- more than any other single input, including the
    volatility, on a three-year note.

    Attributes:
        ZERO: No carry past the board, so the forward grows at ``r`` -- which
            is exactly the project's old default of ``basis_pa = 0``. It
            understates the coupon on a persistently discounted index like the
            CSI 1000, and it is the one rule whose forwards depend on the rate,
            because "no carry" is a claim about carry rather than about the
            board.
        HOLD_LAST: Hold the farthest measured carry flat. The natural reading
            of the data, and the one a desk quoting off the screen uses -- but
            it extrapolates a six-month observation out to three years, which
            on IM means asserting a 10.4% carry for two and a half years nobody
            has quoted.
        EXPLICIT: The caller supplies the number, and owns it.
    """

    ZERO = "zero"
    HOLD_LAST = "hold_last"
    EXPLICIT = "explicit"

    def __str__(self) -> str:
        return self.value


class _FuturesPanel(Protocol):
    """The two panel methods a measurement needs.

    Narrowed deliberately: a measurement should be testable from a handful of
    tuples, not only from the 1.3-million-row institutional panel, which is not
    committed to this repository.
    """

    def futures(self, root: str, as_of: date) -> tuple[tuple[date, float], ...]: ...

    def index_close(self, index_code: str, as_of: date) -> float: ...


@dataclass(frozen=True, slots=True)
class CarryMeasurement:
    """Carry read off one root's listed futures, with the reach of the reading.

    Attributes:
        as_of: The measurement date.
        root: Futures root, e.g. ``IF``.
        index_code: The settling index, whose close anchors the carry.
        index_level: That close. The carry of the nearest usable contract is
            measured against it, so the curve starts where the market is
            rather than where a back-extrapolation puts it.
        curve: The forward shape, as a
            :class:`~otcpricer.marketdata.forward.ForwardCurve`.
        tail: Which tail rule was chosen.
        tail_carry: The carry that rule produced beyond the board.
        maturities: Usable contract tenors in years, ascending.
        forwards: Their settlements.
        rejected_near: Contracts dropped for being inside
            :data:`MIN_FUTURES_TENOR`, whose annualised carry is noise.
        rate: The rate the curve's drift is quoted against.
    """

    as_of: date
    root: str
    index_code: str
    index_level: float
    curve: ForwardCurve
    tail: TailCarry
    tail_carry: float
    maturities: tuple[float, ...]
    forwards: tuple[float, ...]
    rejected_near: int
    rate: float = DEFAULT_RATE

    # -- the measured carries ---------------------------------------------

    @property
    def contract_drifts(self) -> tuple[float, ...]:
        """``log(F / S) / tau`` for each usable contract -- what is observed.

        Each measured against the index close rather than against its
        neighbour, so these are the drifts that actually apply to a note
        starting today, and they involve no rate at all.
        """
        S = self.index_level
        return tuple(
            float(np.log(F / S)) / t for t, F in zip(self.maturities, self.forwards, strict=True)
        )

    @property
    def contract_carries(self) -> tuple[float, ...]:
        """``r - drift`` for each usable contract -- what the engines read.

        This is the quantity to pass as ``basis_pa`` with ``q=0``. The rate
        used is :attr:`rate`, and it cancels downstream: see the module
        docstring on why that makes it a label rather than an input.
        """
        return tuple(self.rate - d for d in self.contract_drifts)

    @property
    def near_carry(self) -> float:
        """Carry to the nearest usable contract."""
        return self.contract_carries[0]

    @property
    def far_carry(self) -> float:
        """Carry to the farthest listed contract.

        The stable aggregate: an sd of 0.026 across the panel against 0.473 for
        the raw front contract.
        """
        return self.contract_carries[-1]

    @property
    def far_drift(self) -> float:
        """``log(F_far / S) / tau_far`` -- the same reading without the rate."""
        return self.contract_drifts[-1]

    @property
    def term_structure_gap(self) -> float:
        """``far_carry - near_carry``.

        Exceeds 2% a year on most panel days, which is why the shape is kept
        rather than collapsed to a level.
        """
        return self.far_carry - self.near_carry

    @property
    def observed_reach(self) -> float:
        """Farthest tenor with a listed contract, in years."""
        return self.maturities[-1]

    def measured_fraction(self, T: float) -> float:
        """Fraction of a note's life the futures board actually covers.

        0.485 for a one-year note and 0.162 for a three-year one on the mean
        panel day. The complement is :class:`TailCarry`'s assumption, and it is
        the largest one in the quote.
        """
        if T <= 0.0:
            raise MarketDataError(f"a tenor must be positive, got {T}")
        return min(self.observed_reach, T) / T

    # -- the curve, anchored at the index -----------------------------------

    def log_forward_ratio(self, T: float) -> float:
        """``log(F(T) / S)``, anchored at the index close.

        Below the nearest usable contract the carry is held flat at
        :attr:`near_carry` -- there is nothing to interpolate against, and the
        rejected front contracts are rejected precisely because their own
        annualised carry is noise.

        Above it the shape comes from :attr:`curve`, whose increments are
        anchor-free: ``log_forward_ratio(T) - log_forward_ratio(T1)`` is
        ``log(F(T)/F(T1))`` whatever spot the curve back-extrapolated to. That
        is what lets the measured near carry and the curve's shape be combined
        without either being distorted.
        """
        first_t = self.maturities[0]
        to_first = float(np.log(self.forwards[0] / self.index_level))
        if first_t >= T:
            return to_first * (T / first_t)
        beyond = float(self.curve.log_forward_ratio(T)) - float(
            self.curve.log_forward_ratio(first_t)
        )
        return to_first + beyond

    def average_drift_to(self, T: float) -> float:
        """``log(F(T)/S) / T`` -- the flat drift equivalent over ``[0, T]``."""
        if T <= 0.0:
            raise MarketDataError(f"a tenor must be positive, got {T}")
        return self.log_forward_ratio(T) / T

    def average_carry_to(self, T: float) -> float:
        """The flat carry equivalent to this curve over ``[0, T]``.

        ``r - log(F(T)/S) / T``. This is the number that goes into
        :attr:`~otcpricer.core.market.EquityMarket.basis_pa` alongside ``q=0``,
        and it is an average rather than a level precisely because the term
        structure is not flat -- a note straddling the board's edge is part
        measurement and part tail assumption, and this weights them by how long
        each applies.
        """
        return self.rate - self.average_drift_to(T)

    def forward_to(self, T: float) -> float:
        """The forward to ``T`` from the index close."""
        return self.index_level * float(np.exp(self.log_forward_ratio(T)))

    @property
    def kink(self) -> float:
        """Jump in drift where the board ends, from the curve itself.

        Zero only if the chosen tail happens to match the last listed segment.
        Under :attr:`TailCarry.HOLD_LAST` it is small by construction but not
        zero, because holding the *spot-to-far* carry flat is not the same as
        holding the last *forward-forward* drift flat.
        """
        return self.curve.kink

    @property
    def anchor_gap(self) -> float:
        """How far the curve's back-extrapolated spot sits from the index.

        :class:`~otcpricer.marketdata.forward.ForwardCurve` infers ``F(0)`` from
        the first listed *segment's* drift, which is a forward-forward carry.
        The index close is a different number, and the difference measures how
        badly a forward-forward carry misrepresents a spot-to-futures one. It
        is reported rather than reconciled, and it is the reason
        :meth:`log_forward_ratio` anchors at the index instead of trusting
        :attr:`~otcpricer.marketdata.forward.ForwardCurve.spot_anchor`.
        """
        return self.curve.spot_anchor / self.index_level - 1.0

    def as_dict(self) -> dict[str, object]:
        return {
            "as_of": self.as_of.isoformat(),
            "root": self.root,
            "index_code": self.index_code,
            "index_level": round(self.index_level, 4),
            "maturities": [round(t, 6) for t in self.maturities],
            "forwards": [round(f, 4) for f in self.forwards],
            "contract_drifts": [round(d, 8) for d in self.contract_drifts],
            "contract_carries": [round(c, 8) for c in self.contract_carries],
            "near_carry": round(self.near_carry, 8),
            "far_carry": round(self.far_carry, 8),
            "rate": round(self.rate, 8),
            "term_structure_gap": round(self.term_structure_gap, 8),
            "observed_reach": round(self.observed_reach, 6),
            "tail": str(self.tail),
            "tail_carry": round(self.tail_carry, 8),
            "kink": round(self.kink, 8),
            "anchor_gap": round(self.anchor_gap, 8),
            "rejected_near": self.rejected_near,
            "measured_fraction_1y": round(self.measured_fraction(1.0), 4),
            "measured_fraction_3y": round(self.measured_fraction(3.0), 4),
            "note": (
                "The board observes a drift; carry is r - drift, and is total "
                "carry (q + basis) rather than a basis. A futures settlement "
                "is one observable against three unknowns, so nothing here "
                "separates the dividend yield from the financing spread. Feed "
                "it as q=0, basis_pa=carry with this same rate, and the rate "
                "cancels: the forward is reproduced exactly."
            ),
        }


def measure_carry(
    panel: _FuturesPanel,
    root: str,
    index_code: str,
    as_of: date,
    *,
    tail: TailCarry,
    tail_carry: float | None = None,
    rate: float = DEFAULT_RATE,
    min_tenor: float = MIN_FUTURES_TENOR,
) -> CarryMeasurement:
    """Measure carry from one root's listed futures on one date.

    Args:
        panel: Anything exposing ``futures`` and ``index_close``.
        root: Futures root, e.g. ``IF``.
        index_code: The settling index's Wind code, e.g. ``000300.SH``.
        as_of: The date to measure.
        tail: Required. What carry applies beyond the board -- which is 48% to
            84% of a note's life, so there is deliberately no default.
        tail_carry: The number, required for and only accepted with
            :attr:`TailCarry.EXPLICIT`.
        rate: Rate carry is quoted against. The board observes a drift; this
            is what turns it into a carry. It cancels downstream, so under
            :attr:`TailCarry.HOLD_LAST` the forwards are invariant to it --
            but pass the rate the pricing will use, so the reported number
            means what it says.
        min_tenor: Shortest contract worth annualising.

    Returns:
        A :class:`CarryMeasurement`.

    Raises:
        MarketDataError: if the board has fewer than :data:`MIN_CONTRACTS`
            usable contracts, or if ``tail_carry`` and ``tail`` disagree.
    """
    if tail is TailCarry.EXPLICIT and tail_carry is None:
        raise MarketDataError(
            "TailCarry.EXPLICIT means the caller supplies the number; pass tail_carry"
        )
    if tail is not TailCarry.EXPLICIT and tail_carry is not None:
        raise MarketDataError(
            f"tail_carry={tail_carry} was given alongside tail={tail}, which computes "
            f"its own. Two sources for one number is how a model ends up quoting a "
            f"carry nobody chose -- pass TailCarry.EXPLICIT to supply it."
        )

    listed = panel.futures(root, as_of)
    if not listed:
        raise MarketDataError(
            f"no {root} futures settled on {as_of.isoformat()}; the panel covers "
            f"2023-06-30 to 2026-06-29 and the last day has no futures quotes"
        )
    index_level = panel.index_close(index_code, as_of)
    if index_level <= 0.0:
        raise MarketDataError(f"{index_code} closed at {index_level} on {as_of.isoformat()}")

    usable: list[tuple[float, float]] = []
    rejected = 0
    for expiry, settle in listed:
        tenor = (expiry - as_of).days / 365.0
        if tenor < min_tenor:
            rejected += 1
            continue
        usable.append((tenor, settle))

    if len(usable) < MIN_CONTRACTS:
        raise MarketDataError(
            f"{root} on {as_of.isoformat()} has {len(usable)} contract(s) beyond "
            f"{min_tenor}y ({rejected} rejected as too near), need {MIN_CONTRACTS}. "
            f"One contract gives a level but no shape, and the shape is the whole "
            f"reason this replaces a scalar input."
        )

    maturities = tuple(t for t, _ in usable)
    forwards = tuple(f for _, f in usable)

    far_carry = rate - float(np.log(forwards[-1] / index_level)) / maturities[-1]
    if tail is TailCarry.ZERO:
        resolved_tail = 0.0
    elif tail is TailCarry.HOLD_LAST:
        resolved_tail = far_carry
    else:
        assert tail_carry is not None  # guarded above
        resolved_tail = float(tail_carry)

    curve = ForwardCurve(
        maturities=maturities,
        forwards=forwards,
        tail_drift=rate - resolved_tail,
        rate=rate,
    )

    measurement = CarryMeasurement(
        as_of=as_of,
        root=root,
        index_code=index_code,
        index_level=index_level,
        curve=curve,
        tail=tail,
        tail_carry=resolved_tail,
        maturities=maturities,
        forwards=forwards,
        rejected_near=rejected,
        rate=rate,
    )
    _log.info(
        "carry measured for %s on %s: near %+.4f, far %+.4f, reach %.4fy, "
        "tail %s %+.4f, %d near contracts rejected",
        root,
        as_of.isoformat(),
        measurement.near_carry,
        measurement.far_carry,
        measurement.observed_reach,
        tail,
        resolved_tail,
        rejected,
    )
    return measurement
