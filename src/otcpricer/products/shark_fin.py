"""Shark fin notes: a barrier option, stated precisely enough to price twice.

Why this exists at all
----------------------
The shark fin was the last product in this project still being priced by two
unconnected pieces of code that disagreed about what it *was*:

- ``/api/shark_fin`` evaluated the Rubinstein-Reiner closed form inline in the
  route, touching no engine, and so modelling a **continuously** monitored
  barrier.
- ``cn_fdm.py`` solved a finite-difference grid truncated at the barrier, and
  so modelled something closer to a **discretely** monitored one, at whatever
  frequency the grid happened to imply.

Neither was wrong in isolation. The problem is that they are prices of
*different contracts* -- a discretely monitored knock-out is strictly more
valuable than a continuously monitored one, because the underlying can breach
and recover between observations -- and nothing in either path said so. A
continuous barrier is also a modelling idealisation no actual term sheet
contains: real notes observe at a close.

So monitoring is a **required, explicit** field here, with no default. Making it
optional would mean choosing one of the two contracts silently, which is how the
project arrived at two disagreeing prices in the first place.

What the contract is
--------------------
A European option whose payoff is extinguished (knocked out) or created
(knocked in) if the underlying crosses a barrier on an observation date, with an
optional cash rebate when it does.

:class:`~otcpricer.core.types.BarrierKind` carries the four-way taxonomy and had
been declared in ``core/types.py`` since Phase 1 without a single caller. This
is it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from otcpricer.core.exceptions import ParameterError
from otcpricer.core.types import BarrierKind, OptionType

__all__ = [
    "CONTINUOUS",
    "SharkFinTerms",
]

#: Observations per year standing for a continuously monitored barrier.
#:
#: Not a sentinel that switches code paths -- it is the *limit* the discrete
#: engines are checked against, and the convergence study in the tests takes
#: the observation frequency towards it. An analytic formula for a continuous
#: barrier exists; nothing else here pretends continuity is achievable on a
#: grid.
CONTINUOUS = "continuous"


@dataclass(frozen=True, slots=True)
class SharkFinTerms:
    """A single-barrier European note.

    Attributes:
        S0: Spot at inception, which the barrier and strike are quoted against.
        K: Strike.
        T: Tenor in years.
        barrier: Absolute barrier level.
        kind: Which of the four barrier types this is.
        option: Call or put payoff, before the barrier is applied.
        rebate: Cash paid if the barrier is touched, as a fraction of notional.
            Paid at the touch for a knock-out; a knock-in pays nothing here.
        monitoring: Observations per year, or :data:`CONTINUOUS`. **Required**:
            see the module docstring for why there is no default.
        participation: Multiplier on the option payoff.
    """

    S0: float
    K: float
    T: float
    barrier: float
    kind: BarrierKind
    monitoring: int | str
    option: OptionType = OptionType.CALL
    rebate: float = 0.0
    participation: float = 1.0

    def __post_init__(self) -> None:
        if self.S0 <= 0.0:
            raise ParameterError(f"S0 must be positive, got {self.S0}")
        if self.K <= 0.0:
            raise ParameterError(f"K must be positive, got {self.K}")
        if self.T <= 0.0:
            raise ParameterError(f"T must be positive, got {self.T}")
        if self.barrier <= 0.0:
            raise ParameterError(f"barrier must be positive, got {self.barrier}")
        if self.rebate < 0.0:
            raise ParameterError(f"rebate cannot be negative, got {self.rebate}")
        if self.participation <= 0.0:
            raise ParameterError(f"participation must be positive, got {self.participation}")

        if self.monitoring != CONTINUOUS:
            if not isinstance(self.monitoring, int) or isinstance(self.monitoring, bool):
                raise ParameterError(
                    f"monitoring must be an observation count or {CONTINUOUS!r}, "
                    f"got {self.monitoring!r}"
                )
            if self.monitoring < 1:
                raise ParameterError(
                    f"monitoring must be at least 1 observation a year, got {self.monitoring}"
                )

        # The barrier must start on the live side of the trade. A knock-out
        # already breached at inception is worth its rebate and nothing else,
        # and a caller who wrote that almost certainly meant something else --
        # so it is refused rather than priced.
        if self.kind.is_up and self.barrier <= self.S0:
            raise ParameterError(
                f"an up barrier at {self.barrier} is already at or below spot "
                f"{self.S0}, so the note is dead at inception"
            )
        if not self.kind.is_up and self.barrier >= self.S0:
            raise ParameterError(
                f"a down barrier at {self.barrier} is already at or above spot "
                f"{self.S0}, so the note is dead at inception"
            )

    # -- Observation schedule ----------------------------------------------

    @property
    def is_continuous(self) -> bool:
        return self.monitoring == CONTINUOUS

    @property
    def n_observations(self) -> int:
        """Barrier observations over the note's life, excluding inception.

        Zero for a continuous barrier, where there is no schedule to build.
        """
        if self.is_continuous:
            return 0
        return max(int(round(self.T * int(self.monitoring))), 1)

    def observation_times(self) -> np.ndarray:
        """Observation dates in years, ascending, ending at maturity.

        Empty for a continuous barrier. Ending exactly at ``T`` matters: a
        schedule that stopped short would leave the final period unmonitored
        and quietly raise the price.
        """
        if self.is_continuous:
            return np.empty(0, dtype=float)
        count = self.n_observations
        return np.linspace(self.T / count, self.T, count)

    # -- Payoff ------------------------------------------------------------

    @property
    def is_knock_out(self) -> bool:
        return self.kind.is_knock_out

    @property
    def barrier_moneyness(self) -> float:
        """Barrier as a fraction of inception spot, which is how it is quoted."""
        return self.barrier / self.S0

    @property
    def strike_moneyness(self) -> float:
        return self.K / self.S0

    def terminal_payoff(self, spot: np.ndarray) -> np.ndarray:
        """The vanilla payoff, before any barrier is applied."""
        values = np.asarray(spot, dtype=float)
        if self.option is OptionType.CALL:
            intrinsic = np.maximum(values - self.K, 0.0)
        else:
            intrinsic = np.maximum(self.K - values, 0.0)
        return self.participation * intrinsic

    def breached(self, spot: np.ndarray) -> np.ndarray:
        """Whether each level is on the far side of the barrier.

        Inclusive of the barrier itself, which is the market convention for a
        touch and also the conservative reading for a knock-out.
        """
        values = np.asarray(spot, dtype=float)
        if self.kind.is_up:
            return values >= self.barrier
        return values <= self.barrier

    def as_dict(self) -> dict[str, object]:
        return {
            "S0": self.S0,
            "K": self.K,
            "T": self.T,
            "barrier": self.barrier,
            "kind": str(self.kind),
            "option": str(self.option),
            "rebate": self.rebate,
            "participation": self.participation,
            "monitoring": self.monitoring,
            "is_continuous": self.is_continuous,
            "n_observations": self.n_observations,
            "barrier_moneyness": round(self.barrier_moneyness, 6),
            "strike_moneyness": round(self.strike_moneyness, 6),
            "note": (
                "Monitoring frequency is part of the contract, not a numerical "
                "setting. A discretely monitored knock-out is worth strictly "
                "more than a continuously monitored one, because the "
                "underlying can breach and recover between observations."
            ),
        }
