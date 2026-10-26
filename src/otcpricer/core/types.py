"""Core enumerations and value types.

The legacy code passes option type around as the bare strings ``'c'`` and
``'p'`` (a py_vollib convention). :class:`OptionType` accepts those spellings
via :meth:`OptionType.parse` so old call sites and stored JSON keep working,
while new code gets a real type.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

__all__ = [
    "OptionType",
    "ExerciseStyle",
    "BarrierKind",
    "ProductType",
    "KIStrikeMode",
    "Moneyness",
    "Greeks",
]


class OptionType(StrEnum):
    """Call or put."""

    CALL = "c"
    PUT = "p"

    @classmethod
    def parse(cls, value: OptionType | str) -> OptionType:
        """Accept ``'c'``/``'p'``, ``'call'``/``'put'``, or an existing member.

        Raises:
            ParameterError: if the value names neither a call nor a put.
        """
        if isinstance(value, cls):
            return value
        key = str(value).strip().lower()
        if key in ("c", "call", "看涨"):
            return cls.CALL
        if key in ("p", "put", "看跌"):
            return cls.PUT
        from otcpricer.core.exceptions import ParameterError

        raise ParameterError(
            f"option type {value!r} is not recognised; expected 'c'/'call' or 'p'/'put'"
        )

    @property
    def sign(self) -> int:
        """``+1`` for a call, ``-1`` for a put.

        Lets the payoff and the d1/d2 terms be written once instead of
        branching: ``max(sign * (S - K), 0)``.
        """
        return 1 if self is OptionType.CALL else -1

    @property
    def label_cn(self) -> str:
        return "看涨期权" if self is OptionType.CALL else "看跌期权"


class ExerciseStyle(StrEnum):
    """When the holder may exercise."""

    EUROPEAN = "european"
    AMERICAN = "american"
    BERMUDAN = "bermudan"


class BarrierKind(StrEnum):
    """Barrier direction and knock sense, in the standard four-way taxonomy."""

    UP_AND_OUT = "uo"
    DOWN_AND_OUT = "do"
    UP_AND_IN = "ui"
    DOWN_AND_IN = "di"

    @property
    def is_up(self) -> bool:
        return self in (BarrierKind.UP_AND_OUT, BarrierKind.UP_AND_IN)

    @property
    def is_knock_out(self) -> bool:
        return self in (BarrierKind.UP_AND_OUT, BarrierKind.DOWN_AND_OUT)


class ProductType(StrEnum):
    """Tradable product families this library prices."""

    VANILLA = "vanilla"
    SNOWBALL = "snowball"
    SHARK_FIN = "shark_fin"
    FORWARD = "forward"
    IRS = "irs"


class KIStrikeMode(StrEnum):
    """Strike of a snowball's knock-in (short put) leg.

    The A-share market convention is ``S0``: once knocked in and not knocked
    out, the holder absorbs the full fall from the *initial* price, i.e. a
    short put struck at S0. Some products instead strike that put at the
    knock-in barrier, which caps the loss -- hence the explicit choice.

    Note:
        The legacy MC implementation hardcoded ``KI`` while the legacy PDE
        hardcoded ``S0``, so the two engines were valuing different products.
        Making this an explicit, shared parameter is what lets the PDE/MC
        cross-validation become meaningful.
    """

    S0 = "s0"
    KI = "ki"


Moneyness = Literal["itm", "atm", "otm"]


@dataclass(frozen=True, slots=True)
class Greeks:
    """First- and second-order risk of a position.

    Units follow desk convention rather than raw calculus, because that is how
    they are read off a risk blotter:

    * ``delta`` -- dV/dS, per 1 unit of underlying
    * ``gamma`` -- d2V/dS2, per 1 unit of underlying squared
    * ``vega``  -- dV/dsigma per **1 percentage point** of vol (not per 1.0)
    * ``theta`` -- dV/dt per **calendar day** (not per year)
    * ``rho``   -- dV/dr per **1 percentage point** of rate (not per 1.0)
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def scaled(self, qty: float) -> Greeks:
        """Return this risk multiplied by a position size."""
        return Greeks(
            delta=self.delta * qty,
            gamma=self.gamma * qty,
            vega=self.vega * qty,
            theta=self.theta * qty,
            rho=self.rho * qty,
        )

    def __add__(self, other: Greeks) -> Greeks:
        """Aggregate risk across positions."""
        if not isinstance(other, Greeks):
            return NotImplemented
        return Greeks(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            vega=self.vega + other.vega,
            theta=self.theta + other.theta,
            rho=self.rho + other.rho,
        )

    def as_dict(self, ndigits: int | None = 6) -> dict[str, float]:
        """Flat dict for JSON responses; ``ndigits=None`` keeps full precision."""
        raw = {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
        }
        if ndigits is None:
            return raw
        return {k: round(v, ndigits) for k, v in raw.items()}
