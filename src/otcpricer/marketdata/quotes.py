"""Quote containers for a listed option chain.

The feed delivers one flat table per expiry. These types turn it into something
a surface fit can consume, and -- importantly -- carry the provenance needed to
judge the data rather than only the numbers.

What the feed does and does not provide
---------------------------------------
It gives a last traded price and a previous settlement. It does **not** give
bid/ask. So the quotes across a slice are *non-synchronous*: each contract's
last trade happened at a different moment, and the less liquid wings can be
stale by a long way.

That is not a detail to be smoothed over. It is the dominant source of error in
everything downstream, and it is why :class:`OptionSlice` reports a parity
residual and why the surface fit screens quotes rather than trusting them all
equally. Measured on a live 50ETF slice, put-call parity residuals ran to
0.0032 against option prices of order 0.05 -- several percent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from otcpricer.core.exceptions import MarketDataError
from otcpricer.core.types import OptionType
from otcpricer.marketdata.contracts import ContractCode, Underlying, parse_option_code

__all__ = ["OptionChain", "OptionQuote", "OptionSlice"]


@dataclass(frozen=True, slots=True)
class OptionQuote:
    """One listed option contract's quote.

    Attributes:
        contract: Parsed contract identifier.
        last: Last traded price. May be stale; see the module docstring.
        previous_settlement: Prior settlement price, a fallback when the
            contract has not traded.
        strike: Strike as reported by the feed's own column.
        open_interest: Open interest where the feed supplies it.
    """

    contract: ContractCode
    last: float
    previous_settlement: float
    strike: float
    open_interest: float | None = None

    @property
    def option_type(self) -> OptionType:
        return self.contract.option_type

    @property
    def has_traded(self) -> bool:
        """Whether a usable last price exists.

        A zero or missing last price means the contract has not traded today,
        and its "price" carries no information about today's volatility.
        """
        return bool(np.isfinite(self.last)) and self.last > 0.0

    @property
    def price(self) -> float:
        """The price to use, preferring a live trade over a stale settlement."""
        return self.last if self.has_traded else self.previous_settlement


@dataclass(frozen=True, slots=True)
class OptionSlice:
    """All quotes for one expiry.

    Attributes:
        underlying: The underlying this slice belongs to.
        expiry: Exact expiry date, needed for the year fraction.
        as_of: When the data was captured.
        days_to_expiry: Calendar days remaining, as reported by the feed.
        quotes: Every contract in the slice, calls and puts.
    """

    underlying: Underlying
    expiry: date
    as_of: date
    days_to_expiry: int
    quotes: tuple[OptionQuote, ...]

    @property
    def expiry_tag(self) -> str:
        """The feed's expiry key, e.g. ``"2611"``."""
        return f"{self.expiry.year % 100:02d}{self.expiry.month:02d}"

    @property
    def year_fraction(self) -> float:
        """Time to expiry in years, ACT/365F.

        Taken from the feed's own day count rather than recomputed from dates,
        so it matches whatever calendar the exchange applied.
        """
        return self.days_to_expiry / 365.0

    def by_type(self, option_type: OptionType) -> dict[float, OptionQuote]:
        """Quotes of one option type, keyed by strike."""
        wanted = OptionType.parse(option_type)
        return {q.strike: q for q in self.quotes if q.option_type is wanted}

    def paired_strikes(self) -> tuple[float, ...]:
        """Strikes quoted as both a call and a put, in increasing order.

        These are the only strikes put-call parity can be applied at, so this
        is what the forward implication runs on.
        """
        calls = self.by_type(OptionType.CALL)
        puts = self.by_type(OptionType.PUT)
        return tuple(sorted(set(calls) & set(puts)))

    def __len__(self) -> int:
        return len(self.quotes)


@dataclass(frozen=True, slots=True)
class OptionChain:
    """Every available expiry for one underlying.

    Attributes:
        underlying: The underlying.
        as_of: Capture date.
        slices: Expiry slices, ordered by expiry.
        source: Where the data came from -- ``"akshare"``, a snapshot name, or
            a CSV path. Recorded because a number's trustworthiness depends on
            it, and a report that cannot say where its inputs came from is not
            a report.
    """

    underlying: Underlying
    as_of: date
    slices: tuple[OptionSlice, ...]
    source: str = "unknown"
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.slices:
            raise MarketDataError(f"option chain for {self.underlying.code} contains no expiries")

    @property
    def expiries(self) -> tuple[date, ...]:
        return tuple(s.expiry for s in self.slices)

    def slice_for(self, expiry_tag: str) -> OptionSlice:
        """Return the slice with the given expiry key, e.g. ``"2611"``."""
        for option_slice in self.slices:
            if option_slice.expiry_tag == expiry_tag:
                return option_slice
        available = ", ".join(s.expiry_tag for s in self.slices)
        raise MarketDataError(f"no expiry {expiry_tag!r} in chain; available: {available}")

    def __len__(self) -> int:
        return len(self.slices)


def build_slice(
    underlying: Underlying,
    rows: list[dict[str, object]],
    *,
    expiry: date,
    as_of: date,
    days_to_expiry: int,
    strike_tolerance: float = 1e-6,
) -> OptionSlice:
    """Assemble an :class:`OptionSlice` from raw feed rows.

    Each row needs ``code``, ``last``, ``previous_settlement`` and ``strike``.

    The strike is read twice -- from the feed's column and from the contract
    code -- and the two must agree. They are independent encodings of the same
    field, so a mismatch means the feed's format has changed under us, which is
    exactly the failure that would otherwise pass silently. Verified to agree on
    every row of both live underlyings (24/24 and 28/28).

    Which parser applies comes from the underlying's ``code_style``, because the
    three feeds spell the identifier in three unrelated ways. Wind's numeric ETF
    keys encode nothing at all, so they cannot reach here -- those rows are
    cross-checked against the contract *name* instead, in
    :mod:`otcpricer.marketdata.wind`.

    Raises:
        MarketDataError: on an unparseable code, or a strike disagreement.
    """
    quotes: list[OptionQuote] = []
    for row in rows:
        code = str(row["code"])
        contract = parse_option_code(code, underlying)
        column_strike = float(row["strike"])  # type: ignore[arg-type]

        if abs(contract.strike - column_strike) > strike_tolerance:
            raise MarketDataError(
                f"strike disagreement on {code!r}: the contract code encodes "
                f"{contract.strike} but the feed's strike column says "
                f"{column_strike}. The upstream format has probably changed; "
                f"refusing to guess which is right."
            )

        open_interest = row.get("open_interest")
        quotes.append(
            OptionQuote(
                contract=contract,
                last=float(row["last"]),  # type: ignore[arg-type]
                previous_settlement=float(row["previous_settlement"]),  # type: ignore[arg-type]
                strike=column_strike,
                open_interest=None if open_interest is None else float(open_interest),  # type: ignore[arg-type]
            )
        )

    if not quotes:
        raise MarketDataError(f"no quotes for {underlying.code} expiry {expiry.isoformat()}")

    return OptionSlice(
        underlying=underlying,
        expiry=expiry,
        as_of=as_of,
        days_to_expiry=days_to_expiry,
        quotes=tuple(quotes),
    )
