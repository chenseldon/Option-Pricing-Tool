"""Listed option contract identifiers, and the registry of supported underlyings.

Why the registry is explicit
----------------------------
The upstream feed is addressed by a Chinese product name, and picking the wrong
one fails *silently*. Measured during exploration: asking for
``嘉实沪深300ETF期权`` returns 136 rows whose strike column is uniformly
``10000`` -- a different payload shape entirely, not an error. Anything built on
it would produce plausible-looking nonsense.

So the mapping from an underlying to its feed name is written down once, here,
with a test, rather than passed around as a string by each caller.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from otcpricer.core.exceptions import MarketDataError
from otcpricer.core.types import OptionType

__all__ = [
    "SUPPORTED_UNDERLYINGS",
    "ContractCode",
    "Underlying",
    "parse_contract_code",
    "resolve_underlying",
]


@dataclass(frozen=True, slots=True)
class Underlying:
    """A listed-option underlying and how to ask the feed for it.

    Attributes:
        code: Exchange ticker, e.g. ``"510050"``.
        name: Human-readable name.
        feed_symbol: The exact product name the upstream board endpoint expects.
        strike_scale: Divisor converting the strike embedded in a contract code
            to a price. ETF option codes carry the strike multiplied by 1000.
        contract_multiplier: Shares per contract, for notional calculations.
    """

    code: str
    name: str
    feed_symbol: str
    strike_scale: float = 1000.0
    contract_multiplier: int = 10_000


#: The underlyings this project supports, keyed by exchange ticker.
#:
#: Both entries were verified against the live feed: 50ETF returns 24 rows
#: (12 calls, 12 puts) and 300ETF 28 rows, with the strike embedded in every
#: contract code matching the strike column exactly.
SUPPORTED_UNDERLYINGS: dict[str, Underlying] = {
    "510050": Underlying(
        code="510050",
        name="华夏上证50ETF",
        feed_symbol="华夏上证50ETF期权",
    ),
    "510300": Underlying(
        code="510300",
        name="华泰柏瑞沪深300ETF",
        # Deliberately 华泰柏瑞 and not 嘉实: the latter answers to a similar
        # name but returns an incompatible payload rather than failing.
        feed_symbol="华泰柏瑞沪深300ETF期权",
    ),
}


def resolve_underlying(code: str) -> Underlying:
    """Look up an underlying by exchange ticker.

    Raises:
        MarketDataError: if the ticker is not one this project supports, listing
            those that are. Better than passing an unknown name to the feed and
            interpreting whatever comes back.
    """
    key = str(code).strip()
    if key in SUPPORTED_UNDERLYINGS:
        return SUPPORTED_UNDERLYINGS[key]
    supported = ", ".join(sorted(SUPPORTED_UNDERLYINGS))
    raise MarketDataError(f"underlying {code!r} is not supported; available: {supported}")


#: ``510050C2611M02800`` -> underlying, call/put, expiry year-month, strike,
#: and an optional trailing letter marking a contract adjusted for a
#: distribution.
_CODE_PATTERN = re.compile(
    r"^(?P<underlying>\d{6})"
    r"(?P<kind>[CP])"
    r"(?P<yymm>\d{4})"
    r"M"
    r"(?P<strike>\d+)"
    r"(?P<adjustment>[A-Z]?)$"
)


@dataclass(frozen=True, slots=True)
class ContractCode:
    """A parsed listed-option contract identifier.

    Attributes:
        raw: The original code.
        underlying: Exchange ticker of the underlying.
        option_type: Call or put.
        expiry_year: Four-digit year.
        expiry_month: Month, 1-12.
        strike: Strike price, already scaled.
        adjustment: Trailing adjustment marker, empty when absent. A contract
            carrying one has had its terms changed by a distribution, so its
            strike no longer sits on the standard ladder.
    """

    raw: str
    underlying: str
    option_type: OptionType
    expiry_year: int
    expiry_month: int
    strike: float
    adjustment: str = ""

    @property
    def is_adjusted(self) -> bool:
        """Whether this contract was adjusted for a distribution."""
        return bool(self.adjustment)

    @property
    def expiry_tag(self) -> str:
        """The feed's expiry key, e.g. ``"2611"``."""
        return f"{self.expiry_year % 100:02d}{self.expiry_month:02d}"


def parse_contract_code(code: str, *, strike_scale: float | None = None) -> ContractCode:
    """Parse a listed-option contract code.

    Args:
        code: Contract identifier, e.g. ``"510050C2611M02800"``.
        strike_scale: Divisor for the embedded strike. Defaults to the value
            registered for the underlying.

    Returns:
        The parsed :class:`ContractCode`.

    Raises:
        MarketDataError: if the code does not match the expected form, or names
            an unsupported underlying.

    Note:
        The strike recovered here is a *cross-check*, not the source of truth --
        the board supplies a strike column directly, and
        :func:`~otcpricer.marketdata.quotes.build_slice` verifies the two agree.
        Two independent readings of the same field is cheap insurance against a
        feed format change going unnoticed.
    """
    match = _CODE_PATTERN.match(str(code).strip())
    if match is None:
        raise MarketDataError(
            f"cannot parse contract code {code!r}; expected a form like "
            f"'510050C2611M02800' (underlying, C/P, YYMM, 'M', strike)"
        )

    underlying = match.group("underlying")
    scale = (
        strike_scale if strike_scale is not None else resolve_underlying(underlying).strike_scale
    )

    yymm = match.group("yymm")
    year_short, month = int(yymm[:2]), int(yymm[2:])
    if not 1 <= month <= 12:
        raise MarketDataError(f"contract code {code!r} carries an invalid month {month:02d}")

    return ContractCode(
        raw=str(code).strip(),
        underlying=underlying,
        option_type=OptionType.CALL if match.group("kind") == "C" else OptionType.PUT,
        # The feed uses two-digit years; these contracts are all 21st century.
        expiry_year=2000 + year_short,
        expiry_month=month,
        strike=int(match.group("strike")) / scale,
        adjustment=match.group("adjustment"),
    )
