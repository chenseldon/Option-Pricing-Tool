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
from enum import Enum

from otcpricer.core.exceptions import MarketDataError
from otcpricer.core.types import OptionType

__all__ = [
    "SUPPORTED_UNDERLYINGS",
    "CodeStyle",
    "ContractCode",
    "Underlying",
    "parse_cffex_code",
    "parse_contract_code",
    "parse_option_code",
    "resolve_underlying",
]


class CodeStyle(Enum):
    """How an underlying's contract codes are spelled.

    Three feeds, three unrelated spellings of the same identifier, so the
    parser is chosen from the registry rather than guessed from the string:

    - ``SSE_ETF``: ``510050C2611M02800`` -- the exchange's own form, used by
      the akshare board. Self-describing: type, expiry and strike are all in
      the code.
    - ``CFFEX_INDEX``: ``IO2307-C-3400.CFE`` -- also self-describing, but a
      completely different shape, and the strike is in index points rather
      than scaled by a thousand.
    - ``WIND_NUMERIC``: ``10005105.SH`` -- Wind's own key for an ETF option.
      Encodes *nothing*. See :func:`parse_option_code` for what this costs.
    """

    SSE_ETF = "sse_etf"
    CFFEX_INDEX = "cffex_index"
    WIND_NUMERIC = "wind_numeric"


@dataclass(frozen=True, slots=True)
class Underlying:
    """A listed-option underlying and how to ask the feed for it.

    Attributes:
        code: Exchange ticker, e.g. ``"510050"``.
        name: Human-readable name.
        feed_symbol: The exact product name the upstream board endpoint
            expects, or ``None`` for an underlying no live board serves. The
            CFFEX index options arrive only from the Wind panel, so there is
            no board nickname to register and pretending otherwise would put a
            string in the registry that nothing can use.
        strike_scale: Divisor converting the strike embedded in a contract code
            to a price. ETF option codes carry the strike multiplied by 1000;
            a CFFEX code carries index points, so the scale is 1.
        contract_multiplier: Shares per contract, for notional calculations.
            Note this is a *per-underlying* default and adjusted ETF contracts
            violate it -- see ``ADJ_SIGN`` handling in
            :mod:`otcpricer.marketdata.wind`.
        tick_size: Minimum price increment. This matters more than it looks:
            a quote sitting at one tick is a *floor*, not a price, and
            inverting it yields a meaningless implied volatility. See
            :mod:`otcpricer.marketdata.surface_input`.
        code_style: Which spelling this underlying's contract codes use.
        index_code: The spot index this option settles against, where one
            exists. Reporting only -- the surface works in forward terms.
        futures_root: The index future sharing this product's expiry calendar,
            which is what makes an independent forward check possible.
        wind_sccode: Wind's product key, used to select this underlying's rows
            out of the panel. Written down rather than derived: the CFFEX block
            also carries 1,308 treasury-futures options whose underlying is a
            futures contract, not an index, and a filter on the exchange name
            would swallow them whole.
    """

    code: str
    name: str
    feed_symbol: str | None
    strike_scale: float = 1000.0
    contract_multiplier: int = 10_000
    tick_size: float = 1e-4
    code_style: CodeStyle = CodeStyle.SSE_ETF
    index_code: str | None = None
    futures_root: str | None = None
    wind_sccode: str | None = None


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
        wind_sccode="510050OP.SH",
    ),
    "510300": Underlying(
        code="510300",
        name="华泰柏瑞沪深300ETF",
        # Deliberately 华泰柏瑞 and not 嘉实: the latter answers to a similar
        # name but returns an incompatible payload rather than failing.
        feed_symbol="华泰柏瑞沪深300ETF期权",
        wind_sccode="510300OP.SH",
    ),
    # ----------------------------------------------------------------- CFFEX
    # The index options, which the Wind panel serves and no live board does.
    #
    # These are the primary underlyings for surface work and the reasons are
    # measured, not stylistic: six expiries reaching 0.98 years against the
    # ETFs' four reaching 0.5, zero adjusted contracts against 1,812, and
    # settlement-price parity residuals of 0.02-0.08% against 0.08-0.69%.
    #
    # ``tick_size`` is 0.2 because that is the *trading* tick: every one of the
    # 478,584 index-option closes in the panel is a multiple of 0.2 and the
    # smallest is exactly 0.2. The settlement column is carried to 0.01, so
    # reading the tick off the price actually used would understate it
    # twentyfold and re-admit the floor quotes that
    # :mod:`otcpricer.marketdata.surface_input` exists to reject.
    "IO": Underlying(
        code="IO",
        name="沪深300股指期权",
        feed_symbol=None,
        strike_scale=1.0,
        contract_multiplier=100,
        tick_size=0.2,
        code_style=CodeStyle.CFFEX_INDEX,
        index_code="000300.SH",
        futures_root="IF",
        wind_sccode="IO.CFE",
    ),
    "HO": Underlying(
        code="HO",
        name="上证50股指期权",
        feed_symbol=None,
        strike_scale=1.0,
        contract_multiplier=100,
        tick_size=0.2,
        code_style=CodeStyle.CFFEX_INDEX,
        index_code="000016.SH",
        futures_root="IH",
        wind_sccode="HO.CFE",
    ),
    "MO": Underlying(
        code="MO",
        name="中证1000股指期权",
        feed_symbol=None,
        strike_scale=1.0,
        contract_multiplier=100,
        tick_size=0.2,
        code_style=CodeStyle.CFFEX_INDEX,
        index_code="000852.SH",
        futures_root="IM",
        wind_sccode="MO.CFE",
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


#: ``IO2307-C-3400.CFE`` -> root, call/put, expiry year-month, strike in index
#: points. Deliberately a separate pattern rather than a widened one: the SSE
#: form is pinned by its own tests and is correct for what it parses, and one
#: regex matching two unrelated shapes is how a format change goes unnoticed.
_CFFEX_PATTERN = re.compile(
    r"^(?P<root>[A-Z]{2})"
    r"(?P<yymm>\d{4})"
    r"-(?P<kind>[CP])-"
    r"(?P<strike>\d+(?:\.\d+)?)"
    r"(?:\.CFE)?$"
)


def parse_cffex_code(code: str) -> ContractCode:
    """Parse a CFFEX index-option code.

    The strike is in index points, so unlike the SSE form there is no scaling
    to undo. ``adjustment`` is always empty: CFFEX does not adjust index
    options, which is verified in the panel -- all 6,012 index-option contracts
    carry a null ``ADJ_SIGN``.

    Raises:
        MarketDataError: on an unparseable code, or an impossible month.
    """
    match = _CFFEX_PATTERN.match(str(code).strip())
    if match is None:
        raise MarketDataError(
            f"cannot parse CFFEX contract code {code!r}; expected a form like "
            f"'IO2307-C-3400.CFE' (root, YYMM, C/P, strike in index points)"
        )

    yymm = match.group("yymm")
    year_short, month = int(yymm[:2]), int(yymm[2:])
    if not 1 <= month <= 12:
        raise MarketDataError(f"contract code {code!r} carries an invalid month {month:02d}")

    return ContractCode(
        raw=str(code).strip(),
        underlying=match.group("root"),
        option_type=OptionType.CALL if match.group("kind") == "C" else OptionType.PUT,
        expiry_year=2000 + year_short,
        expiry_month=month,
        strike=float(match.group("strike")),
    )


def parse_option_code(code: str, underlying: Underlying) -> ContractCode:
    """Parse a contract code using the spelling its underlying actually uses.

    Dispatching on the registry rather than trying each pattern in turn: a
    fallback chain would silently accept a code from the wrong feed, and the
    registry already exists to make exactly this kind of mapping explicit.

    ``WIND_NUMERIC`` has no parser because there is nothing to parse -- see
    :mod:`otcpricer.marketdata.wind` for how those rows are handled instead.

    Raises:
        MarketDataError: on an unparseable code, or a style with no parser.
    """
    if underlying.code_style is CodeStyle.CFFEX_INDEX:
        return parse_cffex_code(code)
    if underlying.code_style is CodeStyle.SSE_ETF:
        return parse_contract_code(code, strike_scale=underlying.strike_scale)
    raise MarketDataError(
        f"contract codes for {underlying.code} are {underlying.code_style.value} and "
        f"encode no contract detail, so {code!r} cannot be parsed; the static "
        f"record is the only source for these fields"
    )
