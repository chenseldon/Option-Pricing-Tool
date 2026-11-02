"""The Wind institutional panel: 725 trading days of listed option data.

This is the project's primary market data from Phase 3a onward, and it replaces
a pair of single-day akshare captures. Those captures are kept -- as the live
path and as committed offline fixtures -- but they can only ever say what one
afternoon looked like.

Why the CFFEX index options are primary
---------------------------------------
Measured on the panel, against the akshare ETF captures:

==========================  ==================  =====================
.                           akshare ETF         CFFEX index (IO/HO/MO)
==========================  ==================  =====================
usable expiries             3                   6
longest tenor               0.41 y              0.98 y
adjusted contracts          1,812 of 11,372     0
parity residual (of F)      0.08-0.69%          0.02-0.08%
matching index future       none                IF / IH / IM
==========================  ==================  =====================

The last row is the one that changes what can be *verified*: an index future
shares its expiry date with the option (confirmed on all 36 expiry tags, zero
exceptions), so the futures settlement is the forward to the option's own
expiry. Put-call parity's forward depends on the assumed rate; the future's
does not. Phase 2a had to take the rate as an input precisely because options
alone cannot identify it, and this is the independent check that was missing.

Two conventions this module gets right, both measured
-----------------------------------------------------
**1. The price is the settlement, not the close.** Put-call parity residuals on
twelve IO slices across two dates, as a fraction of the forward:

    close    0.105  0.202  0.436  0.237  0.220  0.116
             0.211  0.290  0.318  0.267  0.241  0.161   mean 0.234%
    settle   0.063  0.022  0.084  0.056  0.043  0.033
             0.021  0.031  0.016  0.044  0.045  0.031   mean 0.041%

The settlement wins 12 of 12, by 5.7x. The mechanism is the one
:mod:`otcpricer.marketdata.quotes` names as dominant: a settlement is one
exchange-wide reference struck on one convention, so it is synchronous *by
construction*, while closes are last trades at different instants.

**2. An untraded contract must not look traded.** 97,884 of the panel's
1,332,440 rows have zero volume and a non-zero close -- every one of them.
``OptionQuote.has_traded`` asks whether the last price is positive, so taken
naively the entire untraded population would read as live quotes.

These are not stale carry-forwards either: only 2.7% of zero-volume settlements
equal the previous day's, and the median absolute daily move is 3.25%. CFFEX is
marking untraded strikes *theoretically*. Fitting SSVI to them would fit the
exchange's own interpolation and call the result a market surface.

So ``last`` is set from the settlement only when volume is positive, and is
zero otherwise. ``previous_settlement`` carries the genuine prior session's
settlement, from a lagged join. Nothing in :mod:`otcpricer.marketdata.quotes`
changes, and its docstring's promise -- that the fallback is "a price from
yesterday" -- becomes literally true rather than approximately so.

Adjusted ETF contracts, and what dropping them costs
----------------------------------------------------
``ADJ_SIGN`` is ``'M'`` for a standard contract, ``'A'`` for one adjusted after
a distribution, and null on CFFEX where the field does not apply -- a null is
*not* "adjustment unknown". The flag has an exact independent corroboration: on
all 11,372 exchange-traded contracts, ``'A'`` holds if and only if
``S_INFO_COUNIT != 10000``, with zero disagreement either way. SSE carries units
of 10,075 / 10,097 / ... / 10,265 and SZSE one at 26,384.

Adjusted contracts are dropped by default. The reason is not that their strikes
are odd -- it is that an adjustment moves the strike *and* the contract unit
together, and ``OptionSlice.paired_strikes`` matches on strike alone, so a
pairing across two different unit counts would be accepted silently.

The cost is real and worth stating: on 2024-12-25, 140 of the 510050 chain's
220 rows are adjusted. Dropping them removes 64% of that day's ETF data and
leaves 3 expiries of 10 paired strikes. The same date's IO chain has 248 rows,
**zero** adjusted, zero non-positive settlements, zero expired, and 6 expiries.

That contrast is the real argument for the CFFEX options being primary, and it
is why the ETF path is not worth the complexity of unit-aware pairing yet.
``drop_adjusted=False`` exists for the experiment, not for production.

What the panel cannot do
------------------------
The static contract file is a stale snapshot: 79,078 quote rows (5.9%), across
1,704 contracts, have no static record, all of them dated 2025-09-22 or later.
Those contracts are dropped and counted rather than guessed at. A third-Friday
rule would reconstruct only 91.3% of the real maturity dates -- it misses the
holiday rolls -- so it is not used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import date
from functools import cached_property
from pathlib import Path
from typing import Any

from otcpricer.core.exceptions import MarketDataError
from otcpricer.marketdata.contracts import Underlying, resolve_underlying
from otcpricer.marketdata.quotes import OptionChain, OptionSlice, build_slice

__all__ = [
    "WIND_FILES",
    "WindPanel",
    "WindPanelSource",
]

#: The six files the panel is made of, by role.
WIND_FILES: dict[str, str] = {
    "option_static": "期权基本信息数据.parquet",
    "option_quotes": "期权日行情数据.parquet",
    "index_quotes": "指数日行情数据.parquet",
    "futures_static": "期货基本信息数据.parquet",
    "futures_quotes": "期货日行情数据.parquet",
    "rates": "国债收益率收据.xlsx",
}

#: Wind's enumeration for the option type.
_CALL_FLAG = 708001000
_PUT_FLAG = 708002000

#: ``ADJ_SIGN``: ``'M'`` standard, ``'A'`` adjusted, null on CFFEX where the
#: field does not apply. The three are not three grades of one thing, and a
#: null must never be read as "adjustment unknown".
_ADJ_STANDARD = "M"
_ADJ_ADJUSTED = "A"


def _as_date(value: object) -> date:
    """Parse Wind's ``YYYYMMDD``, which arrives as a string."""
    text = str(value).strip()
    if len(text) != 8 or not text.isdigit():
        raise MarketDataError(f"cannot read {value!r} as a Wind YYYYMMDD date")
    return date(int(text[:4]), int(text[4:6]), int(text[6:8]))


def _stamp(day: date) -> str:
    return f"{day.year:04d}{day.month:02d}{day.day:02d}"


@dataclass(frozen=True, slots=True)
class RejectionLedger:
    """Why rows were dropped, counted by reason.

    Kept per fetch so the reconciliation identity -- every raw row either
    becomes a quote or is counted here -- holds for the Wind path exactly as
    :mod:`otcpricer.marketdata.surface_input` makes it hold downstream.
    """

    raw_rows: int = 0
    no_static_record: int = 0
    adjusted_contract: int = 0
    non_positive_price: int = 0
    expired: int = 0

    @property
    def total_rejected(self) -> int:
        return (
            self.no_static_record + self.adjusted_contract + self.non_positive_price + self.expired
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "raw_rows": self.raw_rows,
            "no_static_record": self.no_static_record,
            "adjusted_contract": self.adjusted_contract,
            "non_positive_price": self.non_positive_price,
            "expired": self.expired,
            "total_rejected": self.total_rejected,
        }


#: ``华夏上证50ETF期权2309认购2.65`` / ``上证50指数期权2307认购2225`` -> expiry
#: tag, call/put, strike. Wind's numeric ETF key encodes nothing, so the name
#: is the only *independent* second reading of those fields available. Matched
#: on the numerals and one codepoint for 认购/认沽 rather than on the whole
#: string, because the localised prefix varies by issuer.
_NAME_PATTERN = re.compile(r"(?P<yymm>\d{4})(?P<kind>认购|认沽)(?P<strike>\d+(?:\.\d+)?)\s*$")


def _cross_check_name(
    name: object,
    *,
    code: str,
    strike: float,
    is_call: bool,
    expiry: date,
) -> str | None:
    """Verify a Wind row against its own contract name.

    Returns a warning string when the name cannot be parsed, ``None`` when it
    parses and agrees. Raises when it parses and *disagrees* -- that means the
    two Wind fields contradict each other, which is the format change this
    check exists to catch, and guessing which one is right would be worse than
    stopping.

    This is strictly stronger than the strike-only check
    :func:`~otcpricer.marketdata.quotes.build_slice` applies to self-describing
    codes: three fields are compared, not one.
    """
    match = _NAME_PATTERN.search(str(name))
    if match is None:
        return f"contract {code}: name {name!r} does not parse, fields unverified"

    name_is_call = match.group("kind") == "认购"
    name_strike = float(match.group("strike"))
    name_tag = match.group("yymm")
    expiry_tag = f"{expiry.year % 100:02d}{expiry.month:02d}"

    if name_is_call is not is_call:
        raise MarketDataError(
            f"contract {code}: the name says {match.group('kind')} but "
            f"S_INFO_CALLPUT says {'call' if is_call else 'put'}"
        )
    if abs(name_strike - strike) > 1e-6:
        raise MarketDataError(
            f"contract {code}: the name encodes strike {name_strike} but "
            f"S_INFO_STRIKEPRICE says {strike}"
        )
    if name_tag != expiry_tag:
        raise MarketDataError(
            f"contract {code}: the name encodes expiry {name_tag} but "
            f"S_INFO_MATURITYDATE gives {expiry_tag}"
        )
    return None


@dataclass
class WindPanel:
    """The six Wind files, loaded once and sliced per date.

    Not itself an :class:`~otcpricer.marketdata.sources.OptionChainSource`: a
    source answers for one date, and this holds all 725. The adapter that binds
    a date is :class:`WindPanelSource`.

    The expensive work lives here. Each parquet is parsed on first use and kept,
    so a panel sweep pays for it once rather than 725 times.
    """

    directory: Path
    drop_adjusted: bool = True
    _chains: dict[tuple[str, date], OptionChain] = field(default_factory=dict, repr=False)
    _ledgers: dict[tuple[str, date], RejectionLedger] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)

    # ------------------------------------------------------------------ files

    def _path(self, role: str) -> Path:
        path = self.directory / WIND_FILES[role]
        if not path.exists():
            raise MarketDataError(
                f"the Wind panel file for {role!r} is missing: {path}. "
                f"The panel is institutional data and is not committed to this "
                f"repository; point OTCPRICER_WIND_DIR at a directory holding "
                f"the six files listed in WIND_FILES, or use the 'snapshot' "
                f"source for the committed offline chains."
            )
        return path

    def _read(self, role: str) -> Any:
        import pandas as pd

        path = self._path(role)
        if path.suffix == ".xlsx":
            return pd.read_excel(path)
        return pd.read_parquet(path)

    @cached_property
    def option_static(self) -> Any:
        """Contract reference data, indexed by Wind code."""
        frame = self._read("option_static").set_index("S_INFO_WINDCODE")
        return frame[~frame.index.duplicated(keep="first")]

    @cached_property
    def option_quotes(self) -> Any:
        """Daily option quotes, with a genuine prior-session settlement.

        The lagged settlement is computed here, once, over the whole panel:
        taking it per date would require re-sorting 1.3 million rows each time.
        """
        frame = self._read("option_quotes").sort_values(["S_INFO_WINDCODE", "TRADE_DT"])
        frame["PREV_SETTLE"] = frame.groupby("S_INFO_WINDCODE")["S_DQ_SETTLE"].shift(1)
        return frame

    @cached_property
    def index_quotes(self) -> Any:
        return self._read("index_quotes")

    @cached_property
    def futures_static(self) -> Any:
        return self._read("futures_static")

    @cached_property
    def futures_quotes(self) -> Any:
        return self._read("futures_quotes")

    @cached_property
    def rates(self) -> Any:
        """The government yield file.

        Columns are localised and came back mojibake under a GBK console, so
        they are located by a digit pattern rather than by literal equality --
        the same lesson :mod:`otcpricer.marketdata.sources` records about the
        akshare board headers.
        """
        return self._read("rates")

    # ------------------------------------------------------------------ dates

    @cached_property
    def trading_days(self) -> tuple[date, ...]:
        """Every date the option panel quotes, ascending."""
        stamps = sorted(set(self.option_quotes["TRADE_DT"].astype(str)))
        return tuple(_as_date(s) for s in stamps)

    # ----------------------------------------------------------------- chain

    def chain(self, underlying_code: str, as_of: date) -> OptionChain:
        """Build one underlying's option chain as of one date.

        Rows are selected by the registry's ``wind_sccode``, never by the
        exchange name: the CFFEX block also holds 1,308 treasury-futures
        options whose underlying is a futures contract rather than an index,
        and an exchange filter would take them too.

        Raises:
            MarketDataError: if the underlying is unknown, has no Wind product
                key, or has no usable quotes on this date.
        """
        key = (str(underlying_code).strip(), as_of)
        if key in self._chains:
            return self._chains[key]

        underlying = resolve_underlying(underlying_code)
        if underlying.wind_sccode is None:
            raise MarketDataError(
                f"{underlying.code} has no Wind product key registered, so the "
                f"panel cannot be asked for it"
            )

        chain = self._build_chain(underlying, as_of)
        self._chains[key] = chain
        return chain

    def _build_chain(self, underlying: Underlying, as_of: date) -> OptionChain:
        import numpy as np

        sccode = underlying.wind_sccode
        if sccode is None:  # pragma: no cover - chain() rejects this first
            raise MarketDataError(f"{underlying.code} has no Wind product key")

        static = self.option_static
        codes = static.index[static["S_INFO_SCCODE"] == sccode]
        if len(codes) == 0:
            raise MarketDataError(f"the panel holds no contracts for product {sccode!r}")

        quotes = self.option_quotes
        day = quotes[quotes["TRADE_DT"].astype(str) == _stamp(as_of)]
        if day.empty:
            raise MarketDataError(
                f"the panel has no quotes dated {as_of.isoformat()}; it covers "
                f"{self.trading_days[0].isoformat()} to {self.trading_days[-1].isoformat()}"
            )

        known = set(codes)
        # Rows for this product, including those whose static record is missing:
        # they have to be counted, not filtered away upstream of the ledger.
        product_prefix = sccode.split(".")[0]
        day = day[
            day["S_INFO_WINDCODE"].isin(known)
            | day["S_INFO_WINDCODE"].astype(str).str.startswith(product_prefix)
        ]

        raw_rows = int(len(day))
        no_static = int((~day["S_INFO_WINDCODE"].isin(known)).sum())
        day = day[day["S_INFO_WINDCODE"].isin(known)]

        adjusted = non_positive = expired = 0
        by_expiry: dict[date, list[dict[str, object]]] = {}
        warnings: list[str] = []

        for row in day.itertuples(index=False):
            code = str(row.S_INFO_WINDCODE)
            info = static.loc[code]

            if self.drop_adjusted and str(info.ADJ_SIGN) == _ADJ_ADJUSTED:
                adjusted += 1
                continue

            settle = float(row.S_DQ_SETTLE)
            prev_settle = float(row.PREV_SETTLE) if np.isfinite(row.PREV_SETTLE) else 0.0
            if not np.isfinite(settle) or settle <= 0.0:
                # 8,697 rows in the panel carry a zero settlement. Zero is not
                # a price.
                non_positive += 1
                continue

            expiry = _as_date(info.S_INFO_MATURITYDATE)
            days_to_expiry = (expiry - as_of).days
            if days_to_expiry <= 0:
                expired += 1
                continue

            is_call = int(float(info.S_INFO_CALLPUT)) == _CALL_FLAG
            strike = float(info.S_INFO_STRIKEPRICE)

            warning = _cross_check_name(
                info.S_INFO_NAME, code=code, strike=strike, is_call=is_call, expiry=expiry
            )
            if warning is not None:
                warnings.append(warning)

            volume = float(row.S_DQ_VOLUME) if np.isfinite(row.S_DQ_VOLUME) else 0.0
            by_expiry.setdefault(expiry, []).append(
                {
                    # build_slice reads the strike off the code as an
                    # independent check where the code carries one. Wind's
                    # numeric ETF keys do not, so those are handed the static
                    # record's own code shape and verified against the name
                    # above instead.
                    "code": self._slice_code(underlying, code, info, is_call, strike),
                    # The measured convention: a settlement price, and only
                    # when the contract actually traded.
                    "last": settle if volume > 0.0 else 0.0,
                    "previous_settlement": prev_settle,
                    "strike": strike,
                    "open_interest": float(row.S_DQ_OI) if np.isfinite(row.S_DQ_OI) else None,
                }
            )

        ledger = RejectionLedger(
            raw_rows=raw_rows,
            no_static_record=no_static,
            adjusted_contract=adjusted,
            non_positive_price=non_positive,
            expired=expired,
        )
        self._ledgers[(underlying.code, as_of)] = ledger

        if not by_expiry:
            raise MarketDataError(
                f"no usable {underlying.code} quotes on {as_of.isoformat()}; "
                f"rejections: {ledger.as_dict()}"
            )

        slices: list[OptionSlice] = [
            build_slice(
                underlying,
                rows,
                expiry=expiry,
                as_of=as_of,
                days_to_expiry=(expiry - as_of).days,
            )
            for expiry, rows in sorted(by_expiry.items())
        ]

        kept = sum(len(s) for s in slices)
        if kept + ledger.total_rejected != raw_rows:
            raise MarketDataError(
                f"reconciliation failed for {underlying.code} on {as_of.isoformat()}: "
                f"{kept} kept + {ledger.total_rejected} rejected != {raw_rows} raw rows"
            )
        warnings.append(
            f"rejections: {ledger.total_rejected} of {raw_rows} raw rows ({ledger.as_dict()})"
        )

        if no_static:
            warnings.append(
                f"{no_static} quote rows have no static record and were dropped; "
                f"the static file is a snapshot and stops before the quote panel does"
            )

        return OptionChain(
            underlying=underlying,
            as_of=as_of,
            slices=tuple(slices),
            source=f"wind:{as_of.isoformat()}:settle",
            warnings=tuple(warnings),
        )

    @staticmethod
    def _slice_code(
        underlying: Underlying, code: str, info: Any, is_call: bool, strike: float
    ) -> str:
        """The contract code to hand :func:`build_slice`.

        CFFEX codes are self-describing, so they pass through and the strike
        cross-check runs for real. Wind's numeric ETF keys are not, so the
        exchange-style code is reconstructed from the static record -- which
        makes build_slice's strike check tautological for those rows, and is
        why :func:`_cross_check_name` does the real verification instead. Said
        plainly rather than left to be discovered.
        """
        from otcpricer.marketdata.contracts import CodeStyle

        if underlying.code_style is CodeStyle.CFFEX_INDEX:
            return code
        expiry = _as_date(info.S_INFO_MATURITYDATE)
        scaled = int(round(strike * underlying.strike_scale))
        return (
            f"{underlying.code}{'C' if is_call else 'P'}"
            f"{expiry.year % 100:02d}{expiry.month:02d}M{scaled:05d}"
        )

    def ledger(self, underlying_code: str, as_of: date) -> RejectionLedger:
        """Why rows were dropped for one fetch.

        Populated by :meth:`chain`, so call that first. The reconciliation
        identity -- kept plus rejected equals raw -- is *enforced* there rather
        than merely reported, because a screen whose arithmetic does not close
        is indistinguishable from a screen that silently loses data.
        """
        key = (str(underlying_code).strip(), as_of)
        if key not in self._ledgers:
            raise MarketDataError(
                f"no rejection ledger for {underlying_code} on {as_of.isoformat()}; "
                f"build the chain first"
            )
        return self._ledgers[key]

    # ------------------------------------------------------- spot and futures

    def index_close(self, index_code: str, as_of: date) -> float:
        """The settling index's close. Reporting only.

        The surface works in forward terms and never needs spot; this feeds the
        implied-carry line of a report and anchors a local-volatility grid.
        Anything the *surface* touches uses
        :attr:`~otcpricer.marketdata.forward.ForwardCurve.spot_anchor`, so a
        spot/forward mismatch cannot reach the moneyness coordinate.
        """
        frame = self.index_quotes
        hit = frame[
            (frame["S_INFO_WINDCODE"] == index_code)
            & (frame["TRADE_DT"].astype(str) == _stamp(as_of))
        ]
        if hit.empty:
            raise MarketDataError(f"no close for index {index_code} on {as_of.isoformat()}")
        return float(hit["S_DQ_CLOSE"].iloc[0])

    def futures(self, root: str, as_of: date) -> tuple[tuple[date, float], ...]:
        """``(expiry, settlement)`` for one futures root, ascending by expiry.

        The expiry comes from ``S_INFO_DELISTDATE``, which matches the option
        ``S_INFO_MATURITYDATE`` on all 36 expiry tags with zero exceptions. That
        exact agreement is what makes the forward check interpolation-free: the
        two instruments expire on the same day, so the future's settlement *is*
        the forward to the option's expiry.

        Only four contracts list at a time against six option expiries, so the
        two longest expiries are unchecked. Callers must say so rather than
        quietly covering four of six.
        """
        import numpy as np

        static = self.futures_static
        mine = static[static["S_INFO_WINDCODE"].astype(str).str.startswith(root)]
        expiry_by_code = {
            str(r.S_INFO_WINDCODE): _as_date(r.S_INFO_DELISTDATE)
            for r in mine.itertuples(index=False)
        }

        quotes = self.futures_quotes
        day = quotes[
            (quotes["TRADE_DT"].astype(str) == _stamp(as_of))
            & quotes["S_INFO_WINDCODE"].isin(expiry_by_code)
        ]

        out: list[tuple[date, float]] = []
        for row in day.itertuples(index=False):
            settle = float(row.S_DQ_SETTLE)
            if not np.isfinite(settle) or settle <= 0.0:
                continue
            out.append((expiry_by_code[str(row.S_INFO_WINDCODE)], settle))
        return tuple(sorted(out))


@dataclass(frozen=True)
class WindPanelSource:
    """One date's view of a :class:`WindPanel`, as an ``OptionChainSource``.

    Frozen on purpose, and :meth:`at` returns a new instance rather than
    mutating. :class:`~otcpricer.marketdata.sources.CachingSource` keys on the
    underlying code alone, so a source whose date could change under it would
    serve the wrong day's chain with no error at all.

    The date belongs here rather than on ``fetch``: widening the Protocol would
    force :class:`~otcpricer.marketdata.sources.AkshareSource` to accept an
    argument it cannot honour, since the live board returns today and only
    today. ``SnapshotSource(snapshot=...)`` already sets this precedent -- "which
    capture" has always been constructor state.
    """

    panel: WindPanel
    as_of: date

    @property
    def name(self) -> str:
        return f"wind:{self.as_of.isoformat()}:settle"

    def fetch(self, underlying_code: str) -> OptionChain:
        return self.panel.chain(underlying_code, self.as_of)

    def at(self, as_of: date) -> WindPanelSource:
        """This panel seen from another date. A panel sweep is this in a loop."""
        return replace(self, as_of=as_of)
