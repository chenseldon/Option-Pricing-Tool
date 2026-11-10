"""Tests for the Wind panel adapter.

Split deliberately into two halves. The code-level half -- parsers, registry,
the name cross-check, the source contract -- runs anywhere. The panel half
needs the six Wind files, which are institutional data and are not committed,
so it skips when they are absent rather than failing. Same convention the
QuantLib and snapshot fixtures already use.

The panel tests pin *measured* facts, so a different panel vintage fails loudly
instead of quietly shifting every statistic downstream.
"""

from __future__ import annotations

import os
from dataclasses import FrozenInstanceError
from datetime import date
from pathlib import Path

import pytest

from otcpricer.core.exceptions import MarketDataError
from otcpricer.marketdata.contracts import (
    SUPPORTED_UNDERLYINGS,
    CodeStyle,
    parse_cffex_code,
    parse_option_code,
    resolve_underlying,
)
from otcpricer.marketdata.sources import OptionChainSource, build_source
from otcpricer.marketdata.wind import (
    WIND_FILES,
    WindPanel,
    WindPanelSource,
    _cross_check_name,
)

#: Where the panel sits on the machine it was developed against. Only a
#: fallback -- OTCPRICER_WIND_DIR wins, and absence just skips.
_DEFAULT_WIND_DIR = r"D:\Desk\国泰海通\实习研究项目\data_summerintern_2027\data_summerintern_2027"


def _wind_dir() -> Path | None:
    """The panel directory, or ``None`` if the panel cannot actually be read.

    Two conditions, not one. The files have to be there *and* a parquet engine
    has to be installed -- four of the six files are parquet, and pandas needs
    pyarrow or fastparquet to open them.

    Checking only for the files is what this used to do, and it meant the gate
    did not gate what it claimed. On a machine with the data directory and a
    minimal install, every panel test *failed* with an ImportError from deep
    inside pandas instead of skipping. Found while replicating CI's toolchain
    in a bare virtualenv, which is exactly the situation a reader reproducing
    this project would be in.
    """
    path = Path(os.getenv("OTCPRICER_WIND_DIR") or _DEFAULT_WIND_DIR)
    if not all((path / name).exists() for name in WIND_FILES.values()):
        return None
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        try:
            import fastparquet  # noqa: F401
        except ImportError:
            return None
    return path


WIND_DIR = _wind_dir()
needs_panel = pytest.mark.skipif(
    WIND_DIR is None,
    reason="Wind panel absent, or no parquet engine to read it with (pip install pyarrow)",
)


# ---------------------------------------------------------------------------
# Contract codes
# ---------------------------------------------------------------------------


class TestCffexCodes:
    @pytest.mark.parametrize(
        ("code", "root", "is_call", "year", "month", "strike"),
        [
            ("IO2307-C-3400.CFE", "IO", True, 2023, 7, 3400.0),
            ("HO2601-C-2225.CFE", "HO", True, 2026, 1, 2225.0),
            ("MO2512-P-6200.CFE", "MO", False, 2025, 12, 6200.0),
            # The exchange suffix is optional, and a fractional strike parses.
            ("MO2512-P-6200", "MO", False, 2025, 12, 6200.0),
            ("IO2403-C-3512.5", "IO", True, 2024, 3, 3512.5),
        ],
    )
    def test_parses_the_cffex_shape(
        self, code: str, root: str, is_call: bool, year: int, month: int, strike: float
    ) -> None:
        parsed = parse_cffex_code(code)
        assert parsed.underlying == root
        assert (parsed.option_type.value == "c") is is_call
        assert (parsed.expiry_year, parsed.expiry_month) == (year, month)
        assert parsed.strike == strike
        # CFFEX does not adjust index options, verified on all 6,012 contracts.
        assert parsed.adjustment == ""

    def test_the_strike_is_index_points_not_scaled(self) -> None:
        """The SSE form carries the strike times a thousand; this one does not.

        Getting this wrong would not raise -- it would silently price a 3400
        strike as 3.4, which is why ``strike_scale`` is in the registry.
        """
        assert parse_cffex_code("IO2307-C-3400.CFE").strike == 3400.0
        assert resolve_underlying("IO").strike_scale == 1.0

    @pytest.mark.parametrize(
        "code",
        [
            "510050C2611M02800",  # the SSE form must not be accepted here
            "10005105.SH",  # Wind's numeric key
            "IO2313-C-3400.CFE",  # month 13
            "IO2307-X-3400.CFE",  # not a call or a put
            "I2307-C-3400.CFE",  # one-letter root
            "",
        ],
    )
    def test_refuses_anything_else(self, code: str) -> None:
        with pytest.raises(MarketDataError):
            parse_cffex_code(code)


class TestCodeStyleDispatch:
    def test_each_style_goes_to_its_own_parser(self) -> None:
        assert parse_option_code("IO2307-C-3400.CFE", resolve_underlying("IO")).strike == 3400.0
        assert parse_option_code("510050C2611M02800", resolve_underlying("510050")).strike == 2.8

    def test_a_code_from_the_wrong_feed_is_refused_not_guessed(self) -> None:
        """No fallback chain, on purpose.

        Trying each pattern in turn would make a code from the wrong feed
        parse as something plausible. The registry says which spelling an
        underlying uses, so there is nothing to guess.
        """
        with pytest.raises(MarketDataError):
            parse_option_code("510050C2611M02800", resolve_underlying("IO"))
        with pytest.raises(MarketDataError):
            parse_option_code("IO2307-C-3400.CFE", resolve_underlying("510050"))

    def test_wind_numeric_codes_say_they_encode_nothing(self) -> None:
        from dataclasses import replace

        numeric = replace(resolve_underlying("510050"), code_style=CodeStyle.WIND_NUMERIC)
        with pytest.raises(MarketDataError, match="encode no contract detail"):
            parse_option_code("10005105.SH", numeric)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    @pytest.mark.parametrize(
        ("root", "index_code", "futures_root"),
        [("IO", "000300.SH", "IF"), ("HO", "000016.SH", "IH"), ("MO", "000852.SH", "IM")],
    )
    def test_the_cffex_roots_carry_their_index_and_future(
        self, root: str, index_code: str, futures_root: str
    ) -> None:
        """The future is the point: it shares the option's expiry calendar.

        That is what makes an independent forward check possible, and the
        mapping is written down with a test rather than derived from the
        string, exactly as the ETF feed symbols are.
        """
        underlying = resolve_underlying(root)
        assert underlying.index_code == index_code
        assert underlying.futures_root == futures_root
        assert underlying.wind_sccode == f"{root}.CFE"
        assert underlying.code_style is CodeStyle.CFFEX_INDEX
        assert underlying.contract_multiplier == 100

    def test_the_index_tick_is_the_trading_tick_not_the_settles_resolution(self) -> None:
        """0.2, and the twentyfold trap this avoids.

        Every one of the panel's index-option closes is a multiple of 0.2 and
        the smallest is exactly 0.2 -- quotes do rest on the floor. But the
        settlement column, which is the price actually used, is carried to
        0.01. Reading the tick off that column would understate it twentyfold
        and re-admit the floor quotes that surface_input exists to reject.
        """
        for root in ("IO", "HO", "MO"):
            assert resolve_underlying(root).tick_size == 0.2

    def test_underlyings_with_no_live_board_say_so(self) -> None:
        """Rather than carrying a feed symbol nothing can use."""
        for root in ("IO", "HO", "MO"):
            assert resolve_underlying(root).feed_symbol is None
        for code in ("510050", "510300"):
            assert resolve_underlying(code).feed_symbol is not None

    def test_every_underlying_has_a_wind_product_key(self) -> None:
        for code, underlying in SUPPORTED_UNDERLYINGS.items():
            assert underlying.wind_sccode, f"{code} cannot be looked up in the panel"


# ---------------------------------------------------------------------------
# The name cross-check
# ---------------------------------------------------------------------------

#: Real rows out of the panel's static file, in both name shapes.
_ETF_NAME = "\u534e\u590f\u4e0a\u8bc150ETF\u671f\u67432309\u8ba4\u8d2d2.65"
_CFFEX_NAME = "\u4e0a\u8bc150\u6307\u6570\u671f\u67432307\u8ba4\u8d2d2225"


class TestNameCrossCheck:
    """Wind's numeric ETF keys encode nothing, so the name is the second read.

    Stronger than the strike-only check a self-describing code allows: three
    fields are compared, not one.
    """

    def test_accepts_a_row_that_agrees(self) -> None:
        assert (
            _cross_check_name(
                _ETF_NAME,
                code="10005105.SH",
                strike=2.65,
                is_call=True,
                expiry=date(2023, 9, 27),
            )
            is None
        )

    def test_accepts_the_cffex_name_shape_too(self) -> None:
        assert (
            _cross_check_name(
                _CFFEX_NAME,
                code="HO2307-C-2225.CFE",
                strike=2225.0,
                is_call=True,
                expiry=date(2023, 7, 21),
            )
            is None
        )

    @pytest.mark.parametrize(
        ("strike", "is_call", "expiry", "fragment"),
        [
            (2.70, True, date(2023, 9, 27), "strike"),
            (2.65, False, date(2023, 9, 27), "says"),
            (2.65, True, date(2023, 12, 27), "expiry"),
        ],
    )
    def test_a_disagreement_stops_rather_than_picking_a_side(
        self, strike: float, is_call: bool, expiry: date, fragment: str
    ) -> None:
        """Two Wind fields contradicting each other means the format moved.

        Guessing which one is right is worse than stopping -- that is the
        failure this check exists to make loud.
        """
        with pytest.raises(MarketDataError, match=fragment):
            _cross_check_name(
                _ETF_NAME,
                code="10005105.SH",
                strike=strike,
                is_call=is_call,
                expiry=expiry,
            )

    def test_an_unparseable_name_warns_rather_than_raising(self) -> None:
        """Unverified is not the same as contradicted."""
        warning = _cross_check_name(
            "something else entirely",
            code="10005105.SH",
            strike=2.65,
            is_call=True,
            expiry=date(2023, 9, 27),
        )
        assert warning is not None
        assert "unverified" in warning


# ---------------------------------------------------------------------------
# The source contract
# ---------------------------------------------------------------------------


class TestSourceContract:
    def test_the_adapter_satisfies_the_protocol_unchanged(self) -> None:
        """The date is constructor state, so ``fetch`` keeps its signature.

        Widening the Protocol would force AkshareSource to accept an argument
        it cannot honour: the live board returns today and only today.

        Checked structurally rather than with ``isinstance``, because
        ``OptionChainSource`` is not ``@runtime_checkable`` and making it so
        just to satisfy a test would be changing the very interface this test
        exists to show was left alone.
        """
        import inspect

        source = WindPanelSource(panel=WindPanel("nowhere"), as_of=date(2024, 12, 25))
        assert source.name == "wind:2024-12-25:settle"

        expected = inspect.signature(OptionChainSource.fetch)
        assert inspect.signature(type(source).fetch) == expected

    def test_rebinding_the_date_returns_a_new_source(self) -> None:
        """Frozen on purpose: CachingSource keys on the underlying code alone.

        A source whose date could change under that cache would serve the
        wrong day's chain with no error at all.
        """
        first = WindPanelSource(panel=WindPanel("nowhere"), as_of=date(2024, 12, 25))
        second = first.at(date(2025, 6, 10))

        assert second is not first
        assert first.as_of == date(2024, 12, 25)
        assert second.as_of == date(2025, 6, 10)
        with pytest.raises(FrozenInstanceError):
            first.as_of = date(2020, 1, 1)  # type: ignore[misc]

    def test_a_missing_panel_says_what_to_do_about_it(self) -> None:
        panel = WindPanel(Path("definitely") / "not" / "here")
        with pytest.raises(MarketDataError, match="not committed to this repository"):
            panel.chain("IO", date(2024, 12, 25))

    def test_the_factory_requires_a_date_and_a_directory(self) -> None:
        with pytest.raises(MarketDataError, match="requires as_of"):
            build_source("wind", wind_dir="somewhere")
        with pytest.raises(MarketDataError, match="requires wind_dir"):
            build_source("wind", as_of=date(2024, 12, 25))

    def test_the_factory_still_names_every_valid_kind(self) -> None:
        with pytest.raises(MarketDataError, match="'akshare', 'snapshot', 'csv', 'wind'"):
            build_source("nonsense")


# ---------------------------------------------------------------------------
# The panel itself
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def panel() -> WindPanel:
    assert WIND_DIR is not None
    return WindPanel(WIND_DIR)


#: A date with a complete chain on every root, used throughout below.
A_DATE = date(2024, 12, 25)


@needs_panel
class TestPanelShape:
    def test_covers_725_trading_days(self, panel: WindPanel) -> None:
        days = panel.trading_days
        assert len(days) == 725
        assert days[0] == date(2023, 6, 30)
        assert days[-1] == date(2026, 6, 29)
        assert list(days) == sorted(days)

    def test_the_cffex_roots_reach_six_expiries_and_a_year(self, panel: WindPanel) -> None:
        """The whole reason these are the primary underlyings.

        Against the akshare captures' 3 usable expiries reaching 0.41 years.
        """
        for root in ("IO", "HO", "MO"):
            chain = panel.chain(root, A_DATE)
            assert len(chain.slices) == 6
            assert max(s.days_to_expiry for s in chain.slices) == 359

    def test_treasury_options_are_excluded_by_product_not_by_exchange(
        self, panel: WindPanel
    ) -> None:
        """The CFFEX block also holds 1,308 treasury-futures options.

        Their underlying is a futures contract, not an index, and an exchange
        filter would take them too -- structurally the same failure as the
        wrong-issuer payload the registry was written about. They also carry
        zero quote rows, which is the other half of why equity-only is a
        statement about the data and not just a scope decision.
        """
        roots = {str(c)[:2] for c in panel.option_static["S_INFO_SCCODE"].unique()}
        assert {"TF", "TS", "TL", "TT"} & roots, "expected treasury roots in the static file"

        for root in ("IO", "HO", "MO"):
            chain = panel.chain(root, A_DATE)
            for option_slice in chain.slices:
                for quote in option_slice.quotes:
                    assert quote.contract.underlying == root

    def test_the_adjustment_flag_has_an_exact_second_witness(self, panel: WindPanel) -> None:
        """``ADJ_SIGN == 'A'`` if and only if the contract unit is not round.

        Zero disagreement either way across all 11,372 exchange-traded
        contracts. Two unrelated Wind fields agreeing exactly is the kind of
        redundancy worth pinning: if it ever stops holding, one of them has
        changed meaning.
        """
        static = panel.option_static
        exchange_traded = static[static["S_INFO_EXCHANGE_NAME"] != "CFFEX"]
        adjusted = exchange_traded["ADJ_SIGN"] == "A"
        non_round = exchange_traded["S_INFO_COUNIT"] != 10000.0

        assert len(exchange_traded) == 11_372
        assert int(adjusted.sum()) == 1_812
        assert bool((adjusted == non_round).all())

        # And the field does not apply on CFFEX, which is not the same as
        # "unknown" -- all 6,012 index contracts plus the treasury ones.
        cffex = static[static["S_INFO_EXCHANGE_NAME"] == "CFFEX"]
        assert bool(cffex["ADJ_SIGN"].isna().all())


@needs_panel
class TestMeasuredConventions:
    def test_an_untraded_contract_does_not_look_traded(self, panel: WindPanel) -> None:
        """The defect that would otherwise fit the exchange's interpolation.

        97,884 of the panel's rows have zero volume and a non-zero close, so
        ``has_traded`` -- which asks whether the last price is positive --
        would admit every one of them. They are not stale carry-forwards
        either: only 2.7% match the previous day's settlement and the median
        absolute daily move is 3.25%, so CFFEX is marking untraded strikes
        theoretically.
        """
        chain = panel.chain("HO", A_DATE)
        quotes = [q for s in chain.slices for q in s.quotes]

        traded = [q for q in quotes if q.has_traded]
        untraded = [q for q in quotes if not q.has_traded]

        assert untraded, "this date has zero-volume contracts; expected some"
        assert traded, "and traded ones too"
        # An untraded contract still has a usable fallback price, from the
        # genuine prior session rather than from today's theoretical mark.
        for quote in untraded:
            assert quote.last == 0.0
            assert quote.price == quote.previous_settlement

    def test_the_price_is_the_settlement(self, panel: WindPanel) -> None:
        """Measured: parity residuals of 0.041% against the close's 0.234%.

        The settlement wins 12 of 12 slices across two dates, by 5.7x, because
        it is one exchange-wide reference struck on one convention and so is
        synchronous by construction.
        """
        chain = panel.chain("IO", A_DATE)
        assert chain.source == "wind:2024-12-25:settle"

        quotes = panel.option_quotes
        day = quotes[quotes["TRADE_DT"].astype(str) == "20241225"]
        lookup = day.set_index("S_INFO_WINDCODE")

        checked = 0
        for option_slice in chain.slices:
            for quote in option_slice.quotes:
                row = lookup.loc[quote.contract.raw]
                if float(row["S_DQ_VOLUME"]) > 0.0:
                    assert quote.last == pytest.approx(float(row["S_DQ_SETTLE"]))
                    checked += 1
        assert checked > 100

    def test_every_raw_row_is_either_kept_or_counted(self, panel: WindPanel) -> None:
        """The reconciliation identity, enforced rather than reported.

        A screen whose arithmetic does not close is indistinguishable from one
        that silently loses data, so :meth:`WindPanel.chain` raises instead of
        warning. This asserts the ledger is reachable and adds up.
        """
        for root in ("IO", "510050"):
            chain = panel.chain(root, A_DATE)
            ledger = panel.ledger(root, A_DATE)
            kept = sum(len(s) for s in chain.slices)
            assert kept + ledger.total_rejected == ledger.raw_rows

    def test_the_etf_path_pays_heavily_for_dropping_adjusted_contracts(
        self, panel: WindPanel
    ) -> None:
        """The measured argument for the CFFEX options being primary.

        On this date 140 of 510050's 220 rows are adjusted -- 64% of the chain
        -- against zero for IO. Stated as a test so the cost of the policy
        stays visible instead of being buried in a docstring.
        """
        panel.chain("510050", A_DATE)
        etf = panel.ledger("510050", A_DATE)
        assert etf.raw_rows == 220
        assert etf.adjusted_contract == 140

        panel.chain("IO", A_DATE)
        index = panel.ledger("IO", A_DATE)
        assert index.adjusted_contract == 0
        assert index.non_positive_price == 0
        assert index.expired == 0
        assert index.no_static_record == 0


@needs_panel
class TestCoverage:
    """How much of the panel is actually usable, measured once and pinned.

    The static contract file is a snapshot that stops before the quote panel
    does -- 79,078 rows across 1,704 contracts, all dated 2025-09-22 or later,
    have no static record. So "725 days" is not the denominator for anything,
    and averaging a six-expiry term structure with a three-expiry one would be
    exactly the kind of number this project otherwise refuses to publish.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def slice_counts(cls, panel: WindPanel) -> list[int]:
        counts = []
        for day in panel.trading_days:
            try:
                counts.append(len(panel.chain("IO", day).slices))
            except MarketDataError:
                counts.append(0)
        return counts

    def test_complete_chains_are_516_of_725_days(self, slice_counts: list[int]) -> None:
        assert len(slice_counts) == 725
        assert sum(1 for n in slice_counts if n == 6) == 516
        assert sum(1 for n in slice_counts if n >= 4) == 656
        # MIN_USABLE_SLICES is 2, so this is the population a surface can fit.
        assert sum(1 for n in slice_counts if n >= 2) == 694
        assert sum(1 for n in slice_counts if n == 0) == 6

    def test_the_unusable_days_are_the_panels_last_week(
        self, panel: WindPanel, slice_counts: list[int]
    ) -> None:
        """Not scattered, which is what makes them safe to exclude by date."""
        failed = [day for day, n in zip(panel.trading_days, slice_counts, strict=True) if n == 0]

        assert failed[0] == date(2026, 6, 22)
        assert failed[-1] == date(2026, 6, 29)
        assert len(failed) == 6


@needs_panel
class TestFuturesAndIndex:
    def test_the_future_shares_the_options_expiry_exactly(self, panel: WindPanel) -> None:
        """Which is what makes the forward check interpolation-free.

        Verified across all 36 expiry tags with zero exceptions: the index
        future's delist date equals the option's maturity date, so the futures
        settlement *is* the forward to the option's own expiry.
        """
        chain = panel.chain("IO", A_DATE)
        option_expiries = {s.expiry for s in chain.slices}
        futures_expiries = {e for e, _ in panel.futures("IF", A_DATE)}

        assert futures_expiries
        assert futures_expiries <= option_expiries

    def test_only_four_futures_list_against_six_option_expiries(self, panel: WindPanel) -> None:
        """So the two longest expiries are unchecked, and callers must say so.

        The longest is the one that matters most for a one-to-three year
        snowball, which makes this a limit on the check rather than a detail.
        """
        quotes = panel.futures("IF", A_DATE)
        assert len(quotes) == 4
        assert len(panel.chain("IO", A_DATE).slices) == 6

    def test_the_index_close_is_available_for_every_root(self, panel: WindPanel) -> None:
        for root in ("IO", "HO", "MO"):
            code = resolve_underlying(root).index_code
            assert code is not None
            assert panel.index_close(code, A_DATE) > 0.0
