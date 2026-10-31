"""Tests for the market data layer: contracts, quotes, sources, forwards, screening.

Nothing here touches the network. The one test that would is marked
``network`` and deselected by default, matching the convention the rest of the
suite uses for optional dependencies: a test whose result depends on a third
party's uptime does not tell you whether your code works.

The emphasis is on the failures that are *silent*. A feed that returns an error
is easy; a feed that returns 136 rows of a different payload shape under a
similar product name is what actually causes a wrong number to be published, so
that case gets a test of its own.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from otcpricer.core.exceptions import MarketDataError
from otcpricer.core.types import OptionType
from otcpricer.marketdata.contracts import (
    SUPPORTED_UNDERLYINGS,
    parse_contract_code,
    resolve_underlying,
)
from otcpricer.marketdata.forward import (
    MAX_PARITY_RESIDUAL_REL,
    MIN_PAIRED_STRIKES,
    ForwardCurve,
    imply_forward,
)
from otcpricer.marketdata.quotes import OptionChain, build_slice
from otcpricer.marketdata.snapshots import (
    SNAPSHOT_FORMAT_VERSION,
    list_snapshots,
    load_snapshot,
    save_snapshot,
)
from otcpricer.marketdata.sources import (
    AkshareSource,
    CachingSource,
    CsvSource,
    SnapshotSource,
    build_source,
)
from otcpricer.marketdata.surface_input import (
    MAX_MONEYNESS_SD,
    extract_surface_input,
)

#: The captured chains committed to the repo, used as regression fixtures.
SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshots"


@pytest.fixture
def snapshot_dir() -> Path:
    if not SNAPSHOT_DIR.exists():
        pytest.skip(f"no snapshots committed at {SNAPSHOT_DIR}")
    return SNAPSHOT_DIR


@pytest.fixture
def chain_510050(snapshot_dir: Path) -> OptionChain:
    return load_snapshot(snapshot_dir, "510050")


@pytest.fixture
def chain_510300(snapshot_dir: Path) -> OptionChain:
    return load_snapshot(snapshot_dir, "510300")


# ---------------------------------------------------------------------------
# Contract codes and the underlying registry
# ---------------------------------------------------------------------------


class TestContractCodes:
    def test_parses_a_standard_code(self) -> None:
        contract = parse_contract_code("510050C2611M02800")
        assert contract.underlying == "510050"
        assert contract.option_type is OptionType.CALL
        assert contract.expiry_year == 2026
        assert contract.expiry_month == 11
        assert contract.strike == pytest.approx(2.800)
        assert contract.expiry_tag == "2611"
        assert not contract.is_adjusted

    def test_parses_a_put(self) -> None:
        assert parse_contract_code("510300P2703M04000").option_type is OptionType.PUT

    def test_keeps_the_adjustment_marker(self) -> None:
        """An adjusted contract sits off the standard strike ladder.

        Worth surfacing rather than discarding: the marker is the only signal
        that this contract's terms were changed by a distribution.
        """
        contract = parse_contract_code("510050C2611M02800A")
        assert contract.adjustment == "A"
        assert contract.is_adjusted

    @pytest.mark.parametrize(
        "code",
        [
            "510050X2611M02800",  # neither C nor P
            "510050C26M02800",  # short expiry tag
            "510050C2611-02800",  # missing the M separator
            "50050C2611M02800",  # five-digit underlying
            "",
        ],
    )
    def test_rejects_malformed_codes(self, code: str) -> None:
        with pytest.raises(MarketDataError, match="cannot parse contract code"):
            parse_contract_code(code)

    def test_rejects_an_impossible_month(self) -> None:
        with pytest.raises(MarketDataError, match="invalid month"):
            parse_contract_code("510050C2613M02800")

    def test_strike_scale_comes_from_the_registry(self) -> None:
        """ETF codes carry the strike times 1000; getting this wrong is 1000x."""
        assert parse_contract_code("510050C2611M02800").strike == pytest.approx(2.8)
        assert parse_contract_code("510050C2611M02800", strike_scale=1.0).strike == 2800.0


class TestUnderlyingRegistry:
    def test_resolves_both_supported_underlyings(self) -> None:
        assert resolve_underlying("510050").code == "510050"
        assert resolve_underlying(" 510300 ").code == "510300"

    def test_rejects_an_unknown_ticker_and_says_what_is_available(self) -> None:
        with pytest.raises(MarketDataError, match="510050, 510300"):
            resolve_underlying("159915")

    def test_300etf_maps_to_huatai_not_jiashi(self) -> None:
        """The silent-failure guard.

        Asking the feed for the Jiashi product returns 136 rows whose strike
        column is uniformly 10000 -- a different payload shape, not an error.
        Anything built on it looks plausible and is nonsense. The mapping is
        pinned here so a well-meaning edit cannot quietly change it.
        """
        assert SUPPORTED_UNDERLYINGS["510300"].feed_symbol == "华泰柏瑞沪深300ETF期权"
        assert "嘉实" not in SUPPORTED_UNDERLYINGS["510300"].feed_symbol

    def test_50etf_feed_symbol_is_pinned(self) -> None:
        assert SUPPORTED_UNDERLYINGS["510050"].feed_symbol == "华夏上证50ETF期权"


# ---------------------------------------------------------------------------
# Slice assembly
# ---------------------------------------------------------------------------


def _row(code: str, last: float, strike: float, previous: float = 0.0) -> dict[str, object]:
    return {
        "code": code,
        "last": last,
        "previous_settlement": previous or last,
        "strike": strike,
    }


class TestBuildSlice:
    def test_cross_checks_the_strike_against_the_code(self) -> None:
        """Two independent readings of the same field must agree.

        The feed supplies a strike column and the contract code embeds the same
        strike. A mismatch means the upstream format moved, which is exactly
        the failure that otherwise passes silently.
        """
        rows = [_row("510050C2611M02800", 0.1, 2.900)]
        with pytest.raises(MarketDataError, match="strike disagreement"):
            build_slice(
                resolve_underlying("510050"),
                rows,
                expiry=date(2026, 11, 25),
                as_of=date(2026, 10, 27),
                days_to_expiry=29,
            )

    def test_year_fraction_uses_the_feeds_own_day_count(self) -> None:
        option_slice = build_slice(
            resolve_underlying("510050"),
            [_row("510050C2611M02800", 0.1, 2.800)],
            expiry=date(2026, 11, 25),
            as_of=date(2026, 10, 27),
            days_to_expiry=29,
        )
        assert option_slice.year_fraction == pytest.approx(29.0 / 365.0)

    def test_untraded_quote_falls_back_to_the_settlement(self) -> None:
        option_slice = build_slice(
            resolve_underlying("510050"),
            [_row("510050C2611M02800", 0.0, 2.800, previous=0.12)],
            expiry=date(2026, 11, 25),
            as_of=date(2026, 10, 27),
            days_to_expiry=29,
        )
        quote = option_slice.quotes[0]
        assert not quote.has_traded
        assert quote.price == pytest.approx(0.12)


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


class TestSnapshots:
    def test_committed_snapshots_load(self, chain_510050: OptionChain) -> None:
        assert chain_510050.underlying.code == "510050"
        assert len(chain_510050.slices) >= 2
        assert chain_510050.as_of == date(2026, 10, 27)

    def test_round_trip_preserves_every_quote(
        self, chain_510050: OptionChain, tmp_path: Path
    ) -> None:
        save_snapshot(chain_510050, tmp_path, name="roundtrip")
        reloaded = load_snapshot(tmp_path, "510050", name="roundtrip")

        assert len(reloaded.slices) == len(chain_510050.slices)
        for before, after in zip(chain_510050.slices, reloaded.slices, strict=True):
            assert after.expiry == before.expiry
            assert after.days_to_expiry == before.days_to_expiry
            assert [q.contract.raw for q in after.quotes] == [q.contract.raw for q in before.quotes]
            assert [q.last for q in after.quotes] == [q.last for q in before.quotes]
            assert [q.strike for q in after.quotes] == [q.strike for q in before.quotes]

    def test_refuses_a_foreign_format_version(
        self, chain_510050: OptionChain, tmp_path: Path
    ) -> None:
        """A misread snapshot is worse than a missing one."""
        path = save_snapshot(chain_510050, tmp_path, name="future")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["format_version"] = SNAPSHOT_FORMAT_VERSION + 1
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(MarketDataError, match="format version"):
            load_snapshot(tmp_path, "510050", name="future")

    def test_lists_newest_first(self, chain_510050: OptionChain, tmp_path: Path) -> None:
        for name in ("2026-10-20", "2026-10-27", "2026-10-23"):
            save_snapshot(chain_510050, tmp_path, name=name)
        assert list_snapshots(tmp_path, "510050") == ["2026-10-27", "2026-10-23", "2026-10-20"]

    def test_missing_snapshot_explains_how_to_get_one(self, tmp_path: Path) -> None:
        with pytest.raises(MarketDataError, match="capture_snapshot"):
            load_snapshot(tmp_path, "510050")


# ---------------------------------------------------------------------------
# Sources: resilience, caching, offline default
# ---------------------------------------------------------------------------


class _FlakySource:
    """A source that fails a set number of times before succeeding."""

    def __init__(self, chain: OptionChain, failures: int) -> None:
        self.chain = chain
        self.failures = failures
        self.calls = 0

    @property
    def name(self) -> str:
        return "flaky"

    def fetch(self, underlying_code: str) -> OptionChain:
        self.calls += 1
        if self.calls <= self.failures:
            raise ConnectionError("RemoteDisconnected")
        return self.chain


class TestSourceResilience:
    def test_retry_eventually_succeeds_and_backs_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        source = AkshareSource(max_attempts=4, backoff_seconds=2.0)
        sleeps: list[float] = []
        monkeypatch.setattr("otcpricer.marketdata.sources.time.sleep", sleeps.append)

        attempts = {"n": 0}

        def flaky() -> str:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ConnectionError("RemoteDisconnected")
            return "ok"

        assert source._retry("a board", flaky) == "ok"
        assert attempts["n"] == 3
        # Doubling, and no sleep after the successful attempt.
        assert sleeps == [2.0, 4.0]

    def test_gives_up_with_a_message_naming_what_it_wanted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bare transport error does not say which request died."""
        source = AkshareSource(max_attempts=3, backoff_seconds=0.0)
        monkeypatch.setattr("otcpricer.marketdata.sources.time.sleep", lambda _: None)

        def always_fails() -> None:
            raise ConnectionError("RemoteDisconnected")

        with pytest.raises(MarketDataError) as excinfo:
            source._retry("board for 510050 2611", always_fails)

        message = str(excinfo.value)
        assert "board for 510050 2611" in message
        assert "3 attempts" in message
        assert "ConnectionError" in message

    def test_retry_count_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        source = AkshareSource(max_attempts=5, backoff_seconds=0.0)
        monkeypatch.setattr("otcpricer.marketdata.sources.time.sleep", lambda _: None)
        calls = {"n": 0}

        def always_fails() -> None:
            calls["n"] += 1
            raise OSError("boom")

        with pytest.raises(MarketDataError):
            source._retry("x", always_fails)
        assert calls["n"] == 5

    def test_unsupported_underlying_is_refused_before_any_request(self) -> None:
        """The registry stops a bad symbol reaching the feed at all."""
        with pytest.raises(MarketDataError, match="not supported"):
            AkshareSource().fetch("159915")


class TestCaching:
    def test_second_fetch_is_served_from_cache(self, chain_510050: OptionChain) -> None:
        inner = _FlakySource(chain_510050, failures=0)
        cached = CachingSource(inner=inner, ttl_seconds=300.0)

        first = cached.fetch("510050")
        second = cached.fetch("510050")

        assert inner.calls == 1
        assert second is first

    def test_cache_expires(self, chain_510050: OptionChain) -> None:
        inner = _FlakySource(chain_510050, failures=0)
        cached = CachingSource(inner=inner, ttl_seconds=0.0)

        cached.fetch("510050")
        cached.fetch("510050")

        assert inner.calls == 2

    def test_invalidate_forces_a_refetch(self, chain_510050: OptionChain) -> None:
        inner = _FlakySource(chain_510050, failures=0)
        cached = CachingSource(inner=inner, ttl_seconds=300.0)

        cached.fetch("510050")
        cached.invalidate()
        cached.fetch("510050")

        assert inner.calls == 2

    def test_different_underlyings_do_not_share_an_entry(self, chain_510050: OptionChain) -> None:
        inner = _FlakySource(chain_510050, failures=0)
        cached = CachingSource(inner=inner, ttl_seconds=300.0)

        cached.fetch("510050")
        cached.fetch("510300")

        assert inner.calls == 2


class TestSourceSelection:
    def test_default_is_offline(self, snapshot_dir: Path) -> None:
        """Configuration ships with source='snapshot', so nothing reaches out."""
        source = build_source(snapshot_dir=snapshot_dir)
        assert "snapshot" in source.name
        assert source.fetch("510050").underlying.code == "510050"

    def test_rejects_an_unknown_kind(self) -> None:
        with pytest.raises(MarketDataError, match="unknown market data source"):
            build_source("bloomberg")

    def test_csv_requires_a_path(self) -> None:
        with pytest.raises(MarketDataError, match="requires csv_path"):
            build_source("csv")

    def test_csv_source_reads_a_chain(self, tmp_path: Path) -> None:
        path = tmp_path / "chain.csv"
        path.write_text(
            "code,last,previous_settlement,strike,expiry,days_to_expiry\n"
            "510050C2611M02800,0.1000,0.1010,2.800,2026-11-25,29\n"
            "510050P2611M02800,0.0500,0.0510,2.800,2026-11-25,29\n",
            encoding="utf-8",
        )
        chain = CsvSource(path=path).fetch("510050")
        assert len(chain.slices) == 1
        assert len(chain.slices[0].quotes) == 2
        assert chain.source == "csv:chain.csv"

    def test_missing_csv_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(MarketDataError, match="CSV not found"):
            CsvSource(path=tmp_path / "absent.csv").fetch("510050")

    def test_snapshot_source_lists_what_it_has(self, snapshot_dir: Path) -> None:
        source = SnapshotSource(directory=snapshot_dir)
        assert source.available("510050")


@pytest.mark.network
def test_live_feed_round_trip() -> None:
    """The only test that touches the network; deselected by default.

    Run with ``pytest -m network`` when checking that the upstream format has
    not moved.
    """
    chain = AkshareSource(max_expiries=1).fetch("510050")
    assert len(chain.slices) == 1
    assert len(chain.slices[0].paired_strikes()) >= MIN_PAIRED_STRIKES


# ---------------------------------------------------------------------------
# Forward implication
# ---------------------------------------------------------------------------


def _parity_slice(
    forward: float,
    rate: float,
    year_fraction: float,
    strikes: list[float],
    *,
    noise: np.ndarray | None = None,
) -> object:
    """Build a slice whose quotes satisfy parity exactly, up to optional noise.

    Only the call-minus-put spread matters to the forward, so the absolute
    level is arbitrary; the put is placed at a fixed level and the call set to
    whatever parity requires.
    """
    discount = float(np.exp(-rate * year_fraction))
    rows: list[dict[str, object]] = []
    for index, strike in enumerate(strikes):
        put = 0.5
        call = put + (forward - strike) * discount
        if noise is not None:
            call += float(noise[index])
        code_strike = f"{int(round(strike * 1000)):05d}"
        rows.append(_row(f"510050C2611M{code_strike}", call, strike))
        rows.append(_row(f"510050P2611M{code_strike}", put, strike))

    return build_slice(
        resolve_underlying("510050"),
        rows,
        expiry=date(2026, 11, 25),
        as_of=date(2026, 10, 27),
        days_to_expiry=int(round(year_fraction * 365)),
    )


class TestForwardImplication:
    def test_recovers_an_exact_forward(self) -> None:
        """Clean parity data must give back the forward to machine precision."""
        option_slice = _parity_slice(2.85, 0.02, 29 / 365, [2.6, 2.7, 2.8, 2.9, 3.0, 3.1])
        result = imply_forward(option_slice, rate=0.02)  # type: ignore[arg-type]

        assert result.usable
        assert result.forward == pytest.approx(2.85, abs=1e-12)
        assert result.parity_residual_rms == pytest.approx(0.0, abs=1e-12)
        assert result.n_strikes == 6

    def test_forward_is_insensitive_to_the_assumed_rate(self) -> None:
        """The measurement that justifies taking the rate as an input.

        A 100bp change in the assumed rate moves the implied forward by
        thousandths of a percent, which is why fitting the rate as a free
        parameter fits noise rather than information.
        """
        option_slice = _parity_slice(2.85, 0.02, 29 / 365, [2.6, 2.7, 2.8, 2.9, 3.0, 3.1])
        low = imply_forward(option_slice, rate=0.015).forward  # type: ignore[arg-type]
        high = imply_forward(option_slice, rate=0.025).forward  # type: ignore[arg-type]

        assert abs(high - low) / low < 1e-3

    def test_averaging_dilutes_one_stale_quote(self) -> None:
        """A single bad quote must move the estimate by 1/n of its error."""
        strikes = [2.6, 2.7, 2.8, 2.9, 3.0, 3.1]
        error = 0.06
        noise = np.zeros(len(strikes))
        noise[2] = error

        clean = imply_forward(
            _parity_slice(2.85, 0.02, 29 / 365, strikes),  # type: ignore[arg-type]
            rate=0.02,
        )
        dirty = imply_forward(
            _parity_slice(2.85, 0.02, 29 / 365, strikes, noise=noise),  # type: ignore[arg-type]
            rate=0.02,
        )

        discount = float(np.exp(-0.02 * 29 / 365))
        assert dirty.forward - clean.forward == pytest.approx(error / (len(strikes) * discount))

    def test_rejects_a_slice_with_too_few_traded_strikes(self) -> None:
        option_slice = _parity_slice(2.85, 0.02, 29 / 365, [2.8, 2.9, 3.0])
        result = imply_forward(option_slice, rate=0.02)  # type: ignore[arg-type]

        assert not result.usable
        assert "need 4" in result.reason

    def test_rejects_a_slice_whose_quotes_disagree(self) -> None:
        """Beyond the residual limit the quotes carry no volatility information."""
        strikes = [2.6, 2.7, 2.8, 2.9, 3.0, 3.1]
        noise = np.array([0.2, -0.2, 0.2, -0.2, 0.2, -0.2])
        result = imply_forward(
            _parity_slice(2.85, 0.02, 29 / 365, strikes, noise=noise),  # type: ignore[arg-type]
            rate=0.02,
        )

        assert not result.usable
        assert "parity residual" in result.reason
        assert result.residual_relative > MAX_PARITY_RESIDUAL_REL

    def test_expired_slice_is_refused(self) -> None:
        option_slice = build_slice(
            resolve_underlying("510050"),
            [_row("510050C2611M02800", 0.1, 2.8), _row("510050P2611M02800", 0.05, 2.8)],
            expiry=date(2026, 10, 27),
            as_of=date(2026, 10, 27),
            days_to_expiry=0,
        )
        result = imply_forward(option_slice, rate=0.02)
        assert not result.usable
        assert "0 days" in result.reason

    def test_unpaired_slice_is_structurally_wrong_and_raises(self) -> None:
        """A poor slice comes back unusable; an impossible one raises."""
        option_slice = build_slice(
            resolve_underlying("510050"),
            [_row("510050C2611M02800", 0.1, 2.8), _row("510050C2611M02900", 0.05, 2.9)],
            expiry=date(2026, 11, 25),
            as_of=date(2026, 10, 27),
            days_to_expiry=29,
        )
        with pytest.raises(MarketDataError, match="no strike quoted as both"):
            imply_forward(option_slice, rate=0.02)

    def test_implied_carry_refuses_rather_than_guessing(self) -> None:
        option_slice = _parity_slice(2.85, 0.02, 29 / 365, [2.6, 2.7, 2.8, 2.9])
        result = imply_forward(option_slice, rate=0.02)  # type: ignore[arg-type]
        with pytest.raises(NotImplementedError, match="needs a spot price"):
            _ = result.implied_carry

    @pytest.mark.parametrize("code", ["510050", "510300"])
    def test_real_slices_imply_plausible_forwards(self, snapshot_dir: Path, code: str) -> None:
        """Forwards from the captured chains must bracket sensibly.

        Both ETFs were in mild backwardation on the capture date, so a later
        expiry should not imply a materially higher forward. This is the
        sanity check that would catch a sign error in the discounting.
        """
        chain = load_snapshot(snapshot_dir, code)
        forwards = [imply_forward(s, rate=0.02) for s in chain.slices if s.year_fraction > 1 / 365]
        usable = [f for f in forwards if f.usable]
        assert len(usable) >= 2

        for result in usable:
            assert result.forward > 0.0
            assert result.residual_relative <= MAX_PARITY_RESIDUAL_REL
        # Monotone decreasing, within a tolerance that admits quote noise.
        levels = [f.forward for f in usable]
        assert levels[0] >= levels[-1] - 1e-3


# ---------------------------------------------------------------------------
# Implied volatility extraction and quote screening
# ---------------------------------------------------------------------------


class TestSurfaceInput:
    @pytest.mark.parametrize("code", ["510050", "510300"])
    def test_real_chains_yield_usable_slices(self, snapshot_dir: Path, code: str) -> None:
        surface_input = extract_surface_input(load_snapshot(snapshot_dir, code))
        usable = surface_input.usable_slices

        assert len(usable) >= 2
        for slice_input in usable:
            assert len(slice_input.points) >= 4
            # Implied vols in a range an ETF can actually print.
            for point in slice_input.points:
                assert 0.01 < point.implied_vol < 1.5

    @pytest.mark.parametrize("code", ["510050", "510300"])
    def test_every_quote_is_either_kept_or_counted(self, snapshot_dir: Path, code: str) -> None:
        """The reconciliation identity, per expiry.

        A pipeline that discards quotes without saying so is not auditable, and
        a rejection *summary* that does not add up is worse than none: it reads
        as evidence while hiding a gap. So every quote in a slice with a usable
        forward must be either kept or booked against exactly one reason.
        """
        chain = load_snapshot(snapshot_dir, code)
        surface_input = extract_surface_input(chain)

        for option_slice, slice_input in zip(chain.slices, surface_input.slices, strict=True):
            assert slice_input.expiry_tag == option_slice.expiry_tag
            if not slice_input.forward.usable:
                # Screening never ran, so there is nothing to reconcile.
                assert slice_input.reason.startswith("forward unusable")
                continue
            booked = sum(slice_input.rejections.values()) + len(slice_input.points)
            assert booked == len(option_slice.quotes)

    @pytest.mark.parametrize("code", ["510050", "510300"])
    def test_pooled_rejections_reconcile_and_are_labelled(
        self, snapshot_dir: Path, code: str
    ) -> None:
        surface_input = extract_surface_input(load_snapshot(snapshot_dir, code))
        payload = surface_input.as_dict()

        assert payload["n_rejected"] == sum(surface_input.rejections.values())
        assert surface_input.n_rejected > 0, "the real chains do contain unusable quotes"
        for reason, count in surface_input.rejections.items():
            assert reason and count > 0

        for slice_payload in payload["slices"]:  # type: ignore[union-attr]
            if not slice_payload["usable"]:
                assert slice_payload["reason"]

    def test_moneyness_cap_removes_the_far_wing(self, snapshot_dir: Path) -> None:
        """The cap was added because the far wing bent the global parameters.

        Relaxing it must let more points through; keeping it must hold every
        surviving point inside the stated number of standard deviations.
        """
        chain = load_snapshot(snapshot_dir, "510300")
        capped = extract_surface_input(chain)
        uncapped = extract_surface_input(chain, max_moneyness_sd=float("inf"))

        n_capped = sum(len(s.points) for s in capped.usable_slices)
        n_uncapped = sum(len(s.points) for s in uncapped.usable_slices)
        assert n_uncapped > n_capped

        for slice_input in capped.usable_slices:
            variance = np.array([p.total_variance for p in slice_input.points])
            moneyness = np.array([p.log_moneyness for p in slice_input.points])
            sigma_sqrt_t = float(np.sqrt(float(np.interp(0.0, moneyness, variance))))
            assert np.all(np.abs(moneyness) <= MAX_MONEYNESS_SD * sigma_sqrt_t + 1e-12)

    def test_slices_are_ordered_by_moneyness(self, snapshot_dir: Path) -> None:
        for slice_input in extract_surface_input(
            load_snapshot(snapshot_dir, "510050")
        ).usable_slices:
            moneyness = [p.log_moneyness for p in slice_input.points]
            assert moneyness == sorted(moneyness)

    def test_total_variance_is_consistent_with_the_implied_vol(self, snapshot_dir: Path) -> None:
        """A derived field that disagrees with its source is a silent trap."""
        for slice_input in extract_surface_input(
            load_snapshot(snapshot_dir, "510050")
        ).usable_slices:
            for point in slice_input.points:
                expected = point.implied_vol**2 * point.year_fraction
                assert point.total_variance == pytest.approx(expected, rel=1e-12)

    def test_unusable_forward_disqualifies_the_whole_slice(self) -> None:
        """No forward, no moneyness coordinate, so no slice."""
        option_slice = _parity_slice(2.85, 0.02, 29 / 365, [2.8, 2.9, 3.0])
        chain = OptionChain(
            underlying=resolve_underlying("510050"),
            as_of=date(2026, 10, 27),
            slices=(option_slice,),  # type: ignore[arg-type]
            source="synthetic",
        )
        surface_input = extract_surface_input(chain)

        assert not surface_input.usable_slices
        assert "forward unusable" in surface_input.slices[0].reason


# ---------------------------------------------------------------------------
# The forward term structure
# ---------------------------------------------------------------------------


def _real_curve(snapshot_dir: Path, code: str, tail_drift: float = 0.02) -> ForwardCurve:
    """Build a curve from a committed chain's parity-implied forwards."""
    surface_input = extract_surface_input(load_snapshot(snapshot_dir, code))
    usable = surface_input.usable_slices
    return ForwardCurve(
        maturities=tuple(s.year_fraction for s in usable),
        forwards=tuple(s.forward.forward for s in usable),
        tail_drift=tail_drift,
        rate=0.03,
    )


class TestForwardCurve:
    """The curve the local volatility has to agree with.

    A surface is parameterised in ``log(K / F)``, so if the engine's forward
    curve differs from the one the surface was fitted against, the local
    volatility is evaluated at the wrong moneyness and nothing says so. These
    tests pin the two properties that make that impossible: the curve
    reproduces the quoted forwards exactly, and its drift integrates to its own
    log-forward.
    """

    CURVE = ForwardCurve(
        maturities=(0.25, 0.5, 1.0),
        forwards=(99.0, 98.0, 95.0),
        tail_drift=0.02,
        rate=0.03,
    )

    def test_reproduces_the_quoted_forwards_exactly(self) -> None:
        """Anchored at its own spot, the curve must hit every quoted forward.

        This is what makes the vanilla reprice test a test of the local
        volatility rather than of a spot mismatch.
        """
        for maturity, forward in zip(self.CURVE.maturities, self.CURVE.forwards, strict=True):
            assert float(self.CURVE.forward_from(self.CURVE.spot_anchor, maturity)) == (
                pytest.approx(forward, rel=1e-14)
            )

    def test_drift_integrates_to_the_log_forward(self) -> None:
        """The off-by-one guard.

        ``searchsorted`` returns a knot index while the drift lives on segments,
        so the segment index is one lower. Getting that wrong returns the next
        segment's drift, and the error shows up as a *constant* offset between
        the integrated drift and the log-forward -- measured at 1.62e-03 on the
        real 510050 curve before it was fixed. Inspection did not catch it; this
        comparison did.
        """
        for t_end in (0.1, 0.25, 0.4, 0.5, 1.0, 2.5):
            u = np.linspace(0.0, t_end, 400001)
            quadrature = float(np.trapezoid(self.CURVE.drift(u), u))
            analytic = float(self.CURVE.log_forward_ratio(t_end))
            # The residual is the trapezoid rule's own error on a discontinuous
            # integrand, not the curve's.
            assert analytic == pytest.approx(quadrature, abs=1e-6)

    def test_ratio_vanishes_at_the_origin(self) -> None:
        assert float(self.CURVE.log_forward_ratio(0.0)) == 0.0

    def test_drift_is_the_enclosing_segments_slope(self) -> None:
        """Explicitly, segment by segment, rather than via the integral."""
        T = np.array(self.CURVE.maturities)
        log_F = np.log(np.array(self.CURVE.forwards))
        expected = np.diff(log_F) / np.diff(T)

        # Inside the first segment, and before the first expiry.
        assert float(self.CURVE.drift(0.3)) == pytest.approx(expected[0])
        assert float(self.CURVE.drift(0.1)) == pytest.approx(expected[0])
        # Inside the second.
        assert float(self.CURVE.drift(0.75)) == pytest.approx(expected[1])
        # Left-continuous at a knot: a drift defined on (T_i, T_{i+1}].
        assert float(self.CURVE.drift(0.5)) == pytest.approx(expected[0])

    def test_tail_reverts_to_the_callers_drift(self) -> None:
        assert float(self.CURVE.drift(1.5)) == pytest.approx(0.02)
        assert float(self.CURVE.drift(10.0)) == pytest.approx(0.02)

    def test_tail_is_linear_in_log_forward(self) -> None:
        last = self.CURVE.maturities[-1]
        base = float(self.CURVE.log_forward_ratio(last))
        for extra in (0.5, 1.0, 2.0):
            assert float(self.CURVE.log_forward_ratio(last + extra)) == pytest.approx(
                base + 0.02 * extra, rel=1e-12
            )

    def test_kink_is_the_drift_jump_at_the_last_expiry(self) -> None:
        """The known price of reverting to the caller's carry; reported, not hidden."""
        implied_last = (np.log(95.0) - np.log(98.0)) / (1.0 - 0.5)
        assert self.CURVE.kink == pytest.approx(0.02 - implied_last)
        assert float(self.CURVE.drift(1.0 + 1e-9)) - float(self.CURVE.drift(1.0)) == (
            pytest.approx(self.CURVE.kink, rel=1e-6)
        )

    def test_is_scale_free(self) -> None:
        """The shape is stored, not the level.

        So a surface fitted on an ETF at 2.85 can price a note quoted on an
        index at 5000 -- moneyness only cares about ratios.
        """
        for t in (0.1, 0.6, 2.0):
            small = float(self.CURVE.forward_from(2.8662, t)) / 2.8662
            large = float(self.CURVE.forward_from(5000.0, t)) / 5000.0
            assert small == pytest.approx(large, rel=1e-14)

    def test_carry_is_rate_minus_drift(self) -> None:
        for t in (0.1, 0.6, 2.0):
            assert float(self.CURVE.carry(t)) == pytest.approx(0.03 - float(self.CURVE.drift(t)))

    def test_vectorises(self) -> None:
        t = np.array([0.05, 0.3, 0.75, 1.5])
        assert self.CURVE.drift(t).shape == (4,)
        assert self.CURVE.log_forward_ratio(t).shape == (4,)
        for index, value in enumerate(t):
            assert float(self.CURVE.drift(t)[index]) == pytest.approx(
                float(self.CURVE.drift(value))
            )

    def test_single_expiry_uses_the_tail_drift_throughout(self) -> None:
        """A legitimate degenerate case: no segment exists to take a slope from."""
        curve = ForwardCurve(maturities=(0.5,), forwards=(2.8,), tail_drift=-0.04, rate=0.03)
        assert float(curve.drift(0.2)) == pytest.approx(-0.04)
        assert float(curve.drift(2.0)) == pytest.approx(-0.04)
        assert curve.spot_anchor == pytest.approx(2.8 * np.exp(0.04 * 0.5))
        assert curve.kink == pytest.approx(0.0)

    @pytest.mark.parametrize(
        ("kwargs", "fragment"),
        [
            ({"maturities": (0.5,), "forwards": (1.0, 2.0)}, "1 maturities but 2 forwards"),
            ({"maturities": (), "forwards": ()}, "at least one quoted expiry"),
            ({"maturities": (0.0,), "forwards": (1.0,)}, "maturities must be positive"),
            ({"maturities": (1.0, 0.5), "forwards": (1.0, 2.0)}, "strictly increasing"),
            ({"maturities": (0.5,), "forwards": (0.0,)}, "forwards must be positive"),
        ],
    )
    def test_rejects_malformed_input(self, kwargs: dict[str, object], fragment: str) -> None:
        with pytest.raises(MarketDataError, match=fragment):
            ForwardCurve(tail_drift=0.0, **kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("code", ["510050", "510300"])
    def test_real_curves_reproduce_their_forwards(self, snapshot_dir: Path, code: str) -> None:
        curve = _real_curve(snapshot_dir, code)
        for maturity, forward in zip(curve.maturities, curve.forwards, strict=True):
            assert float(curve.forward_from(curve.spot_anchor, maturity)) == pytest.approx(
                forward, rel=1e-13
            )

    @pytest.mark.parametrize(
        ("code", "carry", "kink"), [("510050", 0.0416, 0.0536), ("510300", 0.0592, 0.0882)]
    )
    def test_real_implied_carry_and_kink_are_large(
        self, snapshot_dir: Path, code: str, carry: float, kink: float
    ) -> None:
        """Pinned because the magnitude is the whole argument for this design.

        The market's forward curve declines at 4-6% a year while a typical
        ``r - q`` of +2% rises. Using the latter would misstate the one-year
        forward by 6-8%, which against barriers at 103% and 80% swamps
        everything else. The kink is the acknowledged cost of reverting to the
        caller's carry past the quoted range.
        """
        curve = _real_curve(snapshot_dir, code, tail_drift=0.02)
        assert curve.implied_carry == pytest.approx(carry, abs=5e-4)
        assert curve.kink == pytest.approx(kink, abs=5e-4)

    @pytest.mark.parametrize("code", ["510050", "510300"])
    def test_real_curves_integrate_consistently(self, snapshot_dir: Path, code: str) -> None:
        curve = _real_curve(snapshot_dir, code)
        for t_end in (0.05, 0.2, curve.maturities[-1], 1.0, 3.0):
            u = np.linspace(0.0, float(t_end), 400001)
            assert float(curve.log_forward_ratio(t_end)) == pytest.approx(
                float(np.trapezoid(curve.drift(u), u)), abs=1e-6
            )

    def test_serialises_with_the_kink_visible(self, snapshot_dir: Path) -> None:
        payload = _real_curve(snapshot_dir, "510050").as_dict()
        assert payload["kink"] != 0.0
        for key in ("maturities", "forwards", "segment_drifts", "implied_carry", "spot_anchor"):
            assert key in payload
