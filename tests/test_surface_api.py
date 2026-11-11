"""Tests for the volatility surface endpoints.

Offline throughout: the configured default source is ``snapshot``, so these
exercise the committed chains and never reach a feed.

Two things get equal weight. The endpoints have to answer correctly, and they
have to *refuse* correctly -- a pricing API that returns 200 with a plausible
number for a malformed request is worse than one that errors, because the
number gets used.

The existing routes are checked here too. Phase 2a was not supposed to touch
them, and "was not supposed to" is not evidence.
"""

from __future__ import annotations

from typing import Any

import pytest

from otcpricer.marketdata.contracts import SUPPORTED_UNDERLYINGS
from tests.test_marketdata import SNAPSHOT_DIR


@pytest.fixture(autouse=True)
def _require_snapshots() -> None:
    if not SNAPSHOT_DIR.exists():
        pytest.skip("no snapshots committed")


@pytest.fixture
def client() -> Any:
    import app as app_module

    app_module.app.config["TESTING"] = True
    # A module-level fit cache would otherwise leak between tests.
    app_module._SURFACE_CACHE.clear()
    return app_module.app.test_client()


# ---------------------------------------------------------------------------
# /api/vol_surface/snapshots
# ---------------------------------------------------------------------------


class TestSnapshotsEndpoint:
    def test_lists_every_underlying_with_its_feed_symbol(self, client: Any) -> None:
        response = client.get("/api/vol_surface/snapshots")
        assert response.status_code == 200
        payload = response.get_json()

        assert set(payload["underlyings"]) == set(SUPPORTED_UNDERLYINGS)
        for code, underlying in SUPPORTED_UNDERLYINGS.items():
            assert payload["underlyings"][code]["feed_symbol"] == underlying.feed_symbol

    def test_the_captured_underlyings_list_their_snapshots(self, client: Any) -> None:
        """Scoped to the underlyings a live board actually serves.

        The CFFEX index options arrive only from the Wind panel, so they carry
        no ``feed_symbol`` and there is nothing to capture a snapshot of. That
        is the registry's own distinction, so the test reads it from there
        rather than naming the two ETF codes and going stale the next time an
        underlying is added.
        """
        payload = client.get("/api/vol_surface/snapshots").get_json()

        captured = [c for c, u in SUPPORTED_UNDERLYINGS.items() if u.feed_symbol is not None]
        assert captured, "no underlying has a live feed, which cannot be right"
        for code in captured:
            assert "2026-10-27" in payload["underlyings"][code]["snapshots"]

    def test_reports_that_the_default_source_is_offline(self, client: Any) -> None:
        """Worth stating in the response: it is what the other endpoints read."""
        payload = client.get("/api/vol_surface/snapshots").get_json()
        assert payload["default_source"] == "snapshot"

    def test_accepts_post_as_well(self, client: Any) -> None:
        assert client.post("/api/vol_surface/snapshots").status_code == 200


# ---------------------------------------------------------------------------
# /api/vol_surface/fit
# ---------------------------------------------------------------------------


class TestFitEndpoint:
    @pytest.mark.parametrize("code", ["510050", "510300"])
    def test_fits_both_underlyings(self, client: Any, code: str) -> None:
        response = client.post("/api/vol_surface/fit", json={"underlying": code})
        assert response.status_code == 200
        payload = response.get_json()

        assert payload["underlying"] == code
        assert payload["as_of"] == "2026-10-27"
        assert payload["calibration"]["converged"] is True
        assert payload["calibration"]["rms_vol_error"] < 0.01
        assert payload["arbitrage"]["arbitrage_free"] is True
        assert payload["local_volatility_ready"] is True

    def test_defaults_to_50etf(self, client: Any) -> None:
        payload = client.post("/api/vol_surface/fit", json={}).get_json()
        assert payload["underlying"] == "510050"

    def test_returns_the_evidence_not_only_the_parameters(self, client: Any) -> None:
        """The whole point of the response shape.

        A surface fitted to non-synchronous last-trade quotes with no bid/ask
        is an estimate. Parameters alone would let a caller treat it as a fact.
        """
        payload = client.post("/api/vol_surface/fit", json={"underlying": "510300"}).get_json()

        assert payload["calibration"]["slices"], "per-expiry fit quality"
        assert payload["input"]["rejections"], "which quotes were discarded and why"
        assert payload["input"]["n_rejected"] > 0
        assert payload["forwards"], "the implied forward per expiry"
        assert "min_butterfly" in payload["arbitrage"]
        assert "min_calendar_relative" in payload["arbitrage"]
        assert payload["local_volatility_note"]
        for slice_payload in payload["input"]["slices"]:
            assert "parity_residual_relative" in slice_payload or "forward" in slice_payload

    def test_report_is_optional_and_complete_when_asked(self, client: Any) -> None:
        plain = client.post("/api/vol_surface/fit", json={"underlying": "510050"}).get_json()
        assert "report" not in plain

        withreport = client.post(
            "/api/vol_surface/fit", json={"underlying": "510050", "report": True}
        ).get_json()
        assert "CALIBRATION ERROR" in withreport["report"]
        assert "STATIC ARBITRAGE CHECK" in withreport["report"]

    def test_rate_is_echoed_and_barely_moves_the_forward(self, client: Any) -> None:
        """The measurement that justifies taking the rate as an input."""
        low = client.post(
            "/api/vol_surface/fit", json={"underlying": "510050", "r": 0.015}
        ).get_json()
        high = client.post(
            "/api/vol_surface/fit", json={"underlying": "510050", "r": 0.025}
        ).get_json()

        assert low["input"]["rate"] == 0.015
        assert high["input"]["rate"] == 0.025
        for tag, forward in low["forwards"].items():
            assert abs(high["forwards"][tag] - forward) / forward < 1e-3

    def test_rejects_an_unsupported_underlying(self, client: Any) -> None:
        response = client.post("/api/vol_surface/fit", json={"underlying": "159915"})
        assert response.status_code == 400
        assert "not supported" in response.get_json()["error"]
        assert "510050" in response.get_json()["error"]

    def test_rejects_an_unknown_source(self, client: Any) -> None:
        response = client.post(
            "/api/vol_surface/fit", json={"underlying": "510050", "source": "bloomberg"}
        )
        assert response.status_code == 400
        assert "unknown market data source" in response.get_json()["error"]

    def test_missing_snapshot_is_a_400_not_a_500(self, client: Any) -> None:
        response = client.post(
            "/api/vol_surface/fit", json={"underlying": "510050", "snapshot": "1999-01-01"}
        )
        assert response.status_code == 400
        assert "snapshot" in response.get_json()["error"].lower()

    def test_caches_the_fit(self, client: Any) -> None:
        """Calibration is the expensive part and the query endpoint repeats it."""
        import app as app_module

        assert not app_module._SURFACE_CACHE
        client.post("/api/vol_surface/fit", json={"underlying": "510050"})
        assert len(app_module._SURFACE_CACHE) == 1
        client.post("/api/vol_surface/fit", json={"underlying": "510050"})
        assert len(app_module._SURFACE_CACHE) == 1

    def test_cache_does_not_serve_a_different_rate(self, client: Any) -> None:
        """A cache keyed too loosely is a wrong-answer machine."""
        import app as app_module

        client.post("/api/vol_surface/fit", json={"underlying": "510050", "r": 0.02})
        client.post("/api/vol_surface/fit", json={"underlying": "510050", "r": 0.04})
        assert len(app_module._SURFACE_CACHE) == 2


# ---------------------------------------------------------------------------
# /api/vol_surface/query
# ---------------------------------------------------------------------------


class TestQueryEndpoint:
    def test_auto_grid_has_consistent_shape(self, client: Any) -> None:
        payload = client.post(
            "/api/vol_surface/query",
            json={"underlying": "510050", "n_moneyness": 7, "n_maturities": 5},
        ).get_json()

        assert len(payload["moneyness"]) == 7
        assert len(payload["strikes"]) == 7
        assert len(payload["maturities"]) == 5
        for key in ("implied_vol", "total_variance", "butterfly", "dw_dT"):
            assert len(payload[key]) == 5
            assert all(len(row) == 7 for row in payload[key])

    def test_auto_grid_reaches_three_years(self, client: Any) -> None:
        """The surface has to be viewable where the products live."""
        payload = client.post("/api/vol_surface/query", json={"underlying": "510050"}).get_json()
        assert payload["maturities"][-1] == pytest.approx(3.0)
        assert max(payload["quoted_maturities"]) < 1.0

    def test_auto_grid_skips_the_meaningless_near_rows(self, client: Any) -> None:
        """The one-day expiry reaches 18 standard deviations at this moneyness.

        SSVI returns a number there, but it is not information, and on a heat
        map it takes over the colour scale. The default grid starts at a week.
        """
        payload = client.post("/api/vol_surface/query", json={"underlying": "510050"}).get_json()
        assert payload["maturities"][0] >= 1.0 / 52.0 - 1e-9
        assert max(max(row) for row in payload["implied_vol"]) < 0.6

    def test_explicit_t_min_is_honoured(self, client: Any) -> None:
        """The auto grid starts at the shortest *fitted* expiry or t_min, whichever is later.

        Both chains now drop their one-day expiry on the minimum-maturity
        screen, so the earliest fitted maturity is 0.0795 and a smaller t_min
        cannot pull the grid below it. Raising t_min still moves it.
        """
        early = client.post(
            "/api/vol_surface/query",
            json={"underlying": "510050", "t_min": 0.002, "n_maturities": 3},
        ).get_json()
        assert early["maturities"][0] == pytest.approx(0.0795, abs=1e-3)

        later = client.post(
            "/api/vol_surface/query",
            json={"underlying": "510050", "t_min": 0.5, "n_maturities": 3},
        ).get_json()
        assert later["maturities"][0] == pytest.approx(0.5, abs=1e-6)

    def test_at_the_money_matches_the_fitted_term_structure(self, client: Any) -> None:
        """The grid must be the same surface the fit reported, not a re-fit."""
        fitted = client.post("/api/vol_surface/fit", json={"underlying": "510300"}).get_json()
        quoted = fitted["calibration"]["params"]["maturities"]
        atm_vols = fitted["calibration"]["params"]["atm_vols"]

        payload = client.post(
            "/api/vol_surface/query",
            json={"underlying": "510300", "moneyness": [0.0], "maturities": quoted},
        ).get_json()

        for row, expected in zip(payload["implied_vol"], atm_vols, strict=True):
            assert row[0] == pytest.approx(expected, abs=1e-5)

    def test_query_by_strike_uses_the_reference_forward(self, client: Any) -> None:
        payload = client.post(
            "/api/vol_surface/query",
            json={"underlying": "510050", "strikes": [2.5, 2.8542, 3.2], "maturities": [1.0]},
        ).get_json()

        forward = payload["reference_forward"]
        assert payload["strikes"] == pytest.approx([2.5, 2.8542, 3.2], abs=1e-4)
        assert payload["moneyness"][1] == pytest.approx(0.0, abs=1e-4)
        # The nearest fitted expiry's forward, now the 29-day one.
        assert forward == pytest.approx(2.8542, abs=1e-3)

    def test_skew_is_to_the_downside(self, client: Any) -> None:
        """An equity index surface with the skew the wrong way is a sign error."""
        payload = client.post(
            "/api/vol_surface/query",
            json={"underlying": "510050", "moneyness": [-0.15, 0.0, 0.15], "maturities": [1.0]},
        ).get_json()
        low, atm, high = payload["implied_vol"][0]

        assert low > atm
        assert low > high

    def test_arbitrage_conditions_travel_with_the_grid(self, client: Any) -> None:
        """The extrapolated region is where they are least guaranteed."""
        payload = client.post(
            "/api/vol_surface/query", json={"underlying": "510050", "t_max": 3.0}
        ).get_json()

        assert payload["arbitrage"]["arbitrage_free"] is True
        assert all(v > 0 for row in payload["butterfly"] for v in row)
        assert all(v >= 0 for row in payload["dw_dT"] for v in row)

    def test_total_variance_agrees_with_the_implied_vol(self, client: Any) -> None:
        payload = client.post(
            "/api/vol_surface/query",
            json={"underlying": "510300", "n_moneyness": 5, "n_maturities": 4},
        ).get_json()

        for maturity, vols, variances in zip(
            payload["maturities"], payload["implied_vol"], payload["total_variance"], strict=True
        ):
            for vol, variance in zip(vols, variances, strict=True):
                assert variance == pytest.approx(vol * vol * maturity, abs=1e-6)

    def test_says_how_to_read_the_matrix(self, client: Any) -> None:
        """An unlabelled matrix gets transposed by someone eventually."""
        payload = client.post("/api/vol_surface/query", json={"underlying": "510050"}).get_json()
        assert "implied_vol[i][j]" in payload["note"]
        assert "extrapolated" in payload["note"]

    @pytest.mark.parametrize(
        ("body", "fragment"),
        [
            ({"underlying": "159915"}, "not supported"),
            ({"underlying": "510050", "t_max": -1}, "t_max must be positive"),
            ({"underlying": "510050", "t_max": 0}, "t_max must be positive"),
            ({"underlying": "510050", "t_min": 5.0}, "t_min must be positive"),
            ({"underlying": "510050", "strikes": [0.0, 1.0]}, "strikes must be positive"),
            ({"underlying": "510050", "strikes": [-1.0]}, "strikes must be positive"),
            ({"underlying": "510050", "maturities": [0.0]}, "maturities must be positive"),
            ({"underlying": "510050", "maturities": [-0.5]}, "maturities must be positive"),
            ({"underlying": "510050", "strikes": ["abc"]}, "Invalid grid"),
            ({"underlying": "510050", "n_moneyness": "many"}, "Invalid parameters"),
        ],
    )
    def test_refuses_malformed_requests(
        self, client: Any, body: dict[str, Any], fragment: str
    ) -> None:
        response = client.post("/api/vol_surface/query", json=body)
        assert response.status_code == 400
        assert fragment in response.get_json()["error"]

    def test_clamps_an_absurd_grid_rather_than_allocating_it(self, client: Any) -> None:
        payload = client.post(
            "/api/vol_surface/query",
            json={"underlying": "510050", "n_moneyness": 100000, "n_maturities": 100000},
        ).get_json()

        assert len(payload["moneyness"]) == 401
        assert len(payload["maturities"]) == 201


# ---------------------------------------------------------------------------
# The routes Phase 2a was not supposed to touch
# ---------------------------------------------------------------------------


class TestExistingRoutesUnchanged:
    EXPECTED_ROUTES = {
        "/",
        "/api/calculate",
        "/api/iv",
        "/api/portfolio",
        "/api/smile_presets",
        "/api/vol_smile",
        "/api/mc_distribution",
        "/api/sensitivity",
        "/api/export",
        "/api/snowball",
        "/api/snowball_pde",
        "/api/snowball_crosscheck",
        "/api/snowball_risk",
        "/api/snowball_risk_crosscheck",
        "/api/shark_fin",
        "/api/forward",
        "/api/irs",
        "/api/cn_fdm",
    }
    NEW_ROUTES = {
        "/api/vol_surface/fit",
        "/api/vol_surface/query",
        "/api/vol_surface/snapshots",
        # Phase 2b
        "/api/local_vol/surface",
        "/api/local_vol/validate",
        "/api/european",
        # Phase 3b
        "/api/heston/calibrate",
        "/api/heston/european",
        "/api/heston/snowball",
        # Phase 5: the quoting and carry capabilities, which existed and had
        # no caller. The solver has been validated since Phase 4 and the carry
        # measurement since Phase 4 too; neither was reachable.
        "/api/snowball_quote",
        "/api/carry",
        "/api/snowball_report",
        # Phase 7: marking and publishing. Pricing routes fitted their own
        # surface and nothing made one *the* surface; a desk marks once,
        # checks, publishes, and values the book off that one mark.
        "/api/marketdata/upload",
        "/api/marketdata/publish",
        "/api/marketdata/mark",
    }

    def test_every_prior_route_still_exists(self) -> None:
        import app as app_module

        registered = {rule.rule for rule in app_module.app.url_map.iter_rules()}
        assert registered >= self.EXPECTED_ROUTES

    def test_the_route_table_is_exactly_what_is_intended(self) -> None:
        import app as app_module

        registered = {
            rule.rule
            for rule in app_module.app.url_map.iter_rules()
            if rule.rule != "/static/<path:filename>"
        }
        assert registered == self.EXPECTED_ROUTES | self.NEW_ROUTES

    def test_vanilla_pricing_is_unaffected(self, client: Any) -> None:
        response = client.post(
            "/api/calculate",
            json={"flag": "c", "S": 100, "K": 100, "T": 0.25, "r": 0.02, "sigma": 0.2, "q": 0.0},
        )
        assert response.status_code == 200
        payload = response.get_json()
        # The Phase 1 reference case, unchanged since the baseline.
        assert payload["bsm_price"] == pytest.approx(4.2322, abs=5e-4)
        assert payload["greeks"]["delta"] == pytest.approx(0.539828, abs=1e-5)

    def test_vol_smile_still_answers_and_now_points_onwards(self, client: Any) -> None:
        response = client.post(
            "/api/vol_smile",
            json={
                "flag": "c",
                "S": 100,
                "T": 0.25,
                "r": 0.02,
                "pairs": [{"K": 95, "market_price": 7.0}, {"K": 100, "market_price": 4.0}],
            },
        )
        assert response.status_code == 200
        payload = response.get_json()

        assert len(payload["results"]) == 2
        assert all(point["iv"] is not None for point in payload["results"])
        assert "/api/vol_surface/fit" in payload["note"]
