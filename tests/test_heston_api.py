"""Tests for the Heston endpoints.

Offline throughout: the configured source is ``snapshot``, so these exercise
the committed chains.

The emphasis beyond "does it answer" is on the two things this particular API
must not let a caller miss. A Heston snowball rests on **one** engine where
every other number in this service rests on two, and a Heston fit is **worse**
than the SSVI fit of the same quotes. Both are easy to overlook in a JSON
response, so both are asserted to be present in it.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.test_marketdata import SNAPSHOT_DIR


@pytest.fixture(autouse=True)
def _require_snapshots() -> None:
    if not SNAPSHOT_DIR.exists():
        pytest.skip("no snapshots committed")


@pytest.fixture
def client() -> Any:
    import app as app_module

    app_module.app.config["TESTING"] = True
    app_module._SURFACE_CACHE.clear()
    app_module._HESTON_CACHE.clear()
    return app_module.app.test_client()


# Textbook parameters, Feller satisfied, so the European price is a known
# constant independent of any data file.
EXPLICIT = {
    "v0": 0.04,
    "kappa": 2.0,
    "theta": 0.04,
    "sigma": 0.3,
    "rho": -0.7,
}


class TestEuropean:
    def test_reproduces_the_quantlib_checked_price(self, client: Any) -> None:
        """9.2425210739 is where QuantLib's analytic Heston engine puts it.

        Asserted through the HTTP layer so the endpoint's own parameter
        handling is in the path, not just the engine.
        """
        payload = client.post(
            "/api/heston/european",
            json={"K": 100.0, "T": 1.0, "flag": "c", "S": 100.0, "r": 0.03, **EXPLICIT},
        ).get_json()

        assert payload["price"] == pytest.approx(9.2425210739, abs=1e-8)
        assert payload["params_source"] == "explicit"
        assert payload["method"] == "cos-fourier"

    def test_put_call_parity_holds_through_the_endpoint(self, client: Any) -> None:
        import numpy as np

        body = {"K": 95.0, "T": 1.0, "S": 100.0, "r": 0.03, **EXPLICIT}
        call = client.post("/api/heston/european", json={**body, "flag": "c"}).get_json()
        put = client.post("/api/heston/european", json={**body, "flag": "p"}).get_json()

        assert call["price"] - put["price"] == pytest.approx(
            100.0 - 95.0 * float(np.exp(-0.03)), abs=1e-8
        )

    def test_says_which_payoff_it_integrates_and_why(self, client: Any) -> None:
        """Because it explains an otherwise surprising implementation choice."""
        payload = client.post(
            "/api/heston/european",
            json={"K": 100.0, "T": 1.0, "S": 100.0, **EXPLICIT},
        ).get_json()
        assert "parity" in payload["note"]
        assert "overflow" in payload["note"]

    def test_reports_the_feller_margin_with_the_parameters(self, client: Any) -> None:
        payload = client.post(
            "/api/heston/european",
            json={"K": 100.0, "T": 1.0, "S": 100.0, **EXPLICIT},
        ).get_json()
        params = payload["params"]
        assert params["satisfies_feller"] is True
        assert params["feller_margin"] == pytest.approx(0.07, abs=1e-9)

    def test_falls_back_to_a_calibration_when_parameters_are_omitted(self, client: Any) -> None:
        """And says the spot it chose, since the surface has none.

        A fitted surface is parameterised in forward terms and carries no spot,
        so the endpoint discounts the nearest expiry's forward back and reports
        the result rather than inventing 100.
        """
        payload = client.post(
            "/api/heston/european", json={"underlying": "510300", "K": 4.1, "T": 0.4055}
        ).get_json()

        assert payload["params_source"] == "calibrated:510300"
        assert 3.0 < payload["spot"] < 5.0
        assert payload["price"] > 0.0

    @pytest.mark.parametrize(
        ("body", "fragment"),
        [
            ({"T": 1.0, **EXPLICIT}, "Invalid parameters"),
            ({"K": 100.0, **EXPLICIT}, "Invalid parameters"),
            ({"K": 0.0, "T": 1.0, **EXPLICIT}, "must be positive"),
            ({"K": 100.0, "T": -1.0, **EXPLICIT}, "must be positive"),
            ({"K": 100.0, "T": 1.0, **{**EXPLICIT, "v0": -1.0}}, "v0 must be positive"),
            ({"K": 100.0, "T": 1.0, **{**EXPLICIT, "rho": 1.0}}, "rho must lie in"),
            ({"K": 100.0, "T": 1.0, "underlying": "159915"}, "not supported"),
        ],
    )
    def test_refuses_malformed_requests(
        self, client: Any, body: dict[str, Any], fragment: str
    ) -> None:
        response = client.post("/api/heston/european", json=body)
        assert response.status_code == 400
        assert fragment in response.get_json()["error"]


@pytest.mark.slow
class TestCalibrate:
    def test_reports_that_it_fits_worse_than_ssvi(self, client: Any) -> None:
        """The comparison the endpoint exists to make, and it is unflattering.

        Served rather than buried: a caller choosing Heston for sophistication
        should see in the response that it describes today's quotes less well.
        """
        payload = client.post("/api/heston/calibrate", json={"underlying": "510300"}).get_json()

        assert payload["fits_worse_than_ssvi"] is True
        assert payload["heston_over_ssvi"] > 1.0
        assert payload["rms_vol_points"] > payload["ssvi_rms_vol_points"]
        # Within the measured 10-12% band on a three-expiry chain.
        assert payload["heston_over_ssvi"] < 1.25

    def test_feller_comes_back_violated_on_real_quotes(self, client: Any) -> None:
        payload = client.post("/api/heston/calibrate", json={"underlying": "510300"}).get_json()
        assert payload["params"]["satisfies_feller"] is False
        assert payload["params"]["feller_margin"] < 0.0

    def test_separates_convergence_from_corroboration(self, client: Any) -> None:
        """Three facts, not one. See the calibration module for why."""
        payload = client.post("/api/heston/calibrate", json={"underlying": "510300"}).get_json()

        for key in (
            "converged",
            "confirmed_by_multiple_starts",
            "ambiguous",
            "well_identified",
            "parameter_spread",
            "n_comparable_starts",
        ):
            assert key in payload

    def test_the_report_leads_with_the_caveats(self, client: Any) -> None:
        payload = client.post(
            "/api/heston/calibrate", json={"underlying": "510300", "report": True}
        ).get_json()
        report = payload["report"]

        assert "READ THIS FIRST" in report
        assert "ONE ENGINE" in report
        # And the caveat arrives before the numbers it qualifies.
        assert report.index("READ THIS FIRST") < report.index("FITTED PARAMETERS")

    def test_the_report_is_optional(self, client: Any) -> None:
        plain = client.post("/api/heston/calibrate", json={"underlying": "510300"}).get_json()
        assert "report" not in plain

    def test_caches_so_a_second_call_is_free(self, client: Any) -> None:
        """A minute a fit is too slow to repeat for the same inputs."""
        import app as app_module

        assert not app_module._HESTON_CACHE
        client.post("/api/heston/calibrate", json={"underlying": "510300"})
        assert len(app_module._HESTON_CACHE) == 1
        client.post("/api/heston/calibrate", json={"underlying": "510300"})
        assert len(app_module._HESTON_CACHE) == 1

        # A different rate is a different fit.
        client.post("/api/heston/calibrate", json={"underlying": "510300", "r": 0.035})
        assert len(app_module._HESTON_CACHE) == 2


class TestSnowball:
    def test_the_response_says_it_has_only_one_engine(self, client: Any) -> None:
        """The single most important field in this response.

        Every other price this service returns is confirmed by an independent
        second engine. This one is not, and a caller reading JSON would have no
        way to know unless the response says so.
        """
        payload = client.post(
            "/api/heston/snowball",
            json={
                "S0": 5000.0,
                "T": 1.0,
                "ko_pct": 1.03,
                "ki_pct": 0.75,
                "coupon_pa": 0.20,
                "r": 0.02,
                "q": 0.01,
                "paths": 20_000,
                "steps_per_year": 63,
                **EXPLICIT,
            },
        ).get_json()

        assert payload["vol_model"] == "heston-qe"
        assert payload["cross_checked_by_a_second_engine"] is False
        assert "single_engine_note" in payload
        assert "QE is verified" in payload["single_engine_note"]

    def test_returns_the_statistics_a_desk_quotes(self, client: Any) -> None:
        payload = client.post(
            "/api/heston/snowball",
            json={"paths": 20_000, "steps_per_year": 63, **EXPLICIT},
        ).get_json()

        for key in ("price", "std_error", "ci95", "ko_prob", "ki_prob", "ki_no_ko_prob"):
            assert key in payload
        assert 0.0 <= payload["ko_prob"] <= 1.0
        assert 0.0 <= payload["ki_no_ko_prob"] <= payload["ki_prob"] + 1e-12
        assert payload["std_error"] > 0.0

    def test_a_higher_coupon_is_worth_more(self, client: Any) -> None:
        """A monotonicity the model cannot be allowed to break."""
        base = {"paths": 20_000, "steps_per_year": 63, "seed": 3, **EXPLICIT}
        low = client.post("/api/heston/snowball", json={**base, "coupon_pa": 0.10}).get_json()
        high = client.post("/api/heston/snowball", json={**base, "coupon_pa": 0.30}).get_json()
        assert high["price"] > low["price"]

    def test_echoes_the_terms_and_the_step_count(self, client: Any) -> None:
        """The step count matters: under QE it is an accuracy parameter."""
        payload = client.post(
            "/api/heston/snowball",
            json={"paths": 20_000, "steps_per_year": 126, "ko_pct": 1.05, **EXPLICIT},
        ).get_json()
        assert payload["steps_per_year"] == 126
        assert payload["terms"]["ko_pct"] == 1.05

    @pytest.mark.parametrize(
        ("body", "fragment"),
        [
            # The product validator's own messages come through unwrapped,
            # which is better than a generic "Invalid parameters" -- they name
            # the structural rule that was broken.
            ({**EXPLICIT, "ko_pct": 0.5, "ki_pct": 0.75}, "barriers cross"),
            ({**EXPLICIT, "T": -1.0}, "T must be greater than 0"),
            ({**{**EXPLICIT, "sigma": 0.0}}, "sigma must be positive"),
            ({"underlying": "159915"}, "not supported"),
        ],
    )
    def test_refuses_malformed_requests(
        self, client: Any, body: dict[str, Any], fragment: str
    ) -> None:
        response = client.post(
            "/api/heston/snowball", json={"paths": 2_000, "steps_per_year": 12, **body}
        )
        assert response.status_code == 400
        assert fragment in response.get_json()["error"]
