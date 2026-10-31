"""Tests for the local-volatility endpoints.

Offline throughout, like the surface endpoints: the configured source is
``snapshot``, so these exercise the committed chains.

The emphasis beyond "does it answer" is on the two things a caller can most
easily get wrong with this particular API. Local volatility is not implied
volatility, and a response that returned only the former would invite the
confusion -- so both are returned at the same points and a test says so. And
an option's implied volatility has to be inverted at its own expiry's forward,
not at spot; on this data the two differ by a 4% carry, and inverting at spot
would report that as a model error.
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
    app_module._LOCAL_VOL_CACHE.clear()
    return app_module.app.test_client()


class TestSurfaceGrid:
    def test_returns_a_consistent_grid(self, client: Any) -> None:
        payload = client.post(
            "/api/local_vol/surface",
            json={"underlying": "510300", "n_spots": 7, "n_times": 5, "horizon": 3.0},
        ).get_json()

        assert len(payload["spots"]) == 7
        assert len(payload["moneyness"]) == 7
        assert len(payload["times"]) == 5
        for key in ("local_vol", "implied_vol"):
            assert len(payload[key]) == 5
            assert all(len(row) == 7 for row in payload[key])

    def test_returns_implied_volatility_alongside(self, client: Any) -> None:
        """The standing trap this guards against.

        Local volatility and implied volatility are different objects at the
        same coordinates, and the gap is large: at three years the 30%-down
        level carries a local volatility well above its implied. Returning only
        one would invite reading it as the other.
        """
        payload = client.post(
            "/api/local_vol/surface",
            json={"underlying": "510300", "n_spots": 5, "n_times": 3},
        ).get_json()

        local = payload["local_vol"][-1]
        implied = payload["implied_vol"][-1]
        assert local != implied
        # Away from the money the local volatility is the steeper of the two.
        assert local[0] > implied[0]
        assert "not" in payload["note"]

    def test_leads_with_the_extrapolation_share(self, client: Any) -> None:
        """The dominant exposure, carried in the response rather than only the report."""
        payload = client.post(
            "/api/local_vol/surface", json={"underlying": "510300", "horizon": 3.0}
        ).get_json()
        assert payload["extrapolated_fraction"] == pytest.approx(0.865, abs=0.01)

        near = client.post(
            "/api/local_vol/surface", json={"underlying": "510300", "horizon": 0.3}
        ).get_json()
        assert near["extrapolated_fraction"] == 0.0

    def test_reports_the_forward_curve_and_its_kink(self, client: Any) -> None:
        payload = client.post("/api/local_vol/surface", json={"underlying": "510300"}).get_json()
        curve = payload["forward_curve"]
        assert curve["kink"] != 0.0
        for key in ("maturities", "forwards", "segment_drifts", "implied_carry"):
            assert key in curve

    def test_reports_the_numerical_diagnostics(self, client: Any) -> None:
        payload = client.post("/api/local_vol/surface", json={"underlying": "510300"}).get_json()
        diagnostics = payload["diagnostics"]

        assert diagnostics["n_undefined"] == 0
        assert diagnostics["min_denominator"] > 0.0
        assert diagnostics["well_defined"] is True
        assert 0.0 < diagnostics["max_peclet"] < 2.0

    def test_the_tail_drift_comes_from_the_callers_carry(self, client: Any) -> None:
        """Not extrapolated from the implied forwards, deliberately.

        Carrying a -4% carry measured on five-month options out to three years
        would drop the forward 14% on that evidence alone. So the caller's own
        ``r - q - basis`` governs past the quotes, and changing it changes the
        curve.
        """
        base = client.post(
            "/api/local_vol/surface", json={"underlying": "510300", "q": 0.0}
        ).get_json()
        with_dividend = client.post(
            "/api/local_vol/surface", json={"underlying": "510300", "q": 0.05}
        ).get_json()

        assert with_dividend["forward_curve"]["tail_drift"] < base["forward_curve"]["tail_drift"]
        # The quoted forwards are observations and must not move.
        assert with_dividend["forward_curve"]["forwards"] == base["forward_curve"]["forwards"]

    @pytest.mark.parametrize(
        ("body", "fragment"),
        [
            ({"underlying": "159915"}, "not supported"),
            ({"underlying": "510300", "horizon": 0}, "horizon must be positive"),
            ({"underlying": "510300", "horizon": -1}, "horizon must be positive"),
            ({"underlying": "510300", "S": 0}, "S must be positive"),
            ({"underlying": "510300", "S": -5}, "S must be positive"),
            ({"underlying": "510300", "n_spots": "many"}, "Invalid parameters"),
        ],
    )
    def test_refuses_malformed_requests(
        self, client: Any, body: dict[str, Any], fragment: str
    ) -> None:
        response = client.post("/api/local_vol/surface", json=body)
        assert response.status_code == 400
        assert fragment in response.get_json()["error"]


@pytest.mark.slow
class TestValidateEndpoint:
    def test_round_trips_within_the_calibration_error(self, client: Any) -> None:
        """The correctness evidence, served.

        The budget is the surface's own calibration error: an engine that adds
        less noise than the data already carries is as accurate as the inputs
        can justify.
        """
        payload = client.post("/api/local_vol/validate", json={"underlying": "510300"}).get_json()

        assert payload["passes"] is True
        assert payload["worst_error_vol_points"] < payload["calibration_rms_vol_points"]
        assert payload["share_of_budget"] < 0.1
        assert len(payload["rows"]) == 15

    def test_each_row_carries_both_volatilities(self, client: Any) -> None:
        payload = client.post("/api/local_vol/validate", json={"underlying": "510300"}).get_json()
        for row in payload["rows"]:
            assert 0.01 < row["target_vol"] < 1.5
            assert 0.01 < row["recovered_vol"] < 1.5
            assert row["error_vol_points"] == pytest.approx(
                (row["recovered_vol"] - row["target_vol"]) * 100, abs=1e-4
            )

    def test_report_is_optional(self, client: Any) -> None:
        plain = client.post("/api/local_vol/validate", json={"underlying": "510300"}).get_json()
        assert "report" not in plain

        full = client.post(
            "/api/local_vol/validate", json={"underlying": "510300", "report": True}
        ).get_json()
        assert "EXTRAPOLATION -- READ THIS FIRST" in full["report"]
        assert "DUPIRE ROUND TRIP" in full["report"]


class TestEuropeanEndpoint:
    def test_prices_and_recovers_the_surface_volatility(self, client: Any) -> None:
        """A skew-consistent vanilla quote, which a flat volatility cannot give."""
        payload = client.post(
            "/api/european",
            json={"underlying": "510300", "K": 4.1, "T": 0.4055, "flag": "c"},
        ).get_json()

        assert payload["price"] > 0.0
        assert payload["vol_model"] == "dupire-local"
        assert payload["implied_vol"] == pytest.approx(payload["surface_implied_vol"], abs=0.005)

    def test_inverts_at_the_forward_not_at_spot(self, client: Any) -> None:
        """Which is what makes implied_vol comparable with the surface's.

        The two differ by the implied carry, about -4% a year here, so
        inverting at spot would measure that and report it as a model error --
        the exact mistake that made the first round trip look five vol points
        wrong.
        """
        payload = client.post(
            "/api/european",
            json={"underlying": "510300", "K": 4.1, "T": 0.4055, "flag": "c"},
        ).get_json()

        assert payload["forward"] != payload["spot"]
        assert payload["forward"] < payload["spot"]  # declining forwards
        assert "forward" in payload["note"]

    def test_local_volatility_differs_from_the_quoted_one(self, client: Any) -> None:
        payload = client.post(
            "/api/european",
            json={"underlying": "510300", "K": 4.1, "T": 1.0, "flag": "c"},
        ).get_json()
        assert payload["local_vol_at_spot"] != payload["surface_implied_vol"]

    def test_puts_and_calls_satisfy_parity(self, client: Any) -> None:
        """Holds whatever the volatility, so it tests the drift and discounting."""
        body = {"underlying": "510300", "K": 4.0, "T": 0.4055}
        call = client.post("/api/european", json={**body, "flag": "c"}).get_json()
        put = client.post("/api/european", json={**body, "flag": "p"}).get_json()

        import numpy as np

        discount = float(np.exp(-0.02 * 0.4055))
        assert call["price"] - put["price"] == pytest.approx(
            discount * (call["forward"] - 4.0), abs=2e-3
        )

    @pytest.mark.parametrize(
        ("body", "fragment"),
        [
            ({"underlying": "510300", "K": 4.1}, "Invalid parameters"),
            ({"underlying": "510300", "T": 1.0}, "Invalid parameters"),
            ({"underlying": "510300", "K": 0, "T": 1.0}, "must be positive"),
            ({"underlying": "510300", "K": 4.1, "T": -1}, "must be positive"),
            ({"underlying": "159915", "K": 4.1, "T": 1.0}, "not supported"),
        ],
    )
    def test_refuses_malformed_requests(
        self, client: Any, body: dict[str, Any], fragment: str
    ) -> None:
        response = client.post("/api/european", json=body)
        assert response.status_code == 400
        assert fragment in response.get_json()["error"]


class TestCache:
    def test_keys_on_everything_that_changes_the_answer(self, client: Any) -> None:
        """A cache keyed more loosely than its inputs is a wrong-answer machine."""
        import app as app_module

        assert not app_module._LOCAL_VOL_CACHE
        client.post("/api/local_vol/surface", json={"underlying": "510300"})
        assert len(app_module._LOCAL_VOL_CACHE) == 1

        client.post("/api/local_vol/surface", json={"underlying": "510300"})
        assert len(app_module._LOCAL_VOL_CACHE) == 1

        # A different carry is a different forward curve, hence a different
        # moneyness coordinate, hence a different local volatility.
        client.post("/api/local_vol/surface", json={"underlying": "510300", "q": 0.05})
        assert len(app_module._LOCAL_VOL_CACHE) == 2

        # And a different anchoring spot.
        client.post("/api/local_vol/surface", json={"underlying": "510300", "S": 100.0})
        assert len(app_module._LOCAL_VOL_CACHE) == 3
