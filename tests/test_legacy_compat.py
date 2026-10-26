"""Tests for the legacy compatibility layer and the Flask API surface.

``bs_pricing.py`` and ``greeks_calc.py`` are now thin shims over
:mod:`otcpricer.models.bsm`, kept so that ``app.py``, ``main.py`` and
``visualization.py`` need no changes. These tests pin that contract: the
signatures, the ``(bool, str)`` return of ``validate_params``, the exception
type, and the DataFrame shape.

They also assert the headline numbers the project publishes in its README,
through the real HTTP layer, so a refactor cannot move them unnoticed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# The legacy modules live at the repository root, not under src/.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import bs_pricing  # noqa: E402
import greeks_calc  # noqa: E402

from .conftest import BASELINE, BASELINE_EXPECTED  # noqa: E402

BASELINE_ARGS = (
    BASELINE.option_type,
    BASELINE.S,
    BASELINE.K,
    BASELINE.T,
    BASELINE.r,
    BASELINE.sigma,
    BASELINE.q,
)


class TestPyVollibIsGone:
    """The deprecated dependency must not be reachable from the runtime path."""

    def test_pricing_modules_do_not_import_py_vollib(self) -> None:
        for module in (bs_pricing, greeks_calc):
            source = Path(module.__file__).read_text(encoding="utf-8")
            assert "import py_vollib" not in source
            assert "from py_vollib" not in source

    def test_importing_the_app_does_not_pull_in_py_vollib(self) -> None:
        """Guards against a transitive re-introduction."""
        for name in [m for m in sys.modules if m.startswith(("py_vollib", "vollib"))]:
            del sys.modules[name]
        for name in [m for m in sys.modules if m == "app"]:
            del sys.modules[name]

        import app  # noqa: F401

        assert "py_vollib" not in sys.modules
        assert "vollib" not in sys.modules


class TestValidateParamsContract:
    """``main.py`` inspects the returned pair, so the shape must not change."""

    def test_returns_tuple_of_bool_and_message(self) -> None:
        result = bs_pricing.validate_params(100.0, 100.0, 1.0, 0.02, 0.2, 0.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        valid, message = result
        assert valid is True
        assert isinstance(message, str)

    def test_reports_failure_without_raising(self) -> None:
        valid, message = bs_pricing.validate_params(-1.0, 100.0, 1.0, 0.02, 0.2, 0.0)
        assert valid is False
        assert message

    @pytest.mark.parametrize(
        "args",
        [
            (0.0, 100.0, 1.0, 0.02, 0.2, 0.0),
            (100.0, 0.0, 1.0, 0.02, 0.2, 0.0),
            (100.0, 100.0, 0.0, 0.02, 0.2, 0.0),
            (100.0, 100.0, 1.0, 0.02, 0.0, 0.0),
            (100.0, 100.0, 1.0, 0.02, 0.2, -0.1),
        ],
    )
    def test_rejects_invalid_inputs(self, args: tuple[float, ...]) -> None:
        valid, _ = bs_pricing.validate_params(*args)
        assert valid is False

    def test_volatility_cap_is_now_500_percent(self) -> None:
        """Deliberate widening: the old 200% cap rejected legitimate quotes."""
        assert bs_pricing.validate_params(100.0, 100.0, 1.0, 0.02, 3.0, 0.0)[0] is True
        assert bs_pricing.validate_params(100.0, 100.0, 1.0, 0.02, 6.0, 0.0)[0] is False


class TestLegacyPricingFunctions:
    def test_bs_price_matches_baseline(self) -> None:
        value = bs_pricing.bs_price(*BASELINE_ARGS)
        assert value == pytest.approx(BASELINE_EXPECTED["price"], abs=1e-4)

    def test_bs_price_raises_value_error_on_bad_input(self) -> None:
        """Legacy callers catch ValueError."""
        with pytest.raises(ValueError):
            bs_pricing.bs_price("c", -1.0, 100.0, 1.0, 0.02, 0.2, 0.0)

    def test_monte_carlo_is_close_to_closed_form(self) -> None:
        analytic = bs_pricing.bs_price(*BASELINE_ARGS)
        simulated = bs_pricing.monte_carlo_price(*BASELINE_ARGS, n_simulations=200_000)
        # Three standard errors is a generous but still meaningful band.
        assert simulated == pytest.approx(analytic, rel=0.02)

    def test_monte_carlo_is_reproducible_for_a_fixed_seed(self) -> None:
        first = bs_pricing.monte_carlo_price(*BASELINE_ARGS, n_simulations=50_000, seed=7)
        second = bs_pricing.monte_carlo_price(*BASELINE_ARGS, n_simulations=50_000, seed=7)
        assert first == second

    def test_monte_carlo_does_not_disturb_global_random_state(self) -> None:
        """The old implementation called np.random.seed, mutating global state.

        That made any later sampling in the same process depend on whether a
        price had been computed first.
        """
        import numpy as np

        # The legacy global RNG API is used deliberately here: the whole point
        # is to prove that pricing no longer perturbs this global state.
        np.random.seed(12345)  # noqa: NPY002
        before = np.random.standard_normal(3).tolist()  # noqa: NPY002

        np.random.seed(12345)  # noqa: NPY002
        bs_pricing.monte_carlo_price(*BASELINE_ARGS, n_simulations=1000, seed=99)
        after = np.random.standard_normal(3).tolist()  # noqa: NPY002

        assert before == after

    def test_monte_carlo_rejects_unknown_option_type(self) -> None:
        with pytest.raises(ValueError, match="flag"):
            bs_pricing.monte_carlo_price("x", 100.0, 100.0, 1.0, 0.02, 0.2, 0.0)

    def test_implied_volatility_round_trip(self) -> None:
        premium = bs_pricing.bs_price(*BASELINE_ARGS)
        recovered = bs_pricing.calc_implied_volatility(
            BASELINE.option_type,
            premium,
            BASELINE.S,
            BASELINE.K,
            BASELINE.T,
            BASELINE.r,
            BASELINE.q,
        )
        assert recovered == pytest.approx(BASELINE.sigma, abs=1e-8)

    def test_implied_volatility_raises_value_error_not_library_error(self) -> None:
        """The documented legacy contract is ValueError; callers rely on it."""
        with pytest.raises(ValueError):
            bs_pricing.calc_implied_volatility("c", 99_999.0, 5500.0, 5500.0, 0.25, 0.02, 0.01)

    def test_bs_price_accepts_arrays_for_curve_plotting(self) -> None:
        import numpy as np

        spots = np.linspace(4000.0, 7000.0, 50)
        values = bs_pricing.bs_price("c", spots, 5500.0, 0.25, 0.02, 0.20, 0.01)
        assert values.shape == (50,)
        assert np.all(np.diff(values) > 0.0)


class TestLegacyGreeks:
    @pytest.mark.parametrize("greek", ["delta", "gamma", "vega", "theta", "rho"])
    def test_individual_functions_match_baseline(self, greek: str) -> None:
        fn = getattr(greeks_calc, f"calc_{greek}")
        assert fn(*BASELINE_ARGS) == pytest.approx(BASELINE_EXPECTED[greek], abs=1e-6)

    def test_calc_all_greeks_returns_expected_frame(self) -> None:
        df = greeks_calc.calc_all_greeks(*BASELINE_ARGS)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["希腊字母", "数值", "业务含义"]
        assert list(df["希腊字母"]) == ["Delta", "Gamma", "Vega", "Theta", "Rho"]
        assert len(df) == 5

    def test_calc_all_greeks_values_match_individual_functions(self) -> None:
        df = greeks_calc.calc_all_greeks(*BASELINE_ARGS)
        values = dict(zip(df["希腊字母"], df["数值"], strict=True))
        assert values["Delta"] == pytest.approx(BASELINE_EXPECTED["delta"], abs=1e-6)
        assert values["Vega"] == pytest.approx(BASELINE_EXPECTED["vega"], abs=1e-6)

    def test_greeks_accept_arrays(self) -> None:
        import numpy as np

        spots = np.linspace(4000.0, 7000.0, 30)
        deltas = greeks_calc.calc_delta("c", spots, 5500.0, 0.25, 0.02, 0.20, 0.01)
        assert deltas.shape == (30,)
        assert np.all(np.diff(deltas) > 0.0)


class TestFlaskApiRegression:
    """End-to-end through the HTTP layer, pinning the published numbers."""

    @pytest.fixture(scope="class")
    def client(self):  # noqa: ANN201
        import app

        app.app.config.update(TESTING=True)
        return app.app.test_client()

    def test_index_page_renders(self, client) -> None:  # noqa: ANN001
        response = client.get("/")
        assert response.status_code == 200
        assert len(response.data) > 10_000

    def test_calculate_returns_baseline_price_and_greeks(self, client) -> None:  # noqa: ANN001
        response = client.post(
            "/api/calculate",
            json={
                "option_type": "c",
                "S": 5500,
                "K": 5500,
                "T": 0.25,
                "r": 0.02,
                "sigma": 0.20,
                "q": 0.01,
            },
        )
        assert response.status_code == 200
        body = response.get_json()
        assert body["bsm_price"] == pytest.approx(BASELINE_EXPECTED["price"], abs=1e-4)
        for greek in ("delta", "gamma", "vega", "theta", "rho"):
            assert body["greeks"][greek] == pytest.approx(BASELINE_EXPECTED[greek], abs=1e-6), greek

    def test_implied_vol_endpoint(self, client) -> None:  # noqa: ANN001
        response = client.post(
            "/api/iv",
            json={
                "option_type": "c",
                "market_price": BASELINE_EXPECTED["price"],
                "S": 5500,
                "K": 5500,
                "T": 0.25,
                "r": 0.02,
                "q": 0.01,
            },
        )
        assert response.status_code == 200
        # NOTE: /api/iv reports a decimal (0.20) while /api/vol_smile reports a
        # percentage (19.9997) for the same quantity. Pinned here as the current
        # behaviour; the inconsistency is on the list to fix when the API layer
        # is rebuilt behind one response schema.
        assert response.get_json()["implied_volatility"] == pytest.approx(0.20, abs=1e-5)

    def test_iv_and_vol_smile_disagree_on_units(self, client) -> None:  # noqa: ANN001
        """Documents a real API inconsistency rather than letting it surprise someone.

        Both endpoints invert the same BSM formula for the same input, but one
        returns 0.20 and the other 20.0. This test records that so the fix is
        a deliberate, visible change rather than an accident.
        """
        quote = {"S": 5500, "T": 0.25, "r": 0.02, "q": 0.01}
        scalar = client.post(
            "/api/iv",
            json={"option_type": "c", "market_price": 225.4236, "K": 5500, **quote},
        ).get_json()["implied_volatility"]
        smile = client.post(
            "/api/vol_smile",
            json={"flag": "c", "pairs": [{"K": 5500, "market_price": 225.4236}], **quote},
        ).get_json()["results"][0]["iv"]
        assert smile == pytest.approx(scalar * 100.0, rel=1e-4)

    def test_vol_smile_shows_a_skew(self, client) -> None:  # noqa: ANN001
        """Lower strikes should imply higher vol for this quote set."""
        response = client.post(
            "/api/vol_smile",
            json={
                "flag": "c",
                "S": 5500,
                "T": 0.25,
                "r": 0.02,
                "q": 0.01,
                "pairs": [
                    {"K": 5000, "market_price": 600},
                    {"K": 5500, "market_price": 225.42},
                    {"K": 6000, "market_price": 40},
                ],
            },
        )
        assert response.status_code == 200
        ivs = [point["iv"] for point in response.get_json()["results"]]
        assert all(iv is not None for iv in ivs)
        assert ivs[0] > ivs[1] > ivs[2]

    @pytest.mark.parametrize(
        ("endpoint", "payload"),
        [
            (
                "/api/sensitivity",
                {
                    "option_type": "c",
                    "S": 5500,
                    "K": 5500,
                    "T": 0.25,
                    "r": 0.02,
                    "sigma": 0.2,
                    "q": 0.01,
                },
            ),
            (
                "/api/mc_distribution",
                {
                    "option_type": "c",
                    "S": 5500,
                    "K": 5500,
                    "T": 0.25,
                    "r": 0.02,
                    "sigma": 0.2,
                    "q": 0.01,
                },
            ),
            (
                "/api/smile_presets",
                {
                    "flag": "c",
                    "S": 5500,
                    "T": 0.25,
                    "r": 0.02,
                    "sigma": 0.2,
                    "q": 0.01,
                    "strikes": [5000, 5500, 6000],
                },
            ),
            (
                "/api/shark_fin",
                {"S": 5500, "K": 5500, "H": 6050, "T": 0.25, "r": 0.02, "sigma": 0.2, "q": 0.01},
            ),
            ("/api/forward", {"S": 5500, "T": 1.0, "r": 0.02, "q": 0.01}),
            (
                "/api/irs",
                {"notional": 1e7, "fixed_rate": 0.025, "float_rate": 0.022, "T": 3, "freq": 2},
            ),
        ],
    )
    def test_endpoint_responds_successfully(self, client, endpoint, payload) -> None:  # noqa: ANN001
        response = client.post(endpoint, json=payload)
        assert response.status_code == 200, response.get_json()

    def test_forward_price_matches_cost_of_carry(self, client) -> None:  # noqa: ANN001
        import numpy as np

        response = client.post("/api/forward", json={"S": 5500, "T": 1.0, "r": 0.02, "q": 0.01})
        body = response.get_json()
        assert body["forward_price"] == pytest.approx(5500 * np.exp((0.02 - 0.01) * 1.0), rel=1e-6)

    def test_invalid_payload_returns_400_not_500(self, client) -> None:  # noqa: ANN001
        """A bad request must be a client error, not a crash."""
        response = client.post(
            "/api/calculate",
            json={"option_type": "c", "S": -100, "K": 5500, "T": 0.25, "r": 0.02, "sigma": 0.2},
        )
        assert response.status_code in (400, 422), response.status_code


class TestSnowballEndpoints:
    """The rewired snowball API, now backed by the corrected engines."""

    PAYLOAD = {
        "S": 5000,
        "T": 1.0,
        "r": 0.02,
        "sigma": 0.20,
        "q": 0.01,
        "KO_pct": 1.03,
        "KI_pct": 0.75,
        "coupon_pa": 0.20,
        "n_paths": 100_000,
    }

    @pytest.fixture(scope="class")
    def client(self):  # noqa: ANN201
        import app

        app.app.config.update(TESTING=True)
        return app.app.test_client()

    def test_legacy_response_keys_are_preserved(self, client) -> None:  # noqa: ANN001
        """The existing frontend reads these; rewiring must not break it."""
        body = client.post("/api/snowball", json=self.PAYLOAD).get_json()
        for key in (
            "mid",
            "bid",
            "ask",
            "std_err",
            "ci95",
            "ko_prob",
            "ki_prob",
            "n_paths",
            "params",
        ):
            assert key in body, key
        assert body["bid"] < body["mid"] < body["ask"]

    def test_defaults_apply_the_market_conventions(self, client) -> None:  # noqa: ANN001
        """Daily knock-in, a three-month lock-up, and an S0-struck put.

        The legacy endpoint monitored knock-in only on the monthly knock-out
        dates and had no lock-up at all.
        """
        params = client.post("/api/snowball", json=self.PAYLOAD).get_json()["params"]
        assert params["ki_obs_freq"] == 252
        assert params["lockup_months"] == 3
        assert params["ki_strike_mode"] == "s0"
        assert params["ki_put_strike"] == pytest.approx(5000.0)
        # Twelve monthly dates less the three inside the lock-up.
        assert params["n_ko_observations"] == 9

    def test_new_parameters_are_honoured(self, client) -> None:  # noqa: ANN001
        payload = {
            **self.PAYLOAD,
            "lockup_months": 0,
            "ki_obs_freq": 12,
            "ki_strike_mode": "ki",
            "basis_pa": 0.05,
        }
        params = client.post("/api/snowball", json=payload).get_json()["params"]
        assert params["lockup_months"] == 0
        assert params["ki_obs_freq"] == 12
        assert params["ki_strike_mode"] == "ki"
        assert params["ki_put_strike"] == pytest.approx(3750.0)
        assert params["basis_pa"] == pytest.approx(0.05)
        assert params["n_ko_observations"] == 12

    def test_reports_the_diagnostics_the_old_endpoint_could_not(self, client) -> None:  # noqa: ANN001
        body = client.post("/api/snowball", json=self.PAYLOAD).get_json()
        # Mean knock-out time is what makes the discounting fix visible.
        assert 0.0 < body["expected_ko_time"] < self.PAYLOAD["T"]
        # The loss-making state is a subset of all knock-ins.
        assert body["ki_no_ko_prob"] <= body["ki_prob"]

    def test_pde_endpoint_prices_the_same_product(self, client) -> None:  # noqa: ANN001
        mc = client.post("/api/snowball", json=self.PAYLOAD).get_json()
        pde = client.post("/api/snowball_pde", json=self.PAYLOAD).get_json()
        assert pde["price"] == pytest.approx(mc["mid"], abs=10.0 * mc["ci95"])
        assert pde["space_nodes"] > 0
        assert pde["time_steps"] > 0

    def test_crosscheck_reports_agreement_with_a_measured_budget(self, client) -> None:  # noqa: ANN001
        """The validation view: a verdict plus the budget that produced it."""
        body = client.post("/api/snowball_crosscheck", json=self.PAYLOAD).get_json()
        assert body["agree"] is True
        assert body["gap"] < body["budget"]
        # Both error terms must be measured, not zero-filled.
        breakdown = body["budget_breakdown"]
        assert breakdown["mc_ci95_times_two"] > 0.0
        assert breakdown["pde_grid_change"] > 0.0
        assert breakdown["mc_ci95_times_two"] + breakdown["pde_grid_change"] == pytest.approx(
            body["budget"], rel=1e-9
        )
        # The legacy engines were 18% apart on this trade.
        assert body["gap_relative"] < 0.01

    @pytest.mark.parametrize(
        "bad",
        [
            {"KI_pct": 1.10},  # knock-in above knock-out
            {"lockup_months": 18},  # lock-up longer than the tenor
            {"S": -1},  # negative spot
            {"sigma": 0.0},  # zero volatility
            {"basis_pa": -0.01},  # negative basis
        ],
    )
    def test_invalid_inputs_return_400_not_500(self, client, bad) -> None:  # noqa: ANN001
        """A structurally impossible trade is a client error, not a crash."""
        response = client.post("/api/snowball", json={**self.PAYLOAD, **bad})
        assert response.status_code == 400, response.get_json()
        assert "error" in response.get_json()
