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
                # monitoring is required now, and deliberately so: a daily
                # knock-out is worth about 19% more than a continuous one, so
                # the frequency is part of the contract rather than a default.
                "/api/shark_fin",
                {
                    "S": 5500,
                    "K": 5500,
                    "H": 6050,
                    "T": 0.25,
                    "r": 0.02,
                    "sigma": 0.2,
                    "q": 0.01,
                    "monitoring": 252,
                },
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
            {"basis_pa": -6.0},  # basis handed over in percent, not a decimal
        ],
    )
    def test_invalid_inputs_return_400_not_500(self, client, bad) -> None:  # noqa: ANN001
        """A structurally impossible trade is a client error, not a crash."""
        response = client.post("/api/snowball", json={**self.PAYLOAD, **bad})
        assert response.status_code == 400, response.get_json()
        assert "error" in response.get_json()


class TestTheLegacyEngineIsGone:
    """cn_fdm.py had one caller left, and it was serving a 45% error.

    The ``calc_greeks=false`` branch of /api/cn_fdm ran ``CN_FDM.price()``,
    which ignored the lock-up, the knock-in strike mode and the basis -- and
    whose own PDE and Monte Carlo struck the knock-in put at different levels,
    so the two halves of one class priced different contracts.

    On the reference trade it returned **0.021250** against the reviewed
    engine's **0.038599**: wrong by 44.9% of price, on a route a caller could
    reach by passing one flag.
    """

    PAYLOAD = {
        "product_type": "snowball",
        "S0": 5000,
        "T": 1.0,
        "r": 0.02,
        "sigma": 0.20,
        "q": 0.01,
        "KI_pct": 0.75,
        "coupon_pa": 0.15,
    }

    @pytest.fixture(scope="class")
    def client(self):  # noqa: ANN201
        import app

        app.app.config.update(TESTING=True)
        return app.app.test_client()

    def test_the_module_cannot_be_imported_because_it_is_deleted(self) -> None:
        """The strongest form of retirement available."""
        with pytest.raises(ModuleNotFoundError):
            import cn_fdm  # noqa: F401

    def test_both_branches_return_the_same_price(self, client) -> None:  # noqa: ANN001
        """Which is what says they are one engine rather than two.

        Asking for Greeks must not change the price. Before this, one branch
        ran the reviewed PDE and the other ran the legacy class, so the flag
        moved the price by 45%.
        """
        with_greeks = client.post(
            "/api/cn_fdm", json={**self.PAYLOAD, "calc_greeks": True}
        ).get_json()
        without = client.post("/api/cn_fdm", json={**self.PAYLOAD, "calc_greeks": False}).get_json()

        assert with_greeks["price"] == without["price"]
        assert without["price"] == pytest.approx(0.038599, abs=1e-6)
        assert without["price"] != pytest.approx(0.021250, abs=1e-4), (
            "this is the legacy number; the branch has not been migrated"
        )

    def test_the_echoed_barrier_is_the_one_that_was_priced(self, client) -> None:  # noqa: ANN001
        """The defect this fixes, stated as the test that catches it.

        /api/cn_fdm defaulted KO_pct to 1.05 for its response echo and the
        parser defaulted it to 1.03 for the actual pricing, so a request
        omitting the field was priced at 1.03 and reported at 1.05 -- a 200
        index point lie about where the barrier sat.

        The echo is now read off the priced SnowballTerms, so the two cannot
        drift apart again.
        """
        body = client.post("/api/cn_fdm", json=self.PAYLOAD).get_json()
        params = body["params"]

        assert params["ko_pct"] == 1.03
        assert params["ko_barrier"] == pytest.approx(5000 * 1.03)
        assert params["ko_barrier"] != pytest.approx(5000 * 1.05)

        # And an explicit value is honoured in both places.
        explicit = client.post("/api/cn_fdm", json={**self.PAYLOAD, "KO_pct": 1.08}).get_json()[
            "params"
        ]
        assert explicit["ko_pct"] == 1.08
        assert explicit["ko_barrier"] == pytest.approx(5000 * 1.08)

    def test_the_echo_finally_shows_what_was_always_priced(self, client) -> None:  # noqa: ANN001
        """lockup_months, ki_strike_mode and basis_pa reach a response at last.

        All three have been priced since Phase 1 and none appeared in this
        route's output, because the echo was rebuilt from the legacy field
        list rather than from the contract.
        """
        body = client.post(
            "/api/cn_fdm", json={**self.PAYLOAD, "lockup_months": 6, "basis_pa": 0.08}
        ).get_json()
        params = body["params"]

        assert params["lockup_months"] == 6
        assert params["ki_strike_mode"] == "s0"
        assert params["basis_pa"] == pytest.approx(0.08)
        # The carry is reported beside its parts, because the sum is what the
        # engines read.
        assert params["carry"] == pytest.approx(0.08 + 0.01)
        assert params["drift"] == pytest.approx(0.02 - 0.09)
        assert "re-parsed from the request" in params["echo_note"]

    def test_an_impossible_trade_is_a_client_error_not_a_crash(self, client) -> None:  # noqa: ANN001
        """It used to reach the blanket handler and come back as a 500."""
        response = client.post("/api/cn_fdm", json={**self.PAYLOAD, "KI_pct": 1.10})
        assert response.status_code == 400
        assert "knock-in" in response.get_json()["error"]

    def test_the_shark_fin_echo_describes_the_barrier_note(self, client) -> None:  # noqa: ANN001
        """The same treatment, from SharkFinTerms rather than SnowballTerms."""
        body = client.post(
            "/api/cn_fdm",
            json={
                "product_type": "shark_fin",
                "S0": 100,
                "T": 1.0,
                "r": 0.03,
                "sigma": 0.20,
                "K_strike": 100,
                "KO_pct": 1.20,
            },
        ).get_json()
        params = body["params"]

        assert params["barrier"] == pytest.approx(120.0)
        assert params["monitoring"] == 252
        assert params["is_continuous"] is False
        # The continuous closed form sits below the daily grid price, because
        # a discretely monitored knock-out is worth strictly more.
        assert body["haug_price"] < body["price"]


class TestSnowballRiskEndpoints:
    """The snowball Greeks API, backed by PDE grid differentiation."""

    PAYLOAD = {
        "S": 5000,
        "T": 1.0,
        "r": 0.02,
        "sigma": 0.20,
        "q": 0.01,
        "KO_pct": 1.03,
        "KI_pct": 0.75,
        "coupon_pa": 0.20,
    }

    @pytest.fixture(scope="class")
    def client(self):  # noqa: ANN201
        import app

        app.app.config.update(TESTING=True)
        return app.app.test_client()

    def test_returns_all_five_greeks_with_units(self, client) -> None:  # noqa: ANN001
        body = client.post("/api/snowball_risk", json=self.PAYLOAD).get_json()
        for greek in ("delta", "gamma", "vega", "theta", "rho"):
            assert greek in body, greek
            assert body[greek] is not None
        # Units stated explicitly: these conventions are not self-evident, and
        # a Greek read in the wrong one is worse than no Greek.
        assert body["units"]["vega"].startswith("per 1 percentage point")
        assert body["units"]["theta"] == "per calendar day"
        assert body["method"] == "pde-grid"

    def test_gamma_keeps_its_significant_figures(self, client) -> None:
        """Gamma is around 1e-07 as a fraction of notional, so the 8 decimals
        used for prices would leave it with two significant figures."""
        body = client.post("/api/snowball_risk", json=self.PAYLOAD).get_json()
        assert abs(body["gamma"]) > 1e-10
        assert f"{body['gamma']:.12f}".rstrip("0") != f"{body['gamma']:.2e}"

    def test_signs_match_the_economics(self, client) -> None:  # noqa: ANN001
        body = client.post("/api/snowball_risk", json=self.PAYLOAD).get_json()
        assert body["delta"] > 0.0  # net long the underlying on this trade
        assert body["vega"] < 0.0  # short volatility
        assert body["theta"] > 0.0  # accrues coupon carry

    def test_spot_only_and_fixing_only_agree_at_inception(self, client) -> None:  # noqa: ANN001
        """``S`` and ``S0`` are interchangeable for a newly struck trade.

        /api/snowball sends S and the legacy /api/cn_fdm sends S0, so both
        spellings have to work.
        """
        base = {k: v for k, v in self.PAYLOAD.items() if k != "S"}
        with_spot = client.post("/api/snowball_risk", json={**base, "S": 5000}).get_json()
        with_fixing = client.post("/api/snowball_risk", json={**base, "S0": 5000}).get_json()
        assert with_spot["price"] == pytest.approx(with_fixing["price"], rel=1e-12)
        assert with_spot["delta"] == pytest.approx(with_fixing["delta"], rel=1e-12)

    def test_seasoned_trade_is_valued_at_current_spot(self, client) -> None:  # noqa: ANN001
        """Barriers stay where they were struck while the market moves.

        This is the gate on the defect where the PDE read its price at the
        initial fixing and so returned the same value at every spot.
        """
        base = {k: v for k, v in self.PAYLOAD.items() if k != "S"}
        prices = [
            client.post("/api/snowball_risk", json={**base, "S0": 5000, "S": spot}).get_json()[
                "price"
            ]
            for spot in (4500, 5000, 5500)
        ]
        assert len(set(prices)) == 3, f"price insensitive to spot: {prices}"
        assert prices[0] < prices[1] < prices[2]

    def test_knocked_in_flag_selects_the_other_surface(self, client) -> None:  # noqa: ANN001
        clean = client.post("/api/snowball_risk", json=self.PAYLOAD).get_json()
        breached = client.post(
            "/api/snowball_risk", json={**self.PAYLOAD, "knocked_in": True}
        ).get_json()
        assert breached["knocked_in"] is True
        assert breached["price"] < clean["price"]
        assert breached["delta"] > clean["delta"]

    def test_crosscheck_agrees_with_a_measured_budget(self, client) -> None:  # noqa: ANN001
        body = client.post(
            "/api/snowball_risk_crosscheck", json={**self.PAYLOAD, "n_paths": 100_000}
        ).get_json()
        assert body["agree"] is True
        for name in ("delta", "gamma", "vega", "rho"):
            verdict = body["verdicts"][name]
            assert verdict["gap"] <= verdict["budget"]
            # Both error terms must be measured, not zero-filled.
            assert verdict["mc_std_error"] > 0.0
        assert body["pde"]["method"] == "pde-grid"
        assert body["mc"]["method"] == "mc-resimulation-crn"

    def test_crosscheck_notes_why_gamma_is_judged_loosely(self, client) -> None:  # noqa: ANN001
        body = client.post(
            "/api/snowball_risk_crosscheck", json={**self.PAYLOAD, "n_paths": 100_000}
        ).get_json()
        assert "note" in body["verdicts"]["gamma"]

    @pytest.mark.parametrize(
        "bad", [{"KI_pct": 1.2}, {"sigma": 0.0}, {"S": -1}, {"lockup_months": 24}]
    )
    def test_invalid_inputs_return_400(self, client, bad) -> None:  # noqa: ANN001
        response = client.post("/api/snowball_risk", json={**self.PAYLOAD, **bad})
        assert response.status_code == 400, response.get_json()


class TestCnFdmGreeksRouting:
    """/api/cn_fdm: every number now comes from a reviewed engine.

    The route was the last one serving legacy output. Its snowball Greeks went
    first, its shark fin followed in Phase 3c, and the snowball price on the
    ``calc_greeks=false`` branch was the final holdout -- retired here. The
    class docstring claimed the shark fin was still legacy long after it was
    not, which is its own small lesson about comments that describe intentions.
    """

    SNOWBALL = {
        "product_type": "snowball",
        "S0": 5000,
        "T": 1.0,
        "r": 0.02,
        "sigma": 0.20,
        "q": 0.01,
        "KO_pct": 1.03,
        "KI_pct": 0.75,
        "coupon_pa": 0.20,
        "mc_paths": 50_000,
        "calc_greeks": True,
    }

    @pytest.fixture(scope="class")
    def client(self):  # noqa: ANN201
        import app

        app.app.config.update(TESTING=True)
        return app.app.test_client()

    def test_snowball_greeks_come_from_the_new_engine(self, client) -> None:  # noqa: ANN001
        body = client.post("/api/cn_fdm", json=self.SNOWBALL).get_json()
        assert body["pde_grid"] is not None
        assert body["pde_grid"]["delta"] > 0.0

    def test_snowball_pde_grid_matches_the_risk_endpoint(self, client) -> None:  # noqa: ANN001
        """Same contract, same answer, whichever door it comes through."""
        from_cn_fdm = client.post("/api/cn_fdm", json=self.SNOWBALL).get_json()["pde_grid"]
        from_risk = client.post(
            "/api/snowball_risk",
            json={
                "S0": 5000,
                "T": 1.0,
                "r": 0.02,
                "sigma": 0.20,
                "q": 0.01,
                "KO_pct": 1.03,
                "KI_pct": 0.75,
                "coupon_pa": 0.20,
            },
        ).get_json()
        assert from_cn_fdm["delta"] == pytest.approx(from_risk["delta"], rel=1e-9)

    def test_pathwise_and_likelihood_are_null_with_reasons(self, client) -> None:  # noqa: ANN001
        """Withdrawn rather than quietly left wrong.

        Pathwise needs a Lipschitz payoff, which a snowball's is not. The
        likelihood ratio implementation carried a sign error, a terminal score
        used for a path-dependent payoff, and a missing discount term in rho.
        """
        body = client.post("/api/cn_fdm", json=self.SNOWBALL).get_json()
        assert body["pathwise"] is None
        assert body["likelihood"] is None
        notes = body["greek_notes"]
        assert "Lipschitz" in notes["pathwise"]
        assert notes["likelihood"]

    def test_shark_fin_is_now_on_the_barrier_engine(self, client) -> None:  # noqa: ANN001
        """Migrated, and the response says which contract it priced.

        The legacy grid ran at 252 observations a year, so daily monitoring is
        what it was actually pricing; that is preserved, but stated rather than
        implied. The request shape cannot express a monitoring frequency, which
        is precisely the ambiguity the migration removed -- on this case the
        daily note is worth 43.02 against 35.17 continuous, a 22% difference
        the old code silently conflated.

        The legacy three-method Greeks are gone rather than carried forward.
        They came from the same code whose snowball deltas were wrong in sign,
        and were never validated for a barrier at all, so returning them
        alongside a corrected price would lend them credibility they never had.
        """
        response = client.post(
            "/api/cn_fdm",
            json={
                "product_type": "shark_fin",
                "S0": 5500,
                "T": 0.25,
                "r": 0.02,
                "sigma": 0.2,
                "q": 0.01,
                "KO_pct": 1.1,
                "K_strike": 5500,
                "mc_paths": 20_000,
                "calc_greeks": True,
            },
        )
        assert response.status_code == 200
        body = response.get_json()

        assert body["pathwise"] is None
        assert "Migrated to engines/barrier.py" in body["greek_notes"]["engine"]
        assert "252 observations a year" in body["greek_notes"]["monitoring"]

        # The daily price sits above the continuous closed form, as it must.
        assert body["price"] > body["haug_price"] > 0.0
        assert body["price"] / body["haug_price"] == pytest.approx(1.223, rel=0.02)


class TestSnowballQuoteEndpoint:
    """The question a desk actually asks, which no route could answer.

    Every other snowball route takes ``coupon_pa`` as an input. The solver has
    existed and been validated since Phase 4 -- affinity residual 7.8e-16 --
    and had no caller outside its own test file, which means the project could
    price a quote but not produce one.
    """

    PAYLOAD = {
        "S": 5000,
        "T": 1.0,
        "KO_pct": 1.03,
        "KI_pct": 0.75,
        "r": 0.02,
        "sigma": 0.20,
        "q": 0.0,
    }

    @pytest.fixture(scope="class")
    def client(self):  # noqa: ANN201
        import app

        app.app.config.update(TESTING=True)
        return app.app.test_client()

    def test_it_quotes_the_measured_carry(self, client) -> None:  # noqa: ANN001
        """15.02% at the CSI 1000's measured carry, against 6.53% at none.

        Both at the endpoint's own 3200x2016 default, which is why these sit
        slightly below the 15.04%/6.54% measured at 1600x1008 in the solver's
        own tests -- the finer grid removes the 2.4e-04 upward bias rather
        than disagreeing with it.
        """
        body = client.post(
            "/api/snowball_quote",
            json={**self.PAYLOAD, "basis_pa": 0.1015, "verify_mc": False},
        ).get_json()
        assert body["par_coupon_pct"] == pytest.approx(15.016, abs=0.02)

        flat = client.post(
            "/api/snowball_quote",
            json={**self.PAYLOAD, "verify_mc": False, "decompose": False},
        ).get_json()
        assert flat["par_coupon_pct"] == pytest.approx(6.528, abs=0.02)
        assert body["par_coupon"] / flat["par_coupon"] == pytest.approx(2.30, abs=0.03)

    def test_the_affinity_invariant_is_reported_per_request(self, client) -> None:  # noqa: ANN001
        """Not a documented property -- a number measured on this call.

        The solver takes a third solve at the midpoint purely so the affinity
        it relies on is checked rather than assumed, and the endpoint passes
        that measurement through. A caller can see the quote is exact instead
        of being told so.
        """
        body = client.post(
            "/api/snowball_quote", json={**self.PAYLOAD, "verify_mc": False}
        ).get_json()
        assert body["affinity_residual"] < 1e-12
        assert abs(body["residual"]) < 1e-12
        assert "machine precision" in body["note"]

    def test_the_split_says_what_pays_for_the_coupon(self, client) -> None:  # noqa: ANN001
        """At the measured CSI 1000 carry the futures discount is the majority."""
        body = client.post(
            "/api/snowball_quote",
            json={**self.PAYLOAD, "basis_pa": 0.1015, "verify_mc": False},
        ).get_json()
        split = body["decomposition"]
        assert split["basis_share"] == pytest.approx(0.565, abs=0.01)
        assert split["volatility_part_pct"] == pytest.approx(6.528, abs=0.02)
        assert "interact through the drift" in split["note"]

    def test_the_simulation_confirms_the_solved_coupon(self, client) -> None:  # noqa: ANN001
        """Which is what establishes the finite-difference price itself.

        The solver is exact *given* the PDE price. Only a different engine
        repricing at the same coupon can say the PDE price was right, so the
        verification is on by default rather than opt-in.
        """
        body = client.post(
            "/api/snowball_quote", json={**self.PAYLOAD, "basis_pa": 0.1015}
        ).get_json()
        check = body["mc_verification"]
        assert check["passes"], check
        assert abs(check["z_score"]) < 3.0
        assert check["paths"] == 400000
        assert "independent engine" in check["note"]

    def test_the_expensive_pieces_can_be_declined(self, client) -> None:  # noqa: ANN001
        """A caller should pay for what it wants, and the grid is the default
        that costs the most: four times /api/snowball_pde's work."""
        body = client.post(
            "/api/snowball_quote",
            json={**self.PAYLOAD, "verify_mc": False, "decompose": False},
        ).get_json()
        assert body["mc_verification"] is None
        assert body["decomposition"] is None
        assert body["grid"]["space_nodes"] == 3200
        assert body["grid"]["steps_per_year"] == 2016

        coarse = client.post(
            "/api/snowball_quote",
            json={
                **self.PAYLOAD,
                "verify_mc": False,
                "decompose": False,
                "space_nodes": 800,
                "steps_per_year": 504,
            },
        ).get_json()
        assert coarse["grid"]["space_nodes"] == 800
        # And the coarse grid is measurably higher, as the note in the
        # response says it will be.
        assert coarse["par_coupon"] > body["par_coupon"]
        assert "Richardson limit" in coarse["grid"]["note"]

    def test_a_target_other_than_par_is_honoured(self, client) -> None:  # noqa: ANN001
        """A note sold above par carries less coupon."""
        at_par = client.post(
            "/api/snowball_quote",
            json={**self.PAYLOAD, "verify_mc": False, "decompose": False},
        ).get_json()["par_coupon"]
        above = client.post(
            "/api/snowball_quote",
            json={
                **self.PAYLOAD,
                "target": 0.01,
                "verify_mc": False,
                "decompose": False,
            },
        ).get_json()["par_coupon"]
        assert above > at_par

    def test_a_knocked_in_note_can_still_be_quoted(self, client) -> None:  # noqa: ANN001
        """The operational case: a breached note being restruck."""
        body = client.post(
            "/api/snowball_quote",
            json={
                **self.PAYLOAD,
                "knocked_in": True,
                "verify_mc": False,
                "decompose": False,
            },
        ).get_json()
        assert body["par_coupon"] > 0.0
        assert body["affinity_residual"] < 1e-12

    def test_the_echo_shows_the_contract_that_was_solved(self, client) -> None:  # noqa: ANN001
        body = client.post(
            "/api/snowball_quote",
            json={**self.PAYLOAD, "basis_pa": 0.1015, "verify_mc": False},
        ).get_json()
        params = body["params"]
        assert params["ko_barrier"] == pytest.approx(5150.0)
        assert params["carry"] == pytest.approx(0.1015)
        assert params["lockup_months"] == 3

    def test_an_unquotable_note_is_refused_with_a_reason(self, client) -> None:  # noqa: ANN001
        """A note that can neither knock out nor survive clean has no coupon
        that reaches par, and dividing by its annuity would hand back inf."""
        response = client.post(
            "/api/snowball_quote",
            json={
                **self.PAYLOAD,
                "KO_pct": 5.0,
                "KI_pct": 4.0,
                "verify_mc": False,
                "decompose": False,
            },
        )
        assert response.status_code == 400
        assert "error" in response.get_json()

    def test_impossible_terms_are_a_client_error(self, client) -> None:  # noqa: ANN001
        response = client.post("/api/snowball_quote", json={**self.PAYLOAD, "KI_pct": 1.10})
        assert response.status_code == 400
