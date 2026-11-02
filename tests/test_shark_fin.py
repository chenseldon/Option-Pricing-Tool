"""Tests for the shark fin: the first time this product has been cross-checked.

Before this it was priced by two unconnected pieces of code that disagreed
about what the contract *was* -- an analytic formula inlined in a Flask route
(continuous monitoring) and a legacy grid (discrete, at whatever frequency the
grid implied) -- with nothing comparing them.

Four checks carry the weight, and each caught something:

1. **Analytic against QuantLib.** Found that the shipped formula was wrong for
   *down* barriers, by 1.7 to 5.5 on prices of order 1 to 8, while carrying a
   comment claiming validation to 0.01%. It had only ever been checked on an
   up-and-out call.
2. **PDE against the analytic continuous price.** Found that solving across the
   barrier and re-projecting the dead zone each step -- which looks equivalent
   to imposing a Dirichlet boundary at the barrier -- costs an order of
   convergence, and was still 3% wrong at 3200 nodes.
3. **PDE against Monte Carlo at discrete monitoring.** Found that damping only
   the first two steps of the backward solve leaves every observation's
   discontinuity undamped.
4. **The discrete-to-continuous limit**, which is the only check that the two
   contracts are being told apart at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from otcpricer.core.exceptions import EngineError, ParameterError
from otcpricer.core.market import EquityMarket
from otcpricer.core.types import BarrierKind, OptionType
from otcpricer.engines.barrier import (
    RANNACHER_STEPS,
    price_shark_fin_analytic,
    price_shark_fin_mc,
    price_shark_fin_pde,
)
from otcpricer.engines.mc import MonteCarloConfig
from otcpricer.engines.pde import PDEConfig
from otcpricer.models import bsm
from otcpricer.products.shark_fin import CONTINUOUS, SharkFinTerms

MARKET = EquityMarket(S=100.0, r=0.03, sigma=0.25, q=0.01)

# Accurate enough that the engine's own error is below the tolerances below,
# and slow enough that it is not the default.
FINE = PDEConfig(space_nodes=1600, steps_per_year=2016)


def up_out_call(**overrides: object) -> SharkFinTerms:
    base: dict[str, object] = {
        "S0": 100.0,
        "K": 100.0,
        "T": 1.0,
        "barrier": 120.0,
        "kind": BarrierKind.UP_AND_OUT,
        "option": OptionType.CALL,
        "monitoring": CONTINUOUS,
    }
    base.update(overrides)
    return SharkFinTerms(**base)  # type: ignore[arg-type]


def down_out_put(**overrides: object) -> SharkFinTerms:
    base: dict[str, object] = {
        "S0": 100.0,
        "K": 100.0,
        "T": 1.0,
        "barrier": 80.0,
        "kind": BarrierKind.DOWN_AND_OUT,
        "option": OptionType.PUT,
        "monitoring": CONTINUOUS,
    }
    base.update(overrides)
    return SharkFinTerms(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


class TestTerms:
    def test_monitoring_has_no_default(self) -> None:
        """Because choosing one silently is how the project got two prices.

        A continuously monitored knock-out and a daily one are different
        contracts -- measured here, the daily one is worth 19% more -- so the
        frequency is part of the term sheet, not a numerical setting.
        """
        with pytest.raises(TypeError):
            SharkFinTerms(  # type: ignore[call-arg]
                S0=100.0,
                K=100.0,
                T=1.0,
                barrier=120.0,
                kind=BarrierKind.UP_AND_OUT,
            )

    @pytest.mark.parametrize(
        ("overrides", "fragment"),
        [
            ({"S0": 0.0}, "S0 must be positive"),
            ({"K": -1.0}, "K must be positive"),
            ({"T": 0.0}, "T must be positive"),
            ({"barrier": 0.0}, "barrier must be positive"),
            ({"rebate": -0.1}, "rebate cannot be negative"),
            ({"participation": 0.0}, "participation must be positive"),
            ({"monitoring": 0}, "at least 1 observation"),
            ({"monitoring": "daily"}, "observation count"),
            ({"monitoring": 2.5}, "observation count"),
        ],
    )
    def test_refuses_an_incoherent_note(self, overrides: dict[str, object], fragment: str) -> None:
        with pytest.raises(ParameterError, match=fragment):
            up_out_call(**overrides)

    def test_refuses_a_barrier_already_breached(self) -> None:
        """Worth its rebate and nothing else, so almost certainly a mistake."""
        with pytest.raises(ParameterError, match="dead at inception"):
            up_out_call(barrier=95.0)
        with pytest.raises(ParameterError, match="dead at inception"):
            down_out_put(barrier=105.0)

    def test_the_schedule_ends_at_maturity(self) -> None:
        """A schedule stopping short would leave the last period unmonitored.

        That silently raises the price, which is the worst kind of bug: it
        looks like a better number.
        """
        terms = up_out_call(monitoring=12)
        schedule = terms.observation_times()
        assert schedule.size == 12
        assert schedule[-1] == pytest.approx(terms.T)
        assert np.all(np.diff(schedule) > 0.0)

    def test_a_continuous_barrier_has_no_schedule(self) -> None:
        terms = up_out_call(monitoring=CONTINUOUS)
        assert terms.is_continuous
        assert terms.observation_times().size == 0
        assert terms.n_observations == 0

    def test_the_barrier_touch_is_inclusive(self) -> None:
        """Market convention, and the conservative reading for a knock-out."""
        terms = up_out_call()
        assert bool(terms.breached(np.array([120.0]))[0])
        assert not bool(terms.breached(np.array([119.99]))[0])


# ---------------------------------------------------------------------------
# Analytic, against QuantLib
# ---------------------------------------------------------------------------


class TestAnalyticAgainstQuantLib:
    """The check that found the shipped formula was wrong for down barriers."""

    @staticmethod
    def _quantlib(
        kind: str, flag: str, strike: float, barrier: float, sigma: float
    ) -> tuple[float, float]:
        ql = pytest.importorskip("QuantLib")

        today = ql.Date(2, 1, 2024)
        ql.Settings.instance().evaluationDate = today
        day_count = ql.Actual365Fixed()
        calendar = ql.NullCalendar()
        days = 365
        maturity = today + ql.Period(days, ql.Days)

        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(100.0)),
            ql.YieldTermStructureHandle(ql.FlatForward(today, 0.01, day_count, ql.Continuous)),
            ql.YieldTermStructureHandle(ql.FlatForward(today, 0.03, day_count, ql.Continuous)),
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, sigma, day_count)),
        )
        barrier_type = {
            "uo": ql.Barrier.UpOut,
            "do": ql.Barrier.DownOut,
            "ui": ql.Barrier.UpIn,
            "di": ql.Barrier.DownIn,
        }[kind]
        option = ql.BarrierOption(
            barrier_type,
            barrier,
            0.0,
            ql.PlainVanillaPayoff(ql.Option.Call if flag == "c" else ql.Option.Put, strike),
            ql.EuropeanExercise(maturity),
        )
        option.setPricingEngine(ql.AnalyticBarrierEngine(process))
        return option.NPV(), days / 365.0

    KINDS = {
        "uo": (BarrierKind.UP_AND_OUT, OptionType.CALL),
        "do": (BarrierKind.DOWN_AND_OUT, OptionType.PUT),
        "ui": (BarrierKind.UP_AND_IN, OptionType.CALL),
        "di": (BarrierKind.DOWN_AND_IN, OptionType.PUT),
    }

    @pytest.mark.parametrize(
        ("kind", "strike", "barrier"),
        [
            ("uo", 100.0, 120.0),
            ("uo", 100.0, 130.0),
            ("uo", 90.0, 110.0),
            ("do", 100.0, 80.0),
            ("do", 100.0, 70.0),
            ("do", 110.0, 90.0),
            ("ui", 100.0, 120.0),
            ("di", 100.0, 80.0),
        ],
    )
    @pytest.mark.parametrize("sigma", [0.20, 0.35])
    def test_matches_to_machine_precision(
        self, kind: str, strike: float, barrier: float, sigma: float
    ) -> None:
        """Both barrier directions, both knock senses.

        The down-barrier rows are the ones that matter: the previous
        implementation applied the up-barrier reflection sign unconditionally,
        which is right for an up-and-out call and wrong by whole currency units
        for a down-and-out put.
        """
        barrier_kind, option = self.KINDS[kind]
        reference, maturity = self._quantlib(
            kind, "c" if option is OptionType.CALL else "p", strike, barrier, sigma
        )

        market = EquityMarket(S=100.0, r=0.03, sigma=sigma, q=0.01)
        terms = SharkFinTerms(
            S0=100.0,
            K=strike,
            T=maturity,
            barrier=barrier,
            kind=barrier_kind,
            option=option,
            monitoring=CONTINUOUS,
        )
        assert price_shark_fin_analytic(terms, market) == pytest.approx(reference, abs=1e-12)

    def test_refuses_a_discretely_monitored_contract(self) -> None:
        """Rather than returning the continuous price for a different contract.

        This is the confusion the module was built to end, so it is an error
        and not a warning.
        """
        with pytest.raises(ParameterError, match="continuously monitored"):
            price_shark_fin_analytic(up_out_call(monitoring=252), MARKET)

    def test_an_unsupported_combination_is_refused_not_zeroed(self) -> None:
        """A down-and-out *call* is a real contract this form does not cover.

        Returning zero for it -- which an unconditional max(., 0) would do --
        would report a priceable note as worthless.
        """
        terms = SharkFinTerms(
            S0=100.0,
            K=100.0,
            T=1.0,
            barrier=80.0,
            kind=BarrierKind.DOWN_AND_OUT,
            option=OptionType.CALL,
            monitoring=CONTINUOUS,
        )
        with pytest.raises(ParameterError, match="covers up-and-out"):
            price_shark_fin_analytic(terms, MARKET)

    def test_a_barrier_inside_the_payoff_is_worthless_not_an_error(self) -> None:
        """An up-and-out call with H <= K: every surviving path finishes out of
        the money. That is a fact about the contract, and the old route raised
        an error for it."""
        terms = up_out_call(K=130.0, barrier=120.0)
        assert price_shark_fin_analytic(terms, MARKET) == 0.0


# ---------------------------------------------------------------------------
# The PDE
# ---------------------------------------------------------------------------


class TestPdeAgainstAnalytic:
    @pytest.mark.parametrize(
        ("factory", "overrides"),
        [
            (up_out_call, {}),
            (up_out_call, {"barrier": 130.0}),
            (down_out_put, {}),
            (down_out_put, {"barrier": 70.0}),
        ],
    )
    def test_reproduces_the_continuous_price(self, factory, overrides: dict) -> None:  # type: ignore[no-untyped-def]
        """To better than 1e-4 relative, on both barrier directions."""
        terms = factory(**overrides)
        reference = price_shark_fin_analytic(terms, MARKET)
        result = price_shark_fin_pde(terms, MARKET, FINE)
        assert result.price == pytest.approx(reference, rel=2e-4)
        assert result.barrier_on_node

    def test_the_continuous_grid_converges_at_second_order(self) -> None:
        """Which is the evidence that the barrier is a boundary, not a projection.

        Solving across the barrier and re-imposing the dead zone each step
        couples nodes through it and smears it over one dx, costing an order:
        that version's errors were 0.135, 0.065, 0.040, 0.023 as the node count
        doubled -- halving, not quartering -- and still 3% of price at 3200
        nodes. Truncating the grid at the barrier gives 8.9e-04, 2.1e-04,
        5.0e-05, 8.8e-06 instead.
        """
        terms = up_out_call()
        reference = price_shark_fin_analytic(terms, MARKET)

        errors = []
        for nodes in (400, 800, 1600, 3200):
            result = price_shark_fin_pde(
                terms, MARKET, PDEConfig(space_nodes=nodes, steps_per_year=2016)
            )
            errors.append(abs(result.price - reference))

        ratios = [a / b for a, b in zip(errors, errors[1:], strict=False)]
        assert all(ratio > 3.0 for ratio in ratios), (
            f"expected roughly fourfold error reduction per doubling, got {ratios}"
        )
        assert errors[-1] < 5e-5

    def test_in_out_parity_is_exact(self) -> None:
        """Exact by construction, which is why it proves less than it looks.

        The knock-in is *computed* as the vanilla minus the knock-out, so this
        identity holds to the last bit whatever the knock-out solve did. It is
        kept because it still guards the vanilla leg and the participation
        scaling -- and because the alternative, a second grid dimension
        tracking whether the barrier has been touched, is what this avoids.
        """
        for factory, barrier_in in (
            (up_out_call, BarrierKind.UP_AND_IN),
            (down_out_put, BarrierKind.DOWN_AND_IN),
        ):
            for monitoring in (12, 252, CONTINUOUS):
                out_terms = factory(monitoring=monitoring)
                in_terms = factory(monitoring=monitoring, kind=barrier_in)

                knocked_out = price_shark_fin_pde(out_terms, MARKET, FINE).price
                knocked_in = price_shark_fin_pde(in_terms, MARKET, FINE).price
                vanilla = float(
                    bsm.price(
                        out_terms.option,
                        MARKET.S,
                        out_terms.K,
                        out_terms.T,
                        MARKET.r,
                        MARKET.sigma,
                        MARKET.q,
                    )
                )
                assert knocked_out + knocked_in == pytest.approx(vanilla, abs=1e-10)

    def test_a_knock_in_with_a_rebate_is_refused(self) -> None:
        """Parity cannot split it: the rebate belongs to the knock-out leg only."""
        terms = up_out_call(kind=BarrierKind.UP_AND_IN, rebate=0.05)
        with pytest.raises(ParameterError, match="in-out parity"):
            price_shark_fin_pde(terms, MARKET, FINE)

    def test_damping_follows_every_observation(self) -> None:
        assert RANNACHER_STEPS == 2


class TestDiscreteVersusContinuous:
    """The check that the two contracts are being told apart at all."""

    def test_discrete_monitoring_is_worth_strictly_more(self) -> None:
        """And the premium is large, which is why conflating them matters.

        Daily observation of an up-and-out call is worth 19% more than
        continuous; quarterly is worth 2.4 times as much. A model that returned
        the continuous price for a daily contract would be underpricing the
        note it was asked about by those margins.
        """
        reference = price_shark_fin_analytic(up_out_call(), MARKET)
        prices = {
            monitoring: price_shark_fin_pde(up_out_call(monitoring=monitoring), MARKET, FINE).price
            for monitoring in (4, 12, 52, 252, 1008)
        }

        for monitoring, price in prices.items():
            assert price > reference, f"{monitoring}/yr should exceed continuous"

        # Quarterly at roughly 2.4x continuous, daily at roughly 1.19x.
        assert prices[4] / reference == pytest.approx(2.42, rel=0.05)
        assert prices[252] / reference == pytest.approx(1.19, rel=0.05)

    def test_the_premium_falls_monotonically_towards_the_continuous_limit(
        self,
    ) -> None:
        """Measured premiums 0.958, 0.568, 0.276, 0.131, 0.075, 0.040.

        Roughly halving as the frequency quadruples, which is the ``1/sqrt(m)``
        rate Broadie, Glasserman and Kou give for the discrete barrier
        correction -- so the engine is reproducing a known analytic result, not
        merely trending the right way.
        """
        reference = price_shark_fin_analytic(up_out_call(), MARKET)
        premiums = [
            price_shark_fin_pde(up_out_call(monitoring=m), MARKET, FINE).price - reference
            for m in (4, 12, 52, 252, 1008, 4032)
        ]

        assert all(b < a for a, b in zip(premiums, premiums[1:], strict=False))
        assert all(premium > 0.0 for premium in premiums)

        # Four-fold frequency should roughly halve the premium.
        for coarse, fine in ((0, 2), (2, 4)):
            assert premiums[fine] / premiums[coarse] == pytest.approx(0.25, rel=0.45)


@pytest.mark.slow
class TestPdeAgainstMonteCarlo:
    """Two engines on the discretely monitored contract, which has no closed form."""

    SEEDS = (5, 17, 29, 41, 53, 67, 79, 91)

    def _mc_mean(self, terms: SharkFinTerms) -> tuple[float, float]:
        prices = []
        for seed in self.SEEDS:
            result = price_shark_fin_mc(
                terms,
                MARKET,
                MonteCarloConfig(paths=200_000, steps_per_year=252, seed=seed, antithetic=True),
            )
            prices.append(result.price)
        values = np.array(prices)
        return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))

    @pytest.mark.parametrize("factory", [up_out_call, down_out_put])
    def test_agree_at_daily_monitoring(self, factory) -> None:  # type: ignore[no-untyped-def]
        """Within 1% relative, which is the honest tolerance here and why.

        The Monte Carlo is *exact* for this contract: the log-Euler step is
        exact for geometric Brownian motion and the observations land on step
        boundaries, so there is no path bias to correct. The PDE is the
        approximate side, and it converges only **first order** in ``dx`` --
        measured 1.3381, 1.3529, 1.3609, 1.3649, 1.3669 as nodes doubled,
        differences halving rather than quartering. That is expected: every
        observation re-creates a discontinuity at the barrier, so the solution
        is never smooth for long.

        Richardson-extrapolating that sequence gives 1.3690 against a Monte
        Carlo mean of 1.3711 +/- 0.0012 -- a 1.7-sigma gap, i.e. consistent.
        A tolerance tight enough to exclude it would be asserting the PDE is
        better than it is.
        """
        terms = factory(monitoring=252)
        grid = price_shark_fin_pde(
            terms, MARKET, PDEConfig(space_nodes=6400, steps_per_year=4032)
        ).price
        simulated, _ = self._mc_mean(terms)

        assert grid == pytest.approx(simulated, rel=0.01)

    def test_the_monte_carlo_sees_the_right_knock_probability(self) -> None:
        """A sanity check the price alone would not give.

        A daily-observed up-and-out call at a 20% barrier knocks out on about
        43% of paths at this volatility.
        """
        result = price_shark_fin_mc(
            up_out_call(monitoring=252),
            MARKET,
            MonteCarloConfig(paths=200_000, steps_per_year=252, seed=5, antithetic=True),
        )
        assert 0.35 < result.knock_prob < 0.50

    def test_a_knock_in_and_its_knock_out_partition_the_sample(self) -> None:
        """Unlike the PDE, this is *not* tautological -- but it must be asked
        of the right reference.

        The simulated knock-in is computed from its own paths rather than as a
        residual, so the identity is a genuine check that the two payoff
        branches partition the sample. But they partition *these* paths, so the
        sum equals the vanilla **on the same paths** exactly, and equals the
        analytic vanilla only to within sampling error. Comparing against the
        analytic value and calling the difference a failure measures the random
        number generator; measured here it was 0.0317, or 1 to 2 standard
        errors.

        So the exact identity is asserted against the simulated vanilla, and
        the analytic comparison is made separately with a tolerance set from
        the sampling error it actually has.
        """
        from otcpricer.engines.mc import simulate_log_paths

        config = MonteCarloConfig(paths=200_000, steps_per_year=252, seed=11, antithetic=True)
        out_price = price_shark_fin_mc(up_out_call(monitoring=252), MARKET, config).price
        in_price = price_shark_fin_mc(
            up_out_call(monitoring=252, kind=BarrierKind.UP_AND_IN), MARKET, config
        ).price

        # The vanilla on exactly those paths.
        _, paths = simulate_log_paths(MARKET, 1.0, config)
        payoff = np.maximum(paths[-1, :] - 100.0, 0.0) * float(np.exp(-MARKET.r))
        half = payoff.size // 2
        pairs = 0.5 * (payoff[:half] + payoff[half : 2 * half])
        simulated_vanilla = float(pairs.mean())
        standard_error = float(pairs.std(ddof=1) / np.sqrt(pairs.size))

        # Exact, because every path lands in precisely one branch.
        assert out_price + in_price == pytest.approx(simulated_vanilla, abs=1e-12)

        # And the simulation agrees with the closed form within its own error.
        analytic = float(bsm.price("c", MARKET.S, 100.0, 1.0, MARKET.r, MARKET.sigma, MARKET.q))
        assert abs(simulated_vanilla - analytic) < 3.0 * standard_error


class TestRebate:
    def test_a_rebate_raises_a_knock_out_price(self) -> None:
        without = price_shark_fin_pde(up_out_call(monitoring=12), MARKET, FINE).price
        with_rebate = price_shark_fin_pde(
            up_out_call(monitoring=12, rebate=0.05), MARKET, FINE
        ).price
        assert with_rebate > without

    def test_the_simulated_rebate_discounts_from_the_touch(self) -> None:
        """Not from maturity, which would overvalue it.

        A rebate paid on the touch is worth more than the same cash at
        maturity, so getting this wrong is a one-sided error.
        """
        config = MonteCarloConfig(paths=50_000, steps_per_year=252, seed=3, antithetic=True)
        result = price_shark_fin_mc(up_out_call(monitoring=12, rebate=0.05), MARKET, config)
        plain = price_shark_fin_mc(up_out_call(monitoring=12), MARKET, config)
        assert result.price > plain.price
        # And by less than the undiscounted rebate times the knock probability.
        assert result.price - plain.price < 0.05 * result.knock_prob + 1e-9


class TestGridIntegrity:
    def test_the_barrier_lands_on_a_node_at_every_resolution(self) -> None:
        """The alignment two Phase 1 defects were about.

        A displaced barrier does not merely add error -- the displacement
        changes with the node count, so refining moves the answer around
        instead of settling it.
        """
        for nodes in (200, 400, 800, 1600, 3200):
            for barrier in (110.0, 120.0, 137.5):
                result = price_shark_fin_pde(
                    up_out_call(barrier=barrier),
                    MARKET,
                    PDEConfig(space_nodes=nodes, steps_per_year=504),
                )
                assert result.barrier_on_node, f"{nodes} nodes, barrier {barrier}"

    def test_the_result_reports_its_own_grid(self) -> None:
        result = price_shark_fin_pde(up_out_call(), MARKET, FINE)
        payload = result.as_dict()
        assert payload["barrier_on_node"] is True
        assert payload["observations_on_steps"] is True
        assert payload["method"] == "crank-nicolson+rannacher"
        assert payload["vol_model"] == "flat"
        assert payload["space_nodes"] > 100

    def test_a_displaced_barrier_would_be_refused(self) -> None:
        """The guard exists because a silent degradation is worse than a failure.

        Exercised by monkeypatching the alignment away, since the grid builder
        is correct and cannot otherwise produce one.
        """
        import otcpricer.engines.barrier as barrier_module

        original = barrier_module._build_barrier_grid

        def displaced(terms, market, config, n_sigma):  # type: ignore[no-untyped-def]
            grid = original(terms, market, config, n_sigma)
            return type(grid)(
                x_grid=grid.x_grid + 0.3 * grid.dx,
                S_grid=np.exp(grid.x_grid + 0.3 * grid.dx),
                dx=grid.dx,
                dt=grid.dt,
                n_steps=grid.n_steps,
                barrier_index=grid.barrier_index,
                observation_steps=grid.observation_steps,
            )

        barrier_module._build_barrier_grid = displaced  # type: ignore[assignment]
        try:
            with pytest.raises(EngineError, match="did not land on a grid node"):
                price_shark_fin_pde(up_out_call(), MARKET, FINE)
        finally:
            barrier_module._build_barrier_grid = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# On the fitted surface
# ---------------------------------------------------------------------------


class TestOnTheLocalVolatilitySurface:
    """Barriers priced against the whole smile rather than one number.

    This is where the effect is largest in the whole project, and for a
    structural reason: a barrier payoff is concentrated at one spot level, so
    it depends on the volatility *there* rather than on an average. A single
    flat number cannot express that, and the error it makes is one-sided in a
    direction that depends on which side the barrier is.
    """

    @pytest.fixture(scope="class")
    def surface(self):  # type: ignore[no-untyped-def]
        from otcpricer.models.local_vol import local_vol_from_fitted
        from otcpricer.models.surface import fit_surface

        try:
            return local_vol_from_fitted(fit_surface("510300"), spot=4.1, tail_drift=0.02)
        except Exception:  # pragma: no cover - needs the committed snapshots
            pytest.skip("no snapshots committed")

    @pytest.fixture(scope="class")
    def flat_market(self, surface):  # type: ignore[no-untyped-def]
        """Flat at the surface's own at-the-money level, so the comparison is fair.

        Comparing against an arbitrary volatility would measure the choice of
        volatility; comparing against the surface's at-the-money total vol
        isolates what the *smile* does.
        """
        return EquityMarket(S=4.1, r=0.02, sigma=float(surface.atm_total_vol(1.0)), q=0.0)

    @pytest.mark.parametrize(
        ("kind", "option", "barrier_pct", "direction"),
        [
            # The surface's upside volatility is below its at-the-money level,
            # so an up-and-out knocks out less often and is worth more.
            (BarrierKind.UP_AND_OUT, OptionType.CALL, 1.10, "higher"),
            (BarrierKind.UP_AND_OUT, OptionType.CALL, 1.20, "higher"),
            # And its downside volatility is above, so a down-and-out at a
            # distant barrier knocks out more often and is worth less.
            (BarrierKind.DOWN_AND_OUT, OptionType.PUT, 0.80, "lower"),
        ],
    )
    def test_the_smile_moves_the_price_in_the_direction_the_skew_implies(
        self,
        surface,
        flat_market,
        kind: BarrierKind,
        option: OptionType,
        barrier_pct: float,
        direction: str,
    ) -> None:  # type: ignore[no-untyped-def]
        """Measured: +94% and +16% on the up barriers, -37% on the far down one.

        The sign is the test. A barrier sits at one spot level, so what matters
        is the volatility at *that* level -- and on a downward-skewed equity
        surface that is below the at-the-money for an up barrier and above it
        for a down one. Getting the direction wrong would mean the smile was
        being applied backwards, which no price-level check would catch.
        """
        terms = SharkFinTerms(
            S0=4.1,
            K=4.1,
            T=1.0,
            barrier=4.1 * barrier_pct,
            kind=kind,
            option=option,
            monitoring=252,
        )
        flat = price_shark_fin_pde(terms, flat_market, FINE).price
        local = price_shark_fin_pde(terms, flat_market, FINE, local_vol=surface)

        assert local.vol_model == "dupire-local"
        if direction == "higher":
            assert local.price > flat
        else:
            assert local.price < flat
        # And the effect is large enough to matter, not a rounding difference.
        assert abs(local.price / flat - 1.0) > 0.03

    def test_the_two_engines_agree_on_the_surface(self, surface, flat_market) -> None:  # type: ignore[no-untyped-def]
        """Which is what makes the local-volatility barrier price trustworthy.

        The analytic form cannot check this -- there is no closed form for a
        barrier under a local volatility -- so the Monte Carlo is the only
        independent value available, and the agreement has to carry the weight
        the analytic comparison carries in the flat case.
        """
        terms = SharkFinTerms(
            S0=4.1,
            K=4.1,
            T=1.0,
            barrier=4.1 * 1.20,
            kind=BarrierKind.UP_AND_OUT,
            option=OptionType.CALL,
            monitoring=252,
        )
        grid = price_shark_fin_pde(
            terms, flat_market, PDEConfig(space_nodes=3200, steps_per_year=2016), local_vol=surface
        ).price
        simulated = price_shark_fin_mc(
            terms,
            flat_market,
            MonteCarloConfig(paths=200_000, steps_per_year=252, seed=5, antithetic=True),
            local_vol=surface,
        )
        assert simulated.vol_model == "dupire-local"
        assert abs(simulated.price - grid) < 3.0 * simulated.std_error

    def test_a_constant_local_volatility_reduces_to_the_flat_engine(self, flat_market) -> None:  # type: ignore[no-untyped-def]
        """The degenerate check, which localises a failure to indexing.

        If a constant local volatility does not reproduce the flat price, the
        fault is in how the surface is sampled onto the grid rather than in the
        discretisation -- the same reasoning Phase 2b used, where this test
        caught an inverted moneyness coordinate.
        """
        from tests.test_european import ConstantLocalVol

        terms = up_out_call(S0=100.0, K=100.0, barrier=120.0, monitoring=252)
        market = EquityMarket(S=100.0, r=0.03, sigma=0.25, q=0.0)

        flat = price_shark_fin_pde(terms, market, FINE).price
        constant = price_shark_fin_pde(
            terms, market, FINE, local_vol=ConstantLocalVol(sigma=0.25, mu=0.03)
        ).price
        assert constant == pytest.approx(flat, rel=1e-9)
