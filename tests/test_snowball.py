"""Tests for the snowball engines, and the gates on the defects they fixed.

The central test here is :class:`TestEngineConsistency`: the PDE and the Monte
Carlo are independent implementations -- a different discretisation of a
different formulation -- so their agreement is real evidence rather than a
tautology. The project previously advertised that cross-validation while the
two engines were valuing different products, 18% apart. These tests exist so
that cannot recur silently.

Each defect gets a gate that would fail if the fix were reverted, plus, where
one exists, a check against a closed-form benchmark so correctness does not
rest on the two engines merely agreeing with each other.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtr

from otcpricer.core.exceptions import EngineError, ParameterError
from otcpricer.core.market import EquityMarket
from otcpricer.core.types import KIStrikeMode
from otcpricer.engines.mc import MonteCarloConfig, price_snowball_mc, simulate_log_paths
from otcpricer.engines.pde import PDEConfig, price_snowball_pde, solve_tridiagonal
from otcpricer.products.snowball import SnowballTerms

# The project's reference trade: CSI-500-like level, one year, 103/75 barriers,
# 20% coupon -- the same case whose legacy mispricing is documented above.
REFERENCE_MARKET = EquityMarket(S=5000.0, r=0.02, sigma=0.20, q=0.01)


def reference_terms(**overrides: object) -> SnowballTerms:
    """Reference trade, with any field overridden."""
    base: dict[str, object] = {
        "S0": 5000.0,
        "T": 1.0,
        "ko_pct": 1.03,
        "ki_pct": 0.75,
        "coupon_pa": 0.20,
    }
    base.update(overrides)
    return SnowballTerms(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Contract terms and schedules
# ---------------------------------------------------------------------------


class TestObservationSchedule:
    def test_lockup_removes_early_knockout_dates(self) -> None:
        """Three months of lock-up on a one-year note leaves nine dates, not twelve."""
        assert reference_terms(lockup_months=0).ko_observation_times().size == 12
        assert reference_terms(lockup_months=3).ko_observation_times().size == 9
        assert reference_terms(lockup_months=6).ko_observation_times().size == 6

    def test_lockup_dates_all_fall_after_the_window(self) -> None:
        terms = reference_terms(lockup_months=3)
        assert np.all(terms.ko_observation_times() > terms.lockup_years + 1e-12)

    def test_knockin_is_monitored_through_the_lockup(self) -> None:
        """The lock-up suspends early termination, not downside monitoring."""
        terms = reference_terms(lockup_months=3)
        ki_times = terms.ki_observation_times()
        assert ki_times.size == 252
        assert np.any(ki_times < terms.lockup_years)

    def test_schedules_are_sorted_and_within_tenor(self) -> None:
        terms = reference_terms(lockup_months=3)
        for schedule in (terms.ko_observation_times(), terms.ki_observation_times()):
            assert np.all(np.diff(schedule) > 0)
            assert schedule[-1] <= terms.T + 1e-12

    def test_final_observation_lands_on_maturity(self) -> None:
        assert reference_terms().ko_observation_times()[-1] == pytest.approx(1.0)
        assert reference_terms().ki_observation_times()[-1] == pytest.approx(1.0)


class TestKnockInStrikeMode:
    """Defect gate: the two engines disagreed on this strike.

    The legacy Monte Carlo struck the knock-in put at the barrier while the
    legacy PDE struck it at the initial level. Both now read it from one place.
    """

    def test_default_is_the_market_convention(self) -> None:
        assert reference_terms().ki_strike_mode is KIStrikeMode.S0
        assert reference_terms().ki_put_strike == pytest.approx(5000.0)

    def test_barrier_mode_available_for_variant_products(self) -> None:
        terms = reference_terms(ki_strike_mode=KIStrikeMode.KI)
        assert terms.ki_put_strike == pytest.approx(3750.0)

    def test_barrier_strike_caps_the_loss_relative_to_s0_strike(self) -> None:
        """Striking at the barrier is strictly less painful, so worth more."""
        s0_loss = reference_terms().knock_in_payoff(3000.0)
        ki_loss = reference_terms(ki_strike_mode=KIStrikeMode.KI).knock_in_payoff(3000.0)
        assert s0_loss < ki_loss < 0.0

    def test_strike_mode_moves_the_price_in_both_engines(self) -> None:
        """If an engine ignored the setting, its price would not move."""
        config = MonteCarloConfig(paths=200_000)
        mc_s0 = price_snowball_mc(reference_terms(), REFERENCE_MARKET, config).price
        mc_ki = price_snowball_mc(
            reference_terms(ki_strike_mode=KIStrikeMode.KI), REFERENCE_MARKET, config
        ).price
        pde_s0 = price_snowball_pde(reference_terms(), REFERENCE_MARKET).price
        pde_ki = price_snowball_pde(
            reference_terms(ki_strike_mode=KIStrikeMode.KI), REFERENCE_MARKET
        ).price
        assert mc_ki > mc_s0
        assert pde_ki > pde_s0


# ---------------------------------------------------------------------------
# Defect gates
# ---------------------------------------------------------------------------


class TestDiscountingAtPaymentDate:
    """Defect gate: the legacy MC discounted every path from maturity.

    A knocked-out note pays at its knock-out date, and most paths knock out
    well before maturity, so discounting them from ``T`` understated the note.
    """

    @staticmethod
    def _price_both_ways(market: EquityMarket, terms: SnowballTerms) -> tuple[float, float]:
        """Return ``(correct, legacy)`` prices from one shared set of paths.

        Both are built from the *same* payoff and state vectors, differing only
        in the discount factor applied -- per-path payment date versus a single
        factor from maturity. That isolates the defect exactly: any difference
        is attributable to discounting and to nothing else.
        """
        config = MonteCarloConfig(paths=200_000, memory_budget_mb=4096.0)
        times, paths = simulate_log_paths(market, terms.T, config)

        ko_idx = np.unique(
            np.clip(
                np.searchsorted(times, terms.ko_observation_times() - 1e-12),
                1,
                times.size - 1,
            )
        )
        ki_idx = np.unique(
            np.clip(
                np.searchsorted(times, terms.ki_observation_times() - 1e-12),
                1,
                times.size - 1,
            )
        )

        ko_hit = paths[ko_idx, :] >= terms.ko_barrier
        knocked_out = ko_hit.any(axis=0)
        ko_time = np.where(knocked_out, times[ko_idx[ko_hit.argmax(axis=0)]], np.nan)

        cutoff = np.where(knocked_out, ko_time, np.inf)[None, :]
        knocked_in = (
            (paths[ki_idx, :] <= terms.ki_barrier) & (times[ki_idx][:, None] <= cutoff)
        ).any(axis=0)

        payoff = np.full(paths.shape[1], terms.full_coupon_payoff(), dtype=float)
        pay_time = np.full(paths.shape[1], terms.T, dtype=float)
        payoff[knocked_out] = terms.knock_out_payoff(ko_time[knocked_out])
        pay_time[knocked_out] = ko_time[knocked_out]
        ki_survived = ~knocked_out & knocked_in
        payoff[ki_survived] = terms.knock_in_payoff(paths[-1, ki_survived])

        correct = float(np.mean(payoff * np.exp(-market.r * pay_time)))
        legacy = float(np.mean(payoff) * np.exp(-market.r * terms.T))
        return correct, legacy

    def test_zero_rate_makes_the_two_discountings_identical(self) -> None:
        """Control case: at r = 0 every discount factor is 1, so the defect
        cannot manifest and the correct and legacy calculations must coincide.

        Together with the next test this pins the cause: the two agree exactly
        when rates vanish and diverge in a signed, predictable way when they
        do not.
        """
        market = EquityMarket(S=5000.0, r=0.0, sigma=0.20, q=0.01)
        correct, legacy = self._price_both_ways(market, reference_terms())
        assert correct == pytest.approx(legacy, abs=1e-14)

    @pytest.mark.parametrize("rate", [0.02, 0.05, 0.08])
    def test_deferring_the_coupon_to_maturity_understates_the_note(self, rate: float) -> None:
        """The legacy calculation is strictly lower whenever rates are positive.

        A knocked-out note pays at its knock-out date; discounting that payment
        from maturity instead applies too large a discount to the majority of
        paths.
        """
        market = EquityMarket(S=5000.0, r=rate, sigma=0.20, q=0.01)
        correct, legacy = self._price_both_ways(market, reference_terms())
        assert correct > legacy

    def test_the_understatement_grows_with_the_rate(self) -> None:
        """The error scales with the rate, as a discounting error must."""
        terms = reference_terms()
        gaps = []
        for rate in (0.01, 0.04, 0.08):
            correct, legacy = self._price_both_ways(
                EquityMarket(S=5000.0, r=rate, sigma=0.20, q=0.01), terms
            )
            gaps.append(correct - legacy)
        assert gaps[0] < gaps[1] < gaps[2]

    def test_expected_knockout_time_is_well_before_maturity(self) -> None:
        """Quantifies why the defect mattered: most paths pay early."""
        result = price_snowball_mc(
            reference_terms(), REFERENCE_MARKET, MonteCarloConfig(paths=200_000)
        )
        assert result.ko_prob > 0.5
        assert 0.0 < result.expected_ko_time < 0.75


class TestLockup:
    """Defect gate: the legacy model had no lock-up parameter at all."""

    def test_lockup_reduces_knockout_probability(self) -> None:
        config = MonteCarloConfig(paths=200_000)
        without = price_snowball_mc(reference_terms(lockup_months=0), REFERENCE_MARKET, config)
        with_lockup = price_snowball_mc(reference_terms(lockup_months=3), REFERENCE_MARKET, config)
        assert with_lockup.ko_prob < without.ko_prob

    def test_lockup_defers_knockout(self) -> None:
        config = MonteCarloConfig(paths=200_000)
        without = price_snowball_mc(reference_terms(lockup_months=0), REFERENCE_MARKET, config)
        with_lockup = price_snowball_mc(reference_terms(lockup_months=3), REFERENCE_MARKET, config)
        assert with_lockup.expected_ko_time > without.expected_ko_time

    def test_both_engines_respond_to_lockup(self) -> None:
        pde_without = price_snowball_pde(reference_terms(lockup_months=0), REFERENCE_MARKET).price
        pde_with = price_snowball_pde(reference_terms(lockup_months=3), REFERENCE_MARKET).price
        assert pde_with != pytest.approx(pde_without, abs=1e-6)

    def test_lockup_longer_than_tenor_is_rejected(self) -> None:
        with pytest.raises(ParameterError, match="shorter than"):
            reference_terms(lockup_months=18)


class TestFuturesBasis:
    """Defect gate: the legacy model could not express the index futures basis."""

    def test_basis_lowers_the_drift_without_touching_discounting(self) -> None:
        plain = EquityMarket(S=5000.0, r=0.02, sigma=0.2, q=0.01)
        with_basis = EquityMarket(S=5000.0, r=0.02, sigma=0.2, q=0.01, basis_pa=0.05)
        assert with_basis.drift < plain.drift
        assert with_basis.r == plain.r  # discounting unchanged
        assert with_basis.forward(1.0) < plain.forward(1.0)

    def test_basis_reduces_the_note_value(self) -> None:
        """A deeper discount lowers the drift, so the note is worth less --
        which is exactly what lets a desk quote a higher coupon against it."""
        config = MonteCarloConfig(paths=200_000)
        terms = reference_terms()
        plain = price_snowball_mc(terms, EquityMarket(5000.0, 0.02, 0.2, 0.01), config)
        basis = price_snowball_mc(
            terms, EquityMarket(5000.0, 0.02, 0.2, 0.01, basis_pa=0.06), config
        )
        assert basis.price < plain.price
        assert basis.ko_prob < plain.ko_prob

    def test_both_engines_respond_to_basis(self) -> None:
        terms = reference_terms()
        plain = price_snowball_pde(terms, EquityMarket(5000.0, 0.02, 0.2, 0.01)).price
        basis = price_snowball_pde(
            terms, EquityMarket(5000.0, 0.02, 0.2, 0.01, basis_pa=0.06)
        ).price
        assert basis < plain

    def test_a_measured_premium_is_accepted(self) -> None:
        """It was refused until the futures panel was measured.

        A negative basis means the future trades above its index, which happens
        on 40.9% of IH days in the CFFEX panel and leaves the carry negative on
        10.3% of them. Rejecting it was reading a market tendency as a law.
        """
        premium = EquityMarket(S=5000.0, r=0.02, sigma=0.2, q=0.01, basis_pa=-0.01)
        assert premium.carry == pytest.approx(0.0)
        assert premium.drift == pytest.approx(0.02)

        plain = price_snowball_pde(reference_terms(), EquityMarket(5000.0, 0.02, 0.2, 0.01))
        lifted = price_snowball_pde(reference_terms(), premium)
        assert lifted.price > plain.price, "a premium lifts the drift and the note with it"

    def test_an_absurd_basis_magnitude_is_still_rejected(self) -> None:
        """Because the sign was not the error worth catching -- the unit is.

        A basis handed over in percent (``6.0`` for 6%) is the realistic
        mistake, and it survives a sign check perfectly well.
        """
        with pytest.raises(ParameterError, match=r"\|basis_pa\|"):
            EquityMarket(S=5000.0, r=0.02, sigma=0.2, q=0.01, basis_pa=6.0)
        with pytest.raises(ParameterError, match=r"\|basis_pa\|"):
            EquityMarket(S=5000.0, r=0.02, sigma=0.2, q=0.01, basis_pa=-6.0)


class TestKnockedInNoteCanStillKnockOut:
    """Defect gate: the legacy PDE modelled the knocked-in state as a plain put.

    A note that knocks in and later recovers above the barrier still terminates
    and collects its coupon. Treating the knocked-in state as a pure short put
    drops that coupon; measured on the reference trade, 0.12% of paths take
    that route and omitting them left the PDE ~0.46% below the Monte Carlo.
    """

    def test_some_paths_knock_in_then_knock_out(self) -> None:
        result = price_snowball_mc(
            reference_terms(), REFERENCE_MARKET, MonteCarloConfig(paths=500_000)
        )
        recovered = result.ki_prob - result.ki_no_ko_prob
        assert recovered > 0.0, "no path knocks in then recovers; the gate is vacuous"
        assert result.ki_no_ko_prob < result.ki_prob


# ---------------------------------------------------------------------------
# Closed-form benchmarks
# ---------------------------------------------------------------------------


class TestAnalyticBenchmarks:
    """Cases with a closed form, so correctness does not rest on engine agreement."""

    @staticmethod
    def _degenerate_terms() -> SnowballTerms:
        """Barriers placed far enough away to be unreachable."""
        return SnowballTerms(
            S0=5000.0, T=1.0, ko_pct=3.0, ki_pct=0.30, coupon_pa=0.20, lockup_months=0
        )

    def test_unreachable_barriers_reduce_to_a_zero_coupon_bond(self) -> None:
        """With neither barrier in play the note is just a discounted coupon."""
        terms = self._degenerate_terms()
        expected = terms.coupon_pa * terms.T * np.exp(-REFERENCE_MARKET.r * terms.T)

        mc = price_snowball_mc(terms, REFERENCE_MARKET, MonteCarloConfig(paths=50_000))
        pde = price_snowball_pde(terms, REFERENCE_MARKET, PDEConfig(space_nodes=800))
        assert mc.price == pytest.approx(expected, abs=1e-8)
        assert pde.price == pytest.approx(expected, abs=1e-6)
        assert mc.ko_prob == pytest.approx(0.0, abs=1e-9)
        assert mc.ki_prob == pytest.approx(0.0, abs=1e-9)

    def test_semiannual_knockout_matches_closed_form(self) -> None:
        """Two knock-out dates with an unreachable knock-in has a closed form.

        If the note does not knock out at the six-month date then its payoff is
        ``coupon * T`` whether or not it knocks out at maturity, so only the
        first observation matters:
        ``V = e^{-0.5r} c (0.5) P + e^{-rT} c T (1 - P)``.
        """
        market = REFERENCE_MARKET
        terms = SnowballTerms(
            S0=5000.0,
            T=1.0,
            ko_pct=1.03,
            ki_pct=0.30,
            coupon_pa=0.20,
            obs_freq=2,
            lockup_months=0,
        )
        barrier = terms.ko_barrier
        prob_ko = float(
            ndtr(
                (np.log(terms.S0 / barrier) + (market.drift - 0.5 * market.sigma**2) * 0.5)
                / (market.sigma * np.sqrt(0.5))
            )
        )
        expected = np.exp(-market.r * 0.5) * terms.coupon_pa * 0.5 * prob_ko + np.exp(
            -market.r * terms.T
        ) * terms.coupon_pa * terms.T * (1.0 - prob_ko)

        mc = price_snowball_mc(terms, market, MonteCarloConfig(paths=1_000_000))
        assert mc.price == pytest.approx(expected, abs=4.0 * mc.std_error)

        # The PDE converges to it at first order; check the error roughly halves.
        errors = [
            abs(
                price_snowball_pde(terms, market, PDEConfig(space_nodes=n, steps_per_year=m)).price
                - expected
            )
            for n, m in ((800, 1008), (1600, 2016), (3200, 4032))
        ]
        assert errors[1] < errors[0]
        assert errors[2] < errors[1]
        assert errors[0] / errors[1] == pytest.approx(2.0, abs=0.35)
        assert errors[1] / errors[2] == pytest.approx(2.0, abs=0.35)

    def test_terminal_knockin_matches_closed_form(self) -> None:
        """A single knock-in observation at maturity is a European payoff.

        Also pins the terminal condition: a knock-in observed *at* maturity
        cannot be applied by the backward loop and has to be folded into the
        terminal row. Omitting it left the full coupon sitting across the whole
        knock-in region.
        """
        market = REFERENCE_MARKET
        terms = SnowballTerms(
            S0=5000.0,
            T=1.0,
            ko_pct=3.0,
            ki_pct=0.75,
            coupon_pa=0.20,
            obs_freq=1,
            ki_obs_freq=1,
            lockup_months=0,
        )
        drift = market.drift - 0.5 * market.sigma**2
        vol = market.sigma * np.sqrt(terms.T)
        d = (np.log(terms.ki_barrier / terms.S0) - drift * terms.T) / vol
        prob_ki = float(ndtr(d))
        expected_S = float(terms.S0 * np.exp((market.r - market.q) * terms.T) * ndtr(d - vol))
        expected = np.exp(-market.r * terms.T) * (
            -(prob_ki - expected_S / terms.S0) + terms.coupon_pa * terms.T * (1.0 - prob_ki)
        )

        mc = price_snowball_mc(terms, market, MonteCarloConfig(paths=1_000_000))
        assert mc.price == pytest.approx(expected, abs=4.0 * mc.std_error)

        pde = price_snowball_pde(terms, market, PDEConfig(space_nodes=3200, steps_per_year=4032))
        assert pde.price == pytest.approx(expected, abs=2e-4)


# ---------------------------------------------------------------------------
# The headline gate: do two independent engines agree?
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestEngineConsistency:
    """PDE against Monte Carlo on the real product.

    Before the fixes these differed by 18% while being described as mutual
    validation.

    The tolerance is built from measured error budgets rather than chosen to
    pass. Two sources contribute:

    * The PDE is first-order accurate and run on a finite grid, so it carries a
      genuine discretisation bias. At 3200 nodes by 4032 steps the measured
      step-to-step change is ~1.1e-4, and since the error series is
      approximately geometric with ratio 2 the remaining error is of the same
      order; :data:`PDE_DISCRETISATION_BOUND` is set to 3e-4 to cover it with
      margin across the parameter range.
    * The Monte Carlo contributes its own sampling error, taken from the run's
      reported confidence interval rather than assumed.

    Allowing the sum of the two means the gate fails on a *modelling*
    disagreement -- which is what the original 18% was -- while tolerating the
    numerical error that is genuinely expected.
    """

    #: Conservative bound on the PDE's residual bias at the grid used here.
    PDE_DISCRETISATION_BOUND = 3e-4

    def _budget(self, mc_ci95: float) -> float:
        return self.PDE_DISCRETISATION_BOUND + 2.0 * mc_ci95

    def test_reference_trade(self) -> None:
        terms = reference_terms()
        mc = price_snowball_mc(terms, REFERENCE_MARKET, MonteCarloConfig(paths=1_000_000))
        pde = price_snowball_pde(
            terms, REFERENCE_MARKET, PDEConfig(space_nodes=3200, steps_per_year=4032)
        )
        gap = abs(mc.price - pde.price)
        assert gap < self._budget(mc.ci95), (
            f"PDE {pde.price:.8f} vs MC {mc.price:.8f} (+/-{mc.ci95:.2e}); gap {gap:.3e}"
        )

    def test_pde_sits_within_a_few_tenths_of_a_percent_of_mc(self) -> None:
        """A relative check, so the gate cannot be passed by a tiny price."""
        terms = reference_terms()
        mc = price_snowball_mc(terms, REFERENCE_MARKET, MonteCarloConfig(paths=1_000_000))
        pde = price_snowball_pde(
            terms, REFERENCE_MARKET, PDEConfig(space_nodes=3200, steps_per_year=4032)
        )
        relative = abs(mc.price - pde.price) / abs(mc.price)
        # The legacy engines were 18% apart on this very trade.
        assert relative < 0.01, f"relative gap {relative:.4%}"

    @pytest.mark.parametrize(
        "overrides",
        [
            {"ko_pct": 1.00},
            {"ko_pct": 1.10},
            {"ki_pct": 0.60},
            {"ki_pct": 0.85},
            {"coupon_pa": 0.08},
            {"lockup_months": 0},
            {"lockup_months": 6},
            {"ki_strike_mode": KIStrikeMode.KI},
            {"T": 2.0},
            {"ki_pct": 0.95},
            {"obs_freq": 4},
        ],
        ids=lambda d: ",".join(f"{k}={v}" for k, v in d.items()),
    )
    def test_agreement_across_contract_variations(self, overrides: dict) -> None:
        """One lucky parameter set proving nothing is the failure mode here."""
        terms = reference_terms(**overrides)
        mc = price_snowball_mc(terms, REFERENCE_MARKET, MonteCarloConfig(paths=500_000))
        pde = price_snowball_pde(
            terms, REFERENCE_MARKET, PDEConfig(space_nodes=3200, steps_per_year=4032)
        )
        gap = abs(mc.price - pde.price)
        budget = self._budget(mc.ci95)
        assert gap < budget, (
            f"{overrides}: PDE {pde.price:.8f} vs MC {mc.price:.8f} "
            f"(+/-{mc.ci95:.2e}); gap {gap:.3e} exceeds {budget:.3e}"
        )

    @pytest.mark.parametrize(
        "market",
        [
            EquityMarket(5000.0, 0.02, 0.15, 0.01),
            EquityMarket(5000.0, 0.02, 0.35, 0.01),
            EquityMarket(5000.0, 0.06, 0.20, 0.01),
            EquityMarket(5000.0, 0.00, 0.20, 0.00),
            EquityMarket(5000.0, 0.02, 0.20, 0.01, basis_pa=0.06),
            EquityMarket(5000.0, 0.02, 0.50, 0.01),
        ],
        ids=lambda m: f"sig{m.sigma:g}_r{m.r:g}_b{m.basis_pa:g}",
    )
    def test_agreement_across_market_states(self, market: EquityMarket) -> None:
        terms = reference_terms()
        mc = price_snowball_mc(terms, market, MonteCarloConfig(paths=500_000))
        pde = price_snowball_pde(terms, market, PDEConfig(space_nodes=3200, steps_per_year=4032))
        gap = abs(mc.price - pde.price)
        budget = self._budget(mc.ci95)
        assert gap < budget, f"{market}: PDE {pde.price:.8f} vs MC {mc.price:.8f}; gap {gap:.3e}"


# ---------------------------------------------------------------------------
# Numerical machinery and convergence
# ---------------------------------------------------------------------------


class TestPDEConvergence:
    def test_barriers_land_exactly_on_grid_nodes(self) -> None:
        """A barrier between nodes is displaced by up to dx/2, which costs an
        order of convergence and makes the price oscillate under refinement."""
        from otcpricer.engines.pde import _SnowballGrid

        terms = reference_terms()
        for nodes in (200, 400, 800, 1600):
            grid = _SnowballGrid(terms, REFERENCE_MARKET, PDEConfig(space_nodes=nodes))
            assert grid.S_grid[grid.ki_index] == pytest.approx(terms.ki_barrier, rel=1e-12)
            assert grid.S_grid[grid.ko_index] == pytest.approx(terms.ko_barrier, rel=1e-12)

    def test_time_grid_contains_every_observation_date(self) -> None:
        """Rounding an observation onto the grid shifts it by up to dt/2."""
        from otcpricer.engines.pde import _SnowballGrid

        terms = reference_terms()
        for requested in (252, 300, 504, 1000):
            grid = _SnowballGrid(terms, REFERENCE_MARKET, PDEConfig(steps_per_year=requested))
            assert grid.steps_per_year % terms.ki_obs_freq == 0
            for t_obs in terms.ki_observation_times():
                steps = t_obs / grid.dt
                assert steps == pytest.approx(round(steps), abs=1e-9)

    @pytest.mark.slow
    def test_price_converges_monotonically_under_joint_refinement(self) -> None:
        terms = reference_terms()
        prices = [
            price_snowball_pde(
                terms, REFERENCE_MARKET, PDEConfig(space_nodes=n, steps_per_year=m)
            ).price
            for n, m in ((800, 1008), (1600, 2016), (3200, 4032), (6400, 8064))
        ]
        changes = np.abs(np.diff(prices))
        assert np.all(np.diff(changes) < 0), f"changes not shrinking: {changes}"
        # First order: each halving of the grid roughly halves the error.
        for ratio in changes[:-1] / changes[1:]:
            assert ratio == pytest.approx(2.0, abs=0.35)

    def test_rannacher_damping_can_be_disabled(self) -> None:
        terms = reference_terms()
        damped = price_snowball_pde(terms, REFERENCE_MARKET, PDEConfig(rannacher_steps=4)).price
        undamped = price_snowball_pde(terms, REFERENCE_MARKET, PDEConfig(rannacher_steps=0)).price
        assert np.isfinite(damped)
        assert np.isfinite(undamped)


class TestTridiagonalSolver:
    def test_matches_a_dense_solve(self) -> None:
        rng = np.random.default_rng(0)
        n = 60
        lower = rng.normal(size=n - 1)
        upper = rng.normal(size=n - 1)
        # Diagonally dominant, as the schemes here always are.
        main = np.abs(lower).sum() + np.abs(upper).sum() + rng.uniform(5.0, 6.0, size=n)
        rhs = rng.normal(size=n)

        dense = np.diag(main) + np.diag(lower, -1) + np.diag(upper, 1)
        np.testing.assert_allclose(
            solve_tridiagonal(lower, main, upper, rhs),
            np.linalg.solve(dense, rhs),
            rtol=1e-11,
            atol=1e-12,
        )

    def test_single_equation(self) -> None:
        result = solve_tridiagonal(np.empty(0), np.array([4.0]), np.empty(0), np.array([8.0]))
        assert result == pytest.approx([2.0])

    def test_empty_system(self) -> None:
        assert solve_tridiagonal(np.empty(0), np.empty(0), np.empty(0), np.empty(0)).size == 0


class TestMonteCarloMachinery:
    def test_paths_start_at_spot(self) -> None:
        _, paths = simulate_log_paths(REFERENCE_MARKET, 1.0, MonteCarloConfig(paths=1000))
        np.testing.assert_allclose(paths[0, :], REFERENCE_MARKET.S)

    def test_paths_stay_positive(self) -> None:
        """Log-space integration cannot produce a non-positive level."""
        _, paths = simulate_log_paths(REFERENCE_MARKET, 2.0, MonteCarloConfig(paths=5000))
        assert np.all(paths > 0.0)

    def test_terminal_mean_matches_the_forward(self) -> None:
        """Validates the drift, including the basis."""
        market = EquityMarket(5000.0, 0.03, 0.2, 0.01, basis_pa=0.04)
        _, paths = simulate_log_paths(market, 1.0, MonteCarloConfig(paths=400_000))
        assert float(paths[-1, :].mean()) == pytest.approx(market.forward(1.0), rel=5e-3)

    def test_antithetic_noise_sums_to_zero(self) -> None:
        from otcpricer.engines.mc import normal_draws

        draws = normal_draws(10, 1000, seed=1, antithetic=True)
        np.testing.assert_allclose(draws.sum(axis=1), 0.0, atol=1e-10)

    def test_antithetic_does_not_bias_the_price(self) -> None:
        terms = reference_terms()
        with_anti = np.mean(
            [
                price_snowball_mc(
                    terms, REFERENCE_MARKET, MonteCarloConfig(paths=200_000, seed=s)
                ).price
                for s in range(4)
            ]
        )
        without = np.mean(
            [
                price_snowball_mc(
                    terms,
                    REFERENCE_MARKET,
                    MonteCarloConfig(paths=200_000, seed=s, antithetic=False),
                ).price
                for s in range(4)
            ]
        )
        assert with_anti == pytest.approx(without, abs=1e-3)

    def test_standard_error_is_realistic(self) -> None:
        """The reported error must bracket the true across-seed scatter.

        The naive ``std / sqrt(n)`` is invalid under antithetic sampling; it
        reported 1.40e-4 against a true scatter of 1.00e-4 before the fix.
        """
        terms = reference_terms()
        results = [
            price_snowball_mc(terms, REFERENCE_MARKET, MonteCarloConfig(paths=200_000, seed=s))
            for s in range(8)
        ]
        observed = float(np.std([r.price for r in results], ddof=1))
        reported = float(np.mean([r.std_error for r in results]))
        # Conservative is fine; wildly wrong in either direction is not.
        assert 0.4 * reported < observed < 2.5 * reported

    def test_error_shrinks_as_root_n(self) -> None:
        terms = reference_terms()
        coarse = price_snowball_mc(terms, REFERENCE_MARKET, MonteCarloConfig(paths=100_000))
        fine = price_snowball_mc(terms, REFERENCE_MARKET, MonteCarloConfig(paths=400_000))
        assert fine.std_error == pytest.approx(coarse.std_error / 2.0, rel=0.35)

    def test_reproducible_for_a_fixed_seed(self) -> None:
        terms = reference_terms()
        config = MonteCarloConfig(paths=50_000, seed=7)
        assert (
            price_snowball_mc(terms, REFERENCE_MARKET, config).price
            == price_snowball_mc(terms, REFERENCE_MARKET, config).price
        )

    def test_probabilities_are_well_formed(self) -> None:
        result = price_snowball_mc(
            reference_terms(), REFERENCE_MARKET, MonteCarloConfig(paths=100_000)
        )
        assert 0.0 <= result.ko_prob <= 1.0
        assert 0.0 <= result.ki_prob <= 1.0
        assert result.ki_no_ko_prob <= result.ki_prob


class TestEconomicSanity:
    """Directional responses a desk would expect."""

    def test_higher_coupon_is_worth_more(self) -> None:
        config = MonteCarloConfig(paths=200_000)
        low = price_snowball_mc(reference_terms(coupon_pa=0.10), REFERENCE_MARKET, config)
        high = price_snowball_mc(reference_terms(coupon_pa=0.30), REFERENCE_MARKET, config)
        assert high.price > low.price

    def test_lower_knockin_barrier_is_worth_more(self) -> None:
        """Less downside exposure, so a higher value."""
        config = MonteCarloConfig(paths=200_000)
        deep = price_snowball_mc(reference_terms(ki_pct=0.60), REFERENCE_MARKET, config)
        shallow = price_snowball_mc(reference_terms(ki_pct=0.85), REFERENCE_MARKET, config)
        assert deep.price > shallow.price
        assert deep.ki_prob < shallow.ki_prob

    def test_higher_volatility_raises_knockin_probability(self) -> None:
        config = MonteCarloConfig(paths=200_000)
        calm = price_snowball_mc(reference_terms(), EquityMarket(5000.0, 0.02, 0.12, 0.01), config)
        wild = price_snowball_mc(reference_terms(), EquityMarket(5000.0, 0.02, 0.40, 0.01), config)
        assert wild.ki_prob > calm.ki_prob
        assert wild.price < calm.price

    def test_price_is_a_sensible_fraction_of_notional(self) -> None:
        """A typical A-share snowball marks in the low single-digit percents."""
        result = price_snowball_mc(
            reference_terms(), REFERENCE_MARKET, MonteCarloConfig(paths=200_000)
        )
        assert 0.0 < result.price < 0.20


class TestScheduleFidelity:
    """A grid too coarse for the contract is refused, not silently coarsened.

    This is a fidelity defect, not the Phase 1 oscillation one, and the
    distinction matters. Phase 1's barrier and observation misalignments made
    the price *jitter* under refinement -- visibly wrong, and fixed by snapping
    the grid. What was left was quieter: when the contract asks for more
    observation dates than the grid has steps, both engines dropped the excess
    and priced a note nobody wrote.

    A note with fewer knock-out chances is worth strictly more, so the error is
    one-signed rather than noise. And the two engines dropped them
    *differently*, which is what made it invisible: the cross-engine check
    compared two substituted contracts.
    """

    @staticmethod
    def _terms(obs_freq: int) -> SnowballTerms:
        return SnowballTerms(
            S0=5000.0, T=1.0, ko_pct=1.03, ki_pct=0.75, coupon_pa=0.20, obs_freq=obs_freq
        )

    def test_the_simulation_refuses_a_schedule_it_cannot_hold(self) -> None:
        """Measured before the fix: obs_freq 252, 300, 400, 600 and 1000 -- notes
        asking for 189, 225, 300, 450 and 750 knock-out dates -- all returned
        0.04795316632228654 at 252 steps a year. One bit-identical price for
        five different contracts.
        """
        config = MonteCarloConfig(paths=5_000, steps_per_year=252, seed=1, memory_budget_mb=4096.0)
        for obs_freq in (300, 400, 600, 1000):
            with pytest.raises(EngineError, match="knock-out dates collide"):
                price_snowball_mc(self._terms(obs_freq), REFERENCE_MARKET, config)

    def test_the_finite_difference_engine_refuses_it_too(self) -> None:
        """And with the same collision count, which is the agreement that
        matters: before the fix it returned five *different* prices spread over
        4.22e-04 with no order to them, because its dict kept the later
        contractual date per step where the simulation kept the grid's.
        """
        config = PDEConfig(space_nodes=400, steps_per_year=252)
        for obs_freq in (300, 400, 600):
            with pytest.raises(EngineError, match="knock-out observation dates collide"):
                price_snowball_pde(self._terms(obs_freq), REFERENCE_MARKET, config)

    def test_both_engines_count_the_same_collisions(self) -> None:
        """Because they are reading one contract against one grid size."""
        import re

        terms = self._terms(400)
        with pytest.raises(EngineError) as mc_error:
            price_snowball_mc(
                terms,
                REFERENCE_MARKET,
                MonteCarloConfig(paths=1_000, steps_per_year=252, seed=1, memory_budget_mb=4096.0),
            )
        with pytest.raises(EngineError) as pde_error:
            price_snowball_pde(
                terms, REFERENCE_MARKET, PDEConfig(space_nodes=400, steps_per_year=252)
            )

        def first_number(message: str) -> int:
            found = re.search(r"\d+", message)
            assert found is not None, message
            return int(found.group())

        assert first_number(str(mc_error.value)) == first_number(str(pde_error.value)) == 111

    def test_the_error_names_a_grid_that_works(self) -> None:
        """A refusal that does not say what to do instead is an obstacle.

        The advice is always a finer grid rather than a coarser contract,
        because the contract is the thing being priced.
        """
        terms = self._terms(400)
        with pytest.raises(EngineError, match="steps_per_year >= 400"):
            price_snowball_mc(
                terms,
                REFERENCE_MARKET,
                MonteCarloConfig(paths=1_000, steps_per_year=252, seed=1, memory_budget_mb=4096.0),
            )

        priced = price_snowball_mc(
            terms,
            REFERENCE_MARKET,
            MonteCarloConfig(paths=5_000, steps_per_year=400, seed=1, memory_budget_mb=8192.0),
        )
        assert 0.0 < priced.price < 0.20

    def test_a_representable_schedule_is_untouched(self) -> None:
        """The check is a gate, not a constraint on anything real.

        Every frequency a term sheet actually carries -- monthly, fortnightly,
        weekly, daily -- fits a 252-step year, and the default monthly schedule
        fits with 20x to spare.
        """
        config = MonteCarloConfig(paths=5_000, steps_per_year=252, seed=1, memory_budget_mb=4096.0)
        for obs_freq in (1, 2, 4, 12, 26, 52, 126, 252):
            priced = price_snowball_mc(self._terms(obs_freq), REFERENCE_MARKET, config)
            assert 0.0 < priced.price < 0.30, obs_freq

    def test_more_observations_are_worth_more_which_is_why_dropping_them_lies(
        self,
    ) -> None:
        """The economic fact behind the refusal.

        A knock-out is a chance to terminate early on the coupon, so more
        observations make the note worth more to the client. Dropping them is
        therefore one-signed error, not rounding -- and the sign is the one
        that flatters the quote.
        """
        config = MonteCarloConfig(
            paths=100_000, steps_per_year=252, seed=3, memory_budget_mb=8192.0
        )
        prices = [
            price_snowball_mc(self._terms(freq), REFERENCE_MARKET, config).price
            for freq in (4, 12, 52, 252)
        ]
        assert all(later < earlier for earlier, later in zip(prices, prices[1:], strict=False)), (
            prices
        )

    def test_a_seasoned_valuation_drops_past_dates_without_complaint(self) -> None:
        """Dropping a date because it has happened is not substitution.

        The distinction is the whole semantics of t0: a past observation is
        history, and its outcome is the caller's to state through knocked_in.
        Only two *future* dates landing on one step is a fidelity failure.
        """
        terms = self._terms(12)
        config = MonteCarloConfig(paths=20_000, steps_per_year=252, seed=1, memory_budget_mb=4096.0)
        seasoned = price_snowball_mc(terms, REFERENCE_MARKET, config, t0=0.5)
        assert 0.0 < seasoned.price < 0.30
        # Half the note's knock-out dates are behind it, so it can knock out
        # on fewer of them and is worth less than the note at inception.
        assert seasoned.ko_prob < price_snowball_mc(terms, REFERENCE_MARKET, config).ko_prob
