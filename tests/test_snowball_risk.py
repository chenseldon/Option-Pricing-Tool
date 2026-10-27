"""Tests for snowball Greeks.

Ordered by strength of evidence, as elsewhere in this suite:

1. **Closed-form anchors.** Two degenerate configurations have exact Greeks, so
   correctness does not rest on two numerical methods agreeing with each other.
2. **Convergence.** The PDE is first-order accurate; refinement must drive the
   error down towards the closed form rather than wander.
3. **Sign and economics.** Relations any correct snowball risk must satisfy.
4. **Cross-method agreement.** PDE against Monte Carlo, on measured budgets.
5. **Regression anchors**, including that the legacy wrong values cannot return.

The legacy implementation reported three Greek sets that disagreed in sign. The
anchors below are what make the replacement verifiable rather than merely
self-consistent.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtr

from otcpricer.core.market import EquityMarket
from otcpricer.core.types import KIStrikeMode
from otcpricer.engines.mc import MonteCarloConfig
from otcpricer.engines.pde import PDEConfig, price_snowball_pde
from otcpricer.engines.risk import (
    RiskComparison,
    compare_snowball_risk,
    snowball_risk_mc,
    snowball_risk_pde,
)
from otcpricer.products.snowball import SnowballTerms

REFERENCE_MARKET = EquityMarket(S=5000.0, r=0.02, sigma=0.20, q=0.01)

#: Fine enough that the first-order discretisation error is well below the
#: tolerances asserted here, while still solving in well under a second.
FINE = PDEConfig(space_nodes=3200, steps_per_year=2016)
STANDARD = PDEConfig(space_nodes=1600, steps_per_year=1008)


def reference_terms(**overrides: object) -> SnowballTerms:
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
# 1. Closed-form anchors
# ---------------------------------------------------------------------------


class TestZeroCouponAnchor:
    """Barriers placed out of reach reduce the note to a zero-coupon bond.

    Then every Greek is known exactly, and three of them are exactly zero --
    the hardest possible gate, because it cannot be passed by accident.
    """

    TERMS = SnowballTerms(
        S0=5000.0, T=1.0, ko_pct=3.0, ki_pct=0.30, coupon_pa=0.20, lockup_months=0
    )

    @property
    def exact_price(self) -> float:
        return 0.20 * 1.0 * float(np.exp(-REFERENCE_MARKET.r * 1.0))

    def test_price_matches_the_discounted_coupon(self) -> None:
        result = price_snowball_pde(self.TERMS, REFERENCE_MARKET, FINE)
        assert result.price == pytest.approx(self.exact_price, abs=1e-8)

    def test_delta_is_zero(self) -> None:
        """No spot sensitivity: the payoff does not depend on the underlying."""
        risk = snowball_risk_pde(self.TERMS, REFERENCE_MARKET, FINE)
        assert risk.greeks.delta == pytest.approx(0.0, abs=1e-9)

    def test_gamma_is_zero(self) -> None:
        risk = snowball_risk_pde(self.TERMS, REFERENCE_MARKET, FINE)
        assert risk.greeks.gamma == pytest.approx(0.0, abs=1e-11)

    def test_vega_is_zero(self) -> None:
        """A payoff independent of the path cannot be sensitive to volatility."""
        risk = snowball_risk_pde(self.TERMS, REFERENCE_MARKET, FINE)
        assert risk.greeks.vega == pytest.approx(0.0, abs=1e-7)

    def test_rho_matches_minus_t_times_value(self) -> None:
        """A fixed cash flow at T has rho = -T * V, per percentage point."""
        risk = snowball_risk_pde(self.TERMS, REFERENCE_MARKET, FINE)
        expected = -1.0 * self.exact_price * 0.01
        assert risk.greeks.rho == pytest.approx(expected, rel=1e-5)

    def test_theta_matches_the_accretion_rate(self) -> None:
        """A discount bond accretes at the risk-free rate: theta = r * V / 365."""
        risk = snowball_risk_pde(self.TERMS, REFERENCE_MARKET, FINE)
        expected = REFERENCE_MARKET.r * self.exact_price / 365.0
        assert risk.greeks.theta == pytest.approx(expected, rel=1e-4)

    def test_monte_carlo_agrees_on_the_zero_greeks(self) -> None:
        risk = snowball_risk_mc(
            self.TERMS, REFERENCE_MARKET, MonteCarloConfig(paths=50_000, memory_budget_mb=4096.0)
        )
        # Every path pays the same amount, so the differences are exactly zero.
        assert risk.greeks.delta == pytest.approx(0.0, abs=1e-12)
        assert risk.greeks.gamma == pytest.approx(0.0, abs=1e-12)
        assert risk.greeks.vega == pytest.approx(0.0, abs=1e-12)


def _european_anchor_value(
    spot: float,
    *,
    r: float = 0.02,
    q: float = 0.01,
    sigma: float = 0.20,
    T: float = 1.0,
    coupon: float = 0.20,
    ki_barrier: float = 3750.0,
    put_strike: float = 5000.0,
) -> float:
    """Closed-form value of a snowball whose only observation is at maturity.

    With the knock-out out of reach and a single knock-in observation at ``T``,
    the note is European: below the barrier it pays the short-put loss, above it
    the full coupon. Both legs are standard lognormal integrals.

    ``ki_barrier`` and ``put_strike`` are contract constants; only ``spot``
    varies, which is what makes differentiating this a *spot* delta rather than
    the derivative of a re-struck trade.
    """
    drift = r - q - 0.5 * sigma**2
    total_vol = sigma * np.sqrt(T)
    d = (np.log(ki_barrier / spot) - drift * T) / total_vol
    prob_ki = float(ndtr(d))
    expected_spot = spot * float(np.exp((r - q) * T)) * float(ndtr(d - total_vol))
    return float(
        np.exp(-r * T) * (-(prob_ki - expected_spot / put_strike) + coupon * T * (1.0 - prob_ki))
    )


class TestEuropeanAnchor:
    """A terminal-only knock-in has a closed form, including its Greeks.

    This is the anchor that proves the grid differentiation itself, not merely
    that two engines agree.
    """

    TERMS = SnowballTerms(
        S0=5000.0,
        T=1.0,
        ko_pct=3.0,
        ki_pct=0.75,
        coupon_pa=0.20,
        obs_freq=1,
        ki_obs_freq=1,
        lockup_months=0,
    )

    @staticmethod
    def _analytic_greeks(spot: float = 5000.0) -> tuple[float, float]:
        """``(delta, gamma)`` of the closed form, by high-accuracy differences."""
        h = 1e-4 * spot
        up = _european_anchor_value(spot + h)
        mid = _european_anchor_value(spot)
        down = _european_anchor_value(spot - h)
        return (up - down) / (2.0 * h), (up - 2.0 * mid + down) / (h * h)

    def test_price_matches_closed_form(self) -> None:
        result = price_snowball_pde(self.TERMS, REFERENCE_MARKET, FINE)
        assert result.price == pytest.approx(_european_anchor_value(5000.0), abs=3e-4)

    def test_delta_matches_closed_form(self) -> None:
        expected, _ = self._analytic_greeks()
        risk = snowball_risk_pde(self.TERMS, REFERENCE_MARKET, FINE)
        assert risk.greeks.delta == pytest.approx(expected, rel=0.01)

    def test_gamma_matches_closed_form(self) -> None:
        _, expected = self._analytic_greeks()
        risk = snowball_risk_pde(self.TERMS, REFERENCE_MARKET, FINE)
        assert risk.greeks.gamma == pytest.approx(expected, rel=0.01)

    def test_delta_matches_closed_form_across_spot(self) -> None:
        """Not just at the fixing -- the whole surface must be right."""
        for spot in (4200.0, 4600.0, 5000.0, 5400.0, 5800.0):
            expected, _ = self._analytic_greeks(spot)
            market = EquityMarket(spot, 0.02, 0.20, 0.01)
            risk = snowball_risk_pde(self.TERMS, market, FINE)
            assert risk.greeks.delta == pytest.approx(expected, rel=0.03), (
                f"spot {spot}: got {risk.greeks.delta:.8e}, expected {expected:.8e}"
            )

    @pytest.mark.slow
    def test_delta_converges_at_first_order(self) -> None:
        """Refinement must drive the error down, roughly halving each time."""
        expected, _ = self._analytic_greeks()
        errors = []
        for nodes, steps in ((800, 504), (1600, 1008), (3200, 2016), (6400, 4032)):
            risk = snowball_risk_pde(
                self.TERMS, REFERENCE_MARKET, PDEConfig(space_nodes=nodes, steps_per_year=steps)
            )
            errors.append(abs(risk.greeks.delta - expected))
        assert all(np.diff(errors) < 0), f"delta errors not shrinking: {errors}"
        for ratio in np.array(errors[:-1]) / np.array(errors[1:]):
            assert ratio == pytest.approx(2.0, abs=0.4)

    @pytest.mark.slow
    def test_gamma_converges(self) -> None:
        _, expected = self._analytic_greeks()
        errors = []
        for nodes, steps in ((800, 504), (1600, 1008), (3200, 2016), (6400, 4032)):
            risk = snowball_risk_pde(
                self.TERMS, REFERENCE_MARKET, PDEConfig(space_nodes=nodes, steps_per_year=steps)
            )
            errors.append(abs(risk.greeks.gamma - expected))
        assert all(np.diff(errors) < 0), f"gamma errors not shrinking: {errors}"


# ---------------------------------------------------------------------------
# 2. Convergence and numerical stability
# ---------------------------------------------------------------------------


class TestGridStability:
    """The reference trade sits 3% below its knock-out, so the second
    difference is taken close to a kink. That is where a grid method is most
    likely to oscillate, so it gets its own test."""

    @pytest.mark.slow
    def test_delta_and_gamma_settle_under_refinement(self) -> None:
        deltas, gammas = [], []
        for nodes, steps in ((800, 504), (1600, 1008), (3200, 2016), (6400, 4032)):
            risk = snowball_risk_pde(
                reference_terms(),
                REFERENCE_MARKET,
                PDEConfig(space_nodes=nodes, steps_per_year=steps),
            )
            deltas.append(risk.greeks.delta)
            gammas.append(risk.greeks.gamma)

        for series, name in ((deltas, "delta"), (gammas, "gamma")):
            changes = np.abs(np.diff(series))
            assert np.all(np.diff(changes) < 0), f"{name} not settling: {series}"

    def test_gamma_is_smooth_in_spot_near_the_barrier(self) -> None:
        """No oscillation as spot sweeps up to and through the knock-out."""
        terms = reference_terms()
        spots = np.linspace(4700.0, 5400.0, 25)
        gammas = np.array(
            [
                snowball_risk_pde(terms, EquityMarket(s, 0.02, 0.20, 0.01), STANDARD).greeks.gamma
                for s in spots
            ]
        )
        assert np.all(np.isfinite(gammas))
        # A sign flip would be legitimate; wild alternation would not. Count
        # changes of direction in the sequence.
        direction_changes = int(np.sum(np.diff(np.sign(np.diff(gammas))) != 0))
        assert direction_changes <= 4, f"gamma oscillating: {gammas}"


# ---------------------------------------------------------------------------
# 3. Sign and economic relations
# ---------------------------------------------------------------------------


class TestEconomicRelations:
    """Relations any correct snowball risk must satisfy."""

    def test_reference_trade_delta_is_positive(self) -> None:
        """On the reference trade the holder is net long the underlying.

        Scoped to this trade deliberately: the sign is *not* universal, as
        :meth:`test_delta_changes_sign_with_the_knockin_level` shows. The
        legacy resimulation FD gave -8.4e-05 here, which is wrong for this
        contract whatever the general case.
        """
        risk = snowball_risk_pde(reference_terms(), REFERENCE_MARKET, STANDARD)
        assert risk.greeks.delta > 0.0

    def test_delta_changes_sign_with_the_knockin_level(self) -> None:
        """A real feature of the structure, not a numerical artefact.

        With the knock-in remote, the dominant exposure is that a rally
        triggers an *early* knock-out, paying ``coupon * t_ko`` instead of the
        full ``coupon * T``. The holder is then effectively short the upside
        and delta is negative. As the knock-in moves closer, downside exposure
        takes over and delta turns positive. Measured crossover sits between
        60% and 70%.

        Pinned as a test because it is the kind of result that looks like a bug
        and is not, and because a model that failed to reproduce it would be
        missing the economics of the product.
        """
        deltas = [
            snowball_risk_pde(reference_terms(ki_pct=ki), REFERENCE_MARKET, STANDARD).greeks.delta
            for ki in (0.50, 0.60, 0.70, 0.80, 0.90)
        ]
        assert deltas[0] < 0.0, "a remote knock-in should give a short-upside delta"
        assert deltas[-1] > 0.0, "a close knock-in should give a long delta"
        assert all(np.diff(deltas) > 0), f"delta should rise monotonically: {deltas}"

    def test_holder_is_short_volatility(self) -> None:
        """Higher vol means more knock-in risk, so vega is negative."""
        risk = snowball_risk_pde(reference_terms(), REFERENCE_MARKET, STANDARD)
        assert risk.greeks.vega < 0.0

    def test_holder_earns_coupon_carry(self) -> None:
        """Theta is positive: the accrued coupon grows as time passes without
        a knock-in, which is the economic point of the structure."""
        risk = snowball_risk_pde(reference_terms(), REFERENCE_MARKET, STANDARD)
        assert risk.greeks.theta > 0.0

    def test_raising_the_knockin_barrier_makes_vega_more_negative(self) -> None:
        """Closer to the barrier, volatility hurts more."""
        low = snowball_risk_pde(reference_terms(ki_pct=0.60), REFERENCE_MARKET, STANDARD)
        high = snowball_risk_pde(reference_terms(ki_pct=0.90), REFERENCE_MARKET, STANDARD)
        assert high.greeks.vega < low.greeks.vega

    def test_delta_is_larger_when_closer_to_the_knockin(self) -> None:
        """Spot near the knock-in is where the note is most spot-sensitive."""
        terms = reference_terms()
        near = snowball_risk_pde(terms, EquityMarket(3900.0, 0.02, 0.20, 0.01), STANDARD)
        far = snowball_risk_pde(terms, EquityMarket(5400.0, 0.02, 0.20, 0.01), STANDARD)
        assert near.greeks.delta > far.greeks.delta

    def test_price_and_delta_are_consistent_across_spot(self) -> None:
        """Delta must be the slope of the price curve it came from.

        An independent check on the chain-rule conversion out of log space: if
        the 1/S factors were wrong, this would fail even though the price curve
        itself is right.
        """
        terms = reference_terms()
        spots = np.array([4800.0, 4900.0, 5000.0, 5100.0, 5200.0])
        prices = np.array(
            [
                price_snowball_pde(terms, EquityMarket(s, 0.02, 0.20, 0.01), FINE).price
                for s in spots
            ]
        )
        secant = (prices[3] - prices[1]) / (spots[3] - spots[1])
        risk = snowball_risk_pde(terms, EquityMarket(5000.0, 0.02, 0.20, 0.01), FINE)
        assert risk.greeks.delta == pytest.approx(secant, rel=0.05)

    def test_knocked_in_note_has_different_risk(self) -> None:
        """A breached note is a short put: more spot-sensitive, no coupon upside."""
        terms = reference_terms()
        clean = snowball_risk_pde(terms, REFERENCE_MARKET, STANDARD, knocked_in=False)
        breached = snowball_risk_pde(terms, REFERENCE_MARKET, STANDARD, knocked_in=True)
        assert breached.price < clean.price
        assert breached.greeks.delta > clean.greeks.delta


# ---------------------------------------------------------------------------
# 4. Cross-method agreement
# ---------------------------------------------------------------------------


class TestMethodAgreement:
    """PDE against Monte Carlo, on budgets built from measured error."""

    def test_reference_trade_agrees_on_every_greek(self) -> None:
        comparison = compare_snowball_risk(reference_terms(), REFERENCE_MARKET)
        assert isinstance(comparison, RiskComparison)
        assert comparison.agree, comparison.verdicts

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "overrides",
        [
            {"ki_pct": 0.60},
            {"ki_pct": 0.85},
            {"ko_pct": 1.10},
            {"coupon_pa": 0.10},
            {"lockup_months": 0},
            {"ki_strike_mode": KIStrikeMode.KI},
        ],
        ids=lambda d: ",".join(f"{k}={v}" for k, v in d.items()),
    )
    def test_agreement_across_contract_variations(self, overrides: dict) -> None:
        comparison = compare_snowball_risk(reference_terms(**overrides), REFERENCE_MARKET)
        assert comparison.agree, f"{overrides}: {comparison.verdicts}"

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "market",
        [
            EquityMarket(5000.0, 0.02, 0.15, 0.01),
            EquityMarket(5000.0, 0.02, 0.35, 0.01),
            EquityMarket(5000.0, 0.06, 0.20, 0.01),
            EquityMarket(5000.0, 0.02, 0.20, 0.01, basis_pa=0.06),
            EquityMarket(5400.0, 0.02, 0.20, 0.01),
        ],
        ids=lambda m: f"S{m.S:g}_sig{m.sigma:g}_r{m.r:g}_b{m.basis_pa:g}",
    )
    def test_agreement_across_market_states(self, market: EquityMarket) -> None:
        comparison = compare_snowball_risk(reference_terms(), market)
        assert comparison.agree, f"{market}: {comparison.verdicts}"

    def test_comparison_reports_a_budget_not_just_a_verdict(self) -> None:
        """A pass carried by a wide sampling error is weaker evidence than one
        with both methods tight, so the breakdown has to be visible."""
        comparison = compare_snowball_risk(reference_terms(), REFERENCE_MARKET)
        for name in ("delta", "vega", "rho"):
            verdict = comparison.verdicts[name]
            assert verdict["budget"] > 0.0
            assert verdict["mc_std_error"] > 0.0
            assert verdict["gap"] <= verdict["budget"]

    def test_gamma_is_judged_against_its_sampling_error(self) -> None:
        """Not on sign.

        The Monte Carlo gamma estimator divides a second difference of a
        discontinuous payoff by ``h^2``, so its sampling error can exceed the
        value. Demanding the simulated sign match would be incoherent given
        that the comparison itself declares the estimator unreliable.
        """
        comparison = compare_snowball_risk(reference_terms(), REFERENCE_MARKET)
        gamma_verdict = comparison.verdicts["gamma"]
        assert "note" in gamma_verdict
        assert gamma_verdict["mc_std_error"] > 0.0
        assert gamma_verdict["gap"] <= gamma_verdict["budget"]

    def test_gamma_verdict_survives_a_near_zero_true_value(self) -> None:
        """Where gamma is small, the simulated sign is a coin toss.

        A remote knock-in gives gamma around -2.9e-08 against a Monte Carlo
        standard error near 2e-07. The comparison must still pass: the two are
        consistent, and only a sign-based rule would call it a disagreement.
        """
        comparison = compare_snowball_risk(reference_terms(ki_pct=0.60), REFERENCE_MARKET)
        assert comparison.agree, comparison.verdicts


# ---------------------------------------------------------------------------
# 5. Regression and provenance
# ---------------------------------------------------------------------------


class TestRegressionAnchors:
    """Measured values, and gates against the legacy wrong ones returning."""

    #: Measured on the corrected engines; see the commit that introduced them.
    EXPECTED = {
        "delta": 4.945e-05,
        "gamma": -1.865e-07,
        "vega": -8.635e-03,
        "theta": 2.522e-04,
        "rho": 2.729e-03,
    }

    @pytest.mark.parametrize("greek", list(EXPECTED))
    def test_reference_greeks(self, greek: str) -> None:
        risk = snowball_risk_pde(reference_terms(), REFERENCE_MARKET, STANDARD)
        assert getattr(risk.greeks, greek) == pytest.approx(self.EXPECTED[greek], rel=0.02)

    def test_legacy_wrong_delta_cannot_return(self) -> None:
        """The legacy methods gave -8.4e-05, +1.0e-05 and +9.0e-06.

        The first has the wrong sign; the other two are roughly a fifth of the
        true value. All three are excluded by a 20% band around the correct
        answer.
        """
        risk = snowball_risk_pde(reference_terms(), REFERENCE_MARKET, STANDARD)
        delta = risk.greeks.delta
        for wrong in (-8.4e-05, 1.0e-05, 9.0e-06):
            assert abs(delta - wrong) > 0.2 * abs(delta), f"delta drifted back to {wrong}"

    def test_method_is_recorded(self) -> None:
        """A Greek should never be read without knowing what produced it."""
        assert snowball_risk_pde(reference_terms(), REFERENCE_MARKET, STANDARD).method == "pde-grid"
        mc = snowball_risk_mc(
            reference_terms(),
            REFERENCE_MARKET,
            MonteCarloConfig(paths=20_000, memory_budget_mb=4096.0),
        )
        assert mc.method == "mc-resimulation-crn"

    def test_pde_reports_no_sampling_error(self) -> None:
        """It is deterministic; claiming a standard error would be misleading."""
        assert snowball_risk_pde(reference_terms(), REFERENCE_MARKET, STANDARD).std_error is None

    def test_monte_carlo_theta_is_nan_not_a_wrong_number(self) -> None:
        """Bumping the tenor deletes a knock-out observation, so a simulated
        theta would measure a contract change. nan is the honest answer."""
        mc = snowball_risk_mc(
            reference_terms(),
            REFERENCE_MARKET,
            MonteCarloConfig(paths=20_000, memory_budget_mb=4096.0),
        )
        assert np.isnan(mc.greeks.theta)

    def test_risk_price_matches_the_quote(self) -> None:
        """The risk and the price must come from the same state."""
        terms, market = reference_terms(), REFERENCE_MARKET
        risk = snowball_risk_pde(terms, market, STANDARD)
        assert risk.price == pytest.approx(price_snowball_pde(terms, market, STANDARD).price)

    def test_as_dict_is_json_friendly(self) -> None:
        import json

        payload = snowball_risk_pde(reference_terms(), REFERENCE_MARKET, STANDARD).as_dict()
        json.dumps(payload)
        assert payload["method"] == "pde-grid"
        assert set(payload) >= {"delta", "gamma", "vega", "theta", "rho", "price", "method"}
