"""Tests for the snowball coupon solver and the knocked-in cross-check.

Two things are established here, and they carry very different weights.

**The affinity is a proof, not a tolerance.** The price is exactly linear in the
coupon, so the solver is exact rather than iterative, and the test that says so
asserts collinearity at machine precision. An iterative solver could only ever
be checked against a tolerance, and would never reveal that the relationship is
linear at all.

**The knocked-in surface finally has a second engine.** Before this,
``knocked_in`` existed only on the PDE, so the value of a note that has already
breached -- a real operational state, revalued daily -- was the one price in
this project with no independent confirmation. It now agrees with the Monte
Carlo to 0.13% relative, and the residual is dominated by the PDE's own
first-order convergence rather than by either engine being wrong.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from otcpricer.core.exceptions import CalibrationError
from otcpricer.core.market import EquityMarket
from otcpricer.engines.mc import MonteCarloConfig, price_snowball_mc
from otcpricer.engines.pde import PDEConfig, price_snowball_pde
from otcpricer.products import snowball_quote
from otcpricer.products.snowball import SnowballTerms
from otcpricer.products.snowball_quote import (
    MAX_AFFINITY_RESIDUAL,
    MIN_COUPON_ANNUITY,
    decompose_par_coupon,
    solve_par_coupon,
    verify_par_coupon_mc,
)

# The project's reference trade, shared with tests/test_snowball.py.
MARKET = EquityMarket(S=5000.0, r=0.02, sigma=0.20, q=0.01)
TERMS = SnowballTerms(S0=5000.0, T=1.0, ko_pct=1.03, ki_pct=0.75, coupon_pa=0.20)
CONFIG = PDEConfig(space_nodes=1600, steps_per_year=1008)

MC_CONFIG = MonteCarloConfig(paths=400_000, steps_per_year=252, seed=42, antithetic=True)


class TestAffinity:
    """The property the solver is built on, asserted at machine precision."""

    def test_the_price_is_collinear_in_the_coupon(self) -> None:
        """Residual against a straight line, across six coupons from 0 to 1.

        Measured at 7.8e-16 -- machine precision. This is the strongest
        statement available about the solver, and it is only available *because*
        the relationship is exact: a bisection would converge to a tolerance and
        could never say this.

        It holds because the knock-out and knock-in events depend on the spot
        path against fixed barriers and never on the coupon, while every
        coupon-bearing term in the engine is linear in it.
        """
        coupons = [0.0, 0.05, 0.10, 0.20, 0.50, 1.0]
        prices = [
            price_snowball_pde(replace(TERMS, coupon_pa=c), MARKET, CONFIG).price for c in coupons
        ]

        intercept = prices[0]
        slope = prices[-1] - prices[0]
        worst = max(abs(p - (slope * c + intercept)) for c, p in zip(coupons, prices, strict=True))

        assert worst < 1e-14, f"worst residual {worst:.3e} -- the price is not affine"

    def test_the_monte_carlo_is_affine_too(self) -> None:
        """On shared paths, so this is the payoff's own linearity, not luck."""
        coupons = [0.0, 0.10, 0.50, 1.0]
        prices = [
            price_snowball_mc(replace(TERMS, coupon_pa=c), MARKET, MC_CONFIG).price for c in coupons
        ]
        intercept = prices[0]
        slope = prices[-1] - prices[0]
        worst = max(abs(p - (slope * c + intercept)) for c, p in zip(coupons, prices, strict=True))
        assert worst < 1e-14

    def test_the_knock_in_leg_does_not_move_with_the_coupon(self) -> None:
        """Which is the economic reason affinity holds.

        At a zero coupon the note *is* its knock-in leg, so this pins the
        intercept to something independently meaningful rather than to an
        artefact of the fit.
        """
        solution = solve_par_coupon(TERMS, MARKET, CONFIG)
        at_zero = price_snowball_pde(replace(TERMS, coupon_pa=0.0), MARKET, CONFIG).price
        assert solution.knock_in_value == pytest.approx(at_zero, abs=1e-15)
        assert solution.knock_in_value < 0.0, "a reachable downside must cost something"

    def test_the_solver_measures_its_own_assumption_on_every_call(self) -> None:
        """The third solve, and the reason it exists.

        Two solves answer the question; the third checks that the question was
        the right one. Measured at 1e-16 on this trade, four decades inside the
        limit -- so the limit is not what makes this pass.
        """
        solution = solve_par_coupon(TERMS, MARKET, CONFIG)
        assert solution.affinity_residual < 1e-14
        assert solution.affinity_residual <= MAX_AFFINITY_RESIDUAL
        assert solution.as_dict()["affinity_residual"] < 1e-14

    def test_a_coupon_dependent_price_is_refused_rather_than_solved(self) -> None:
        """What the invariant is actually for.

        The failure this guards against is not noisy -- a coupon-dependent grid
        or boundary would make ``-B/A`` the wrong answer while every other
        check in this file still passed, because they all rest on the same
        assumption. So affinity is injected with a break and the solver must
        refuse.

        The perturbation is quadratic and tiny (1e-9 of price at c=1), which is
        far below any tolerance elsewhere in the project and still three orders
        above the limit here. That gap is the point: this catches breaks long
        before they are large enough to notice in a price.
        """
        real = snowball_quote.price_snowball_pde

        def bent(terms, market, config=None, **kwargs):  # type: ignore[no-untyped-def]
            result = real(terms, market, config, **kwargs)
            return replace(result, price=result.price + 1e-9 * terms.coupon_pa**2)

        snowball_quote.price_snowball_pde = bent  # type: ignore[assignment]
        try:
            with pytest.raises(CalibrationError, match="no longer affine"):
                solve_par_coupon(TERMS, MARKET, CONFIG)
        finally:
            snowball_quote.price_snowball_pde = real  # type: ignore[assignment]

    def test_the_two_legs_price_every_other_coupon_for_free(self) -> None:
        """``price_at`` against the engine, across the quotable range.

        This is the practical dividend of exactness: a quote ladder costs three
        solves in total rather than three per rung. If affinity were merely
        approximate this would drift, and the drift would grow with distance
        from the solved coupon -- so the far ends are included deliberately.
        """
        solution = solve_par_coupon(TERMS, MARKET, CONFIG)
        for coupon in (0.0, 0.03, solution.par_coupon, 0.25, 0.60, 1.0):
            engine = price_snowball_pde(replace(TERMS, coupon_pa=coupon), MARKET, CONFIG).price
            assert solution.price_at(coupon) == pytest.approx(engine, abs=1e-14)

        assert solution.price_at(solution.par_coupon) == pytest.approx(0.0, abs=1e-14)


class TestTheSolver:
    def test_the_solved_coupon_prices_to_zero(self) -> None:
        """To machine precision, not to a tolerance.

        Measured residual 8.5e-17 on the reference trade.
        """
        solution = solve_par_coupon(TERMS, MARKET, CONFIG)
        assert solution.verified_price == pytest.approx(0.0, abs=1e-12)
        assert abs(solution.residual) < 1e-12
        assert solution.method == "affine-exact"

    def test_the_reference_trade_quotes_at_seven_percent(self) -> None:
        """Pinned, and worth reading rather than just asserting.

        7.12% at a zero basis and 20% volatility. Real market snowballs quote
        15-25%, and the gap is not an error -- it is the futures discount,
        which this trade does not have. See TestDecomposition.
        """
        solution = solve_par_coupon(TERMS, MARKET, CONFIG)
        assert solution.par_coupon == pytest.approx(0.0711873, rel=1e-4)
        assert solution.coupon_annuity == pytest.approx(0.4897596, rel=1e-5)

    def test_the_coupon_leg_reads_as_discounted_time_alive(self) -> None:
        """``A`` is the present value of one unit of coupon.

        Since the coupon accrues linearly at ``coupon * t``, ``A`` is the
        discounted expected time the note stays alive -- so it must sit between
        zero and the tenor, and below ``T`` because most paths terminate early.
        """
        solution = solve_par_coupon(TERMS, MARKET, CONFIG)
        assert 0.0 < solution.coupon_annuity < TERMS.T

    def test_a_target_other_than_zero_is_honoured(self) -> None:
        """A desk quoting away from par still needs the coupon."""
        for target in (-0.01, 0.0, 0.02):
            solution = solve_par_coupon(TERMS, MARKET, CONFIG, target=target)
            assert solution.verified_price == pytest.approx(target, abs=1e-12)

    def test_terms_coupon_is_ignored(self) -> None:
        """It is the unknown. Passing a different one must change nothing."""
        first = solve_par_coupon(TERMS, MARKET, CONFIG)
        second = solve_par_coupon(replace(TERMS, coupon_pa=0.99), MARKET, CONFIG)
        assert first.par_coupon == pytest.approx(second.par_coupon, abs=1e-15)

    @pytest.mark.parametrize("ki_pct", [0.60, 0.70, 0.75, 0.85])
    def test_a_nearer_knock_in_demands_a_higher_coupon(self, ki_pct: float) -> None:
        """Monotone, and the direction is the whole economics of the product.

        Raising the knock-in barrier makes the downside more reachable, so the
        buyer must be paid more to accept it.
        """
        coupons = [
            solve_par_coupon(replace(TERMS, ki_pct=k), MARKET, CONFIG).par_coupon
            for k in (0.60, 0.70, 0.75, 0.85)
        ]
        assert all(b > a for a, b in zip(coupons, coupons[1:], strict=False))

    def test_more_volatility_demands_a_higher_coupon(self) -> None:
        low = solve_par_coupon(TERMS, replace(MARKET, sigma=0.15), CONFIG).par_coupon
        high = solve_par_coupon(TERMS, replace(MARKET, sigma=0.30), CONFIG).par_coupon
        assert high > low

    def test_an_unquotable_note_is_refused_rather_than_returned_as_infinity(self) -> None:
        """When the coupon leg is worthless, no coupon reaches the target.

        Driven by making the knock-out unreachable while the knock-in is
        certain: the note can neither terminate early nor survive clean, so
        the coupon buys nothing. Dividing by that would hand back a confident
        ``inf``.
        """
        hopeless = SnowballTerms(
            S0=5000.0,
            T=1.0,
            ko_pct=50.0,  # unreachable
            ki_pct=49.0,  # already breached in practice at any spot
            coupon_pa=0.20,
            lockup_months=0,
        )
        with pytest.raises(CalibrationError, match="no coupon makes this note"):
            solve_par_coupon(hopeless, MARKET, CONFIG)

    def test_the_guard_threshold_is_not_doing_real_work_on_a_normal_trade(self) -> None:
        """The reference trade's coupon leg is eleven orders of magnitude above it."""
        solution = solve_par_coupon(TERMS, MARKET, CONFIG)
        assert solution.coupon_annuity > 1e11 * MIN_COUPON_ANNUITY

    def test_serialises_with_its_own_caveat(self) -> None:
        payload = solve_par_coupon(TERMS, MARKET, CONFIG).as_dict()
        assert payload["par_coupon_pct"] == pytest.approx(7.11873, rel=1e-4)
        assert "affine" in str(payload["note"])


@pytest.mark.slow
class TestMonteCarloVerification:
    """What establishes the PDE price itself, which the solver cannot."""

    def test_the_simulation_also_prices_to_zero_at_the_solved_coupon(self) -> None:
        """Measured z = +0.54 on the reference trade at 400,000 paths.

        Three standard errors, which is the honest bar: the solver is exact
        *given* the finite-difference price, so this is the only test that can
        catch the finite-difference price being wrong.
        """
        solution = solve_par_coupon(TERMS, MARKET, CONFIG)
        check = verify_par_coupon_mc(solution, TERMS, MARKET, MC_CONFIG)

        assert check["passes"] is True
        assert abs(float(check["z_score"])) < 3.0

    def test_the_two_engines_solve_the_same_coupon(self) -> None:
        """Independently, each from its own two prices.

        The Monte Carlo's own ``-B/A`` came out 7.1007% against the PDE's
        7.1187% -- 1.8 basis points apart, which is its sampling error rather
        than a disagreement.
        """
        pde = solve_par_coupon(TERMS, MARKET, CONFIG).par_coupon

        at_zero = price_snowball_mc(replace(TERMS, coupon_pa=0.0), MARKET, MC_CONFIG).price
        at_one = price_snowball_mc(replace(TERMS, coupon_pa=1.0), MARKET, MC_CONFIG).price
        simulated = -at_zero / (at_one - at_zero)

        assert simulated == pytest.approx(pde, abs=5e-4)


class TestDecomposition:
    """Splitting the quote into what pays for it."""

    def test_the_basis_nearly_doubles_the_quotable_coupon(self) -> None:
        """The market view, measured: "the deeper the discount, the higher the coupon".

        At MO's measured basis of about 7.7% a year the reference trade quotes
        13.40% against 7.12% at a zero basis -- so the futures discount
        accounts for roughly 47% of the coupon. That is first-order, not a
        refinement, which is why the old default of zero mattered.
        """
        with_basis = replace(MARKET, basis_pa=0.077)
        split = decompose_par_coupon(TERMS, with_basis, CONFIG)

        assert split.without_basis == pytest.approx(0.0711873, rel=1e-3)
        assert split.par_coupon == pytest.approx(0.1339660, rel=1e-3)
        assert split.basis_part > split.without_basis * 0.8
        assert split.basis_share == pytest.approx(0.469, abs=0.01)

    def test_a_zero_basis_leaves_nothing_to_attribute(self) -> None:
        split = decompose_par_coupon(TERMS, MARKET, CONFIG)
        assert split.basis_pa == 0.0
        assert split.basis_part == pytest.approx(0.0, abs=1e-12)
        assert split.basis_share == pytest.approx(0.0, abs=1e-10)

    def test_a_deeper_discount_buys_more_coupon(self) -> None:
        shares = [
            decompose_par_coupon(TERMS, replace(MARKET, basis_pa=b), CONFIG).basis_part
            for b in (0.0, 0.01, 0.04, 0.077)
        ]
        assert all(b > a for a, b in zip(shares, shares[1:], strict=False))

    def test_serialises_with_the_interaction_caveat(self) -> None:
        split = decompose_par_coupon(TERMS, replace(MARKET, basis_pa=0.077), CONFIG)
        payload = split.as_dict()
        assert payload["basis_part_pct"] > 0.0
        # The split is a comparison, not an orthogonal decomposition.
        assert "not additive" in str(payload["note"])


class TestKnockedInAcrossBothEngines:
    """The surface that had never been cross-checked."""

    @pytest.mark.parametrize("spot", [3600.0, 4000.0, 4500.0, 5000.0, 5400.0])
    @pytest.mark.parametrize("sigma", [0.20, 0.35])
    def test_the_engines_agree_on_a_knocked_in_note(self, spot: float, sigma: float) -> None:
        """On a budget built from both engines' error, not a relative bar.

        A *relative* tolerance is the wrong instrument here, and trying one
        showed why: the knocked-in price crosses zero. At spot 5000 and 20%
        volatility it is +0.0005, so a 0.5% relative bar is 2.5e-06 -- far
        below either engine's resolution -- while the same bar is loose at
        spot 3600 where the price is -0.26.

        So the budget is absolute, in fractions of notional, and is the sum of
        the two things that actually differ:

        - **three Monte Carlo standard errors**, and
        - **5e-04 for the PDE's own discretisation**, measured: on the worst
          point the grid converges first-order through -0.104714, -0.105031,
          -0.105189, -0.105268 as it doubles, so roughly 2e-04 remains at this
          resolution, and the Monte Carlo shows no step bias at all.

        The largest observed gap across these ten points was 6.5e-04 against a
        budget of about 1.1e-03, so the bar passes on the real agreement while
        still catching any genuine semantic divergence -- which is the point,
        since each engine defines the knocked-in state independently.
        """
        market = replace(MARKET, S=spot, sigma=sigma)
        grid = price_snowball_pde(TERMS, market, CONFIG, knocked_in=True).price
        result = price_snowball_mc(TERMS, market, MC_CONFIG, knocked_in=True)

        budget = 3.0 * result.std_error + 5e-4
        assert abs(result.price - grid) < budget, (
            f"PDE {grid:.8f} vs MC {result.price:.8f} +/- {result.std_error:.2e}: "
            f"gap {abs(result.price - grid):.2e} exceeds budget {budget:.2e}"
        )

    def test_every_simulated_path_is_knocked_in(self) -> None:
        """The flag means the breach is in the past, not that it is likely."""
        result = price_snowball_mc(TERMS, MARKET, MC_CONFIG, knocked_in=True)
        assert result.ki_prob == 1.0
        # And none can be clean: every path either knocks out or takes the put.
        assert result.ko_prob + result.ki_no_ko_prob == pytest.approx(1.0, abs=1e-12)

    def test_a_knocked_in_note_is_worth_less_than_a_clean_one(self) -> None:
        """In both engines, which is the cheapest sanity check available."""
        clean_pde = price_snowball_pde(TERMS, MARKET, CONFIG, knocked_in=False).price
        ki_pde = price_snowball_pde(TERMS, MARKET, CONFIG, knocked_in=True).price
        assert ki_pde < clean_pde

        clean_mc = price_snowball_mc(TERMS, MARKET, MC_CONFIG, knocked_in=False).price
        ki_mc = price_snowball_mc(TERMS, MARKET, MC_CONFIG, knocked_in=True).price
        assert ki_mc < clean_mc

    def test_a_knocked_in_note_can_still_knock_out(self) -> None:
        """Which is the one thing the legacy two-state solver got wrong.

        It modelled the knocked-in state as a plain short put, silently
        dropping the coupon on every path that breached and then rallied
        through the barrier. Here a knocked-in note well above the barrier is
        worth a *positive* amount, because it will terminate on its coupon.
        """
        high = replace(MARKET, S=5400.0)
        assert price_snowball_mc(TERMS, high, MC_CONFIG, knocked_in=True).ko_prob > 0.5
        assert price_snowball_pde(TERMS, high, CONFIG, knocked_in=True).price > 0.0

    def test_the_knocked_in_coupon_can_also_be_solved(self) -> None:
        """A desk revaluing a breached note still needs to know its par coupon."""
        solution = solve_par_coupon(TERMS, MARKET, CONFIG, knocked_in=True)
        assert solution.verified_price == pytest.approx(0.0, abs=1e-12)
        # It must be dearer than the clean quote: the downside is certain now.
        clean = solve_par_coupon(TERMS, MARKET, CONFIG, knocked_in=False)
        assert solution.par_coupon > clean.par_coupon


class TestUnderLocalVolatility:
    def test_the_coupon_solves_on_a_fitted_surface(self) -> None:
        """Affinity does not depend on the volatility model.

        The coupon multiplies the same coupon-free quantities whatever the
        diffusion is, so the solver is exact under a local volatility too --
        and the method string says which model produced the answer.
        """
        from otcpricer.models.local_vol import local_vol_from_fitted
        from otcpricer.models.surface import fit_surface

        try:
            surface = local_vol_from_fitted(fit_surface("510300"), spot=4.1, tail_drift=0.02)
        except Exception:  # pragma: no cover - needs the committed snapshots
            pytest.skip("no snapshots committed")

        terms = SnowballTerms(S0=4.1, T=1.0, ko_pct=1.03, ki_pct=0.75, coupon_pa=0.20)
        market = EquityMarket(S=4.1, r=0.02, sigma=0.20, q=0.0)
        config = PDEConfig(space_nodes=1600, steps_per_year=1008)

        solution = solve_par_coupon(terms, market, config, local_vol=surface)
        assert solution.verified_price == pytest.approx(0.0, abs=1e-10)
        assert solution.method == "affine-exact+dupire-local"
        assert 0.0 < solution.par_coupon < 1.0

    def test_the_surface_changes_the_quote(self) -> None:
        """Otherwise the surface is not reaching the solver at all."""
        from otcpricer.models.local_vol import local_vol_from_fitted
        from otcpricer.models.surface import fit_surface

        try:
            surface = local_vol_from_fitted(fit_surface("510300"), spot=4.1, tail_drift=0.02)
        except Exception:  # pragma: no cover
            pytest.skip("no snapshots committed")

        terms = SnowballTerms(S0=4.1, T=1.0, ko_pct=1.03, ki_pct=0.75, coupon_pa=0.20)
        atm = float(surface.atm_total_vol(1.0))
        market = EquityMarket(S=4.1, r=0.02, sigma=atm, q=0.0)
        config = PDEConfig(space_nodes=1600, steps_per_year=1008)

        flat = solve_par_coupon(terms, market, config).par_coupon
        local = solve_par_coupon(terms, market, config, local_vol=surface).par_coupon
        assert local != flat
        assert abs(local / flat - 1.0) > 0.01, "the smile should move the quote materially"


class TestNumpyFree:
    def test_no_stray_array_leaks_into_the_result(self) -> None:
        """Everything on CouponSolution must be a plain float, for JSON."""
        solution = solve_par_coupon(TERMS, MARKET, CONFIG)
        for value in (
            solution.par_coupon,
            solution.coupon_annuity,
            solution.knock_in_value,
            solution.verified_price,
        ):
            assert isinstance(value, float)
            assert not isinstance(value, np.ndarray)
