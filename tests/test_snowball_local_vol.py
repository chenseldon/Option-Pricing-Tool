"""The snowball PDE under local volatility.

The central test is a degenerate one. A constant local volatility must
reproduce the flat engine *exactly*, and because ``atm_total_vol`` and
``boundary_moneyness`` reduce to the flat rules, both paths land on the same
grid node for node. So the comparison isolates the thing that is hard to get
right -- band slicing, node offsets, time levels, boundary growth factors --
from discretisation, which would otherwise mask all of it. It holds to 8.3e-17
on the price and 2.7e-15 across the whole surface.

The second theme is what the local-volatility path must *not* do: read
``market.sigma``. The surface carries both the volatility and the drift, so a
volatility bump on the market object has to be structurally incapable of
moving the price. That is what makes a spurious vega impossible rather than
merely avoided -- and it matters because ``sigma`` also drives the flat grid's
geometry, so a bump would otherwise produce a small, plausible, meaningless
number rather than a clean zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from otcpricer.core.exceptions import CalibrationError
from otcpricer.core.market import EquityMarket
from otcpricer.engines.mc import MonteCarloConfig, price_snowball_mc
from otcpricer.engines.pde import (
    LOCAL_VOL_PDE_CONFIG,
    PDEConfig,
    _LocalVolGrid,
    _SnowballGrid,
    price_snowball_pde,
)
from otcpricer.engines.risk import snowball_risk_mc, snowball_risk_pde
from otcpricer.models.local_vol import local_vol_from_fitted
from otcpricer.models.surface import fit_surface
from otcpricer.products.snowball import SnowballTerms
from tests.test_european import ConstantLocalVol
from tests.test_marketdata import SNAPSHOT_DIR

TERMS = SnowballTerms(S0=5000.0, T=1.0, ko_pct=1.03, ki_pct=0.75, coupon_pa=0.20)
MARKET = EquityMarket(S=5000.0, r=0.03, sigma=0.20, q=0.0)
CONFIG = PDEConfig(space_nodes=1600, steps_per_year=1008)

#: The Phase 1 reference price. Recorded before any of this phase's work, and
#: repeated here so a regression in the flat path is caught in this file too.
FLAT_PRICE = 0.0683263509055402


class TestDegenerateEquivalence:
    def test_constant_local_vol_reproduces_the_flat_engine(self) -> None:
        """The decisive localising test.

        Identical grid, so any difference is an indexing or coefficient error
        with nothing to hide behind.
        """
        flat = price_snowball_pde(TERMS, MARKET, CONFIG)
        local = price_snowball_pde(
            TERMS, MARKET, CONFIG, local_vol=ConstantLocalVol(MARKET.sigma, MARKET.drift)
        )

        assert np.array_equal(flat.S_grid, local.S_grid), "grids must coincide"
        assert flat.space_nodes == local.space_nodes
        assert flat.time_steps == local.time_steps
        assert local.price == pytest.approx(flat.price, abs=1e-14)

    def test_the_whole_surface_agrees_not_just_the_price(self) -> None:
        """A price can agree by luck; two thousand nodes cannot."""
        flat = price_snowball_pde(TERMS, MARKET, CONFIG)
        local = price_snowball_pde(
            TERMS, MARKET, CONFIG, local_vol=ConstantLocalVol(MARKET.sigma, MARKET.drift)
        )

        for name in ("V_no_ki", "V_ki", "V_no_ki_next_step", "V_ki_next_step"):
            assert np.allclose(getattr(flat, name), getattr(local, name), atol=1e-12, rtol=0.0), (
                name
            )

    def test_also_in_the_knocked_in_state(self) -> None:
        flat = price_snowball_pde(TERMS, MARKET, CONFIG, knocked_in=True)
        local = price_snowball_pde(
            TERMS,
            MARKET,
            CONFIG,
            knocked_in=True,
            local_vol=ConstantLocalVol(MARKET.sigma, MARKET.drift),
        )
        assert local.price == pytest.approx(flat.price, abs=1e-14)

    @pytest.mark.parametrize("sigma", [0.12, 0.35])
    @pytest.mark.parametrize("T", [0.5, 2.0])
    def test_across_volatilities_and_tenors(self, sigma: float, T: float) -> None:
        terms = SnowballTerms(S0=5000.0, T=T, ko_pct=1.03, ki_pct=0.75, coupon_pa=0.20)
        market = EquityMarket(S=5000.0, r=0.03, sigma=sigma, q=0.01, basis_pa=0.02)
        flat = price_snowball_pde(terms, market, CONFIG)
        local = price_snowball_pde(
            terms, market, CONFIG, local_vol=ConstantLocalVol(sigma, market.drift)
        )
        assert local.price == pytest.approx(flat.price, abs=1e-13)

    def test_the_basis_reaches_the_boundary_too(self) -> None:
        """``bc_ki_low`` is a discounted forward intrinsic, so it carries the drift.

        Leaving it on a constant carry while the operator follows a forward
        curve would be inconsistent, and the error leaks inward through the
        diffusion term rather than staying at the boundary. A non-zero basis
        makes the two expressions differ, so this would catch it.
        """
        market = EquityMarket(S=5000.0, r=0.03, sigma=0.2, q=0.01, basis_pa=0.05)
        flat = price_snowball_pde(TERMS, market, CONFIG)
        local = price_snowball_pde(
            TERMS, market, CONFIG, local_vol=ConstantLocalVol(0.2, market.drift)
        )
        assert local.price == pytest.approx(flat.price, abs=1e-13)


class TestFlatPathUnchanged:
    def test_default_still_takes_the_scalar_path(self) -> None:
        """The hard constraint of the phase, asserted rather than assumed."""
        assert price_snowball_pde(TERMS, MARKET, CONFIG).price == FLAT_PRICE

    def test_the_two_grids_agree_on_geometry_under_a_constant_surface(self) -> None:
        """The geometry was extracted to be shared; this pins the extraction."""
        flat = _SnowballGrid(TERMS, MARKET, CONFIG)
        local = _LocalVolGrid(TERMS, MARKET, CONFIG, ConstantLocalVol(MARKET.sigma, MARKET.drift))
        for name in ("n_nodes", "dx", "ki_index", "ko_index", "n_steps", "dt", "steps_per_year"):
            assert getattr(flat, name) == getattr(local, name), name
        assert np.array_equal(flat.x_grid, local.x_grid)

    def test_barriers_stay_on_exact_nodes_under_local_volatility(self) -> None:
        """Phase 1's hardest-won property, and it must survive the new grid."""
        if not SNAPSHOT_DIR.exists():
            pytest.skip("no snapshots committed")
        local_vol = local_vol_from_fitted(
            fit_surface("510050", source_kind="snapshot"), spot=5000.0, tail_drift=0.0
        )
        grid = _LocalVolGrid(TERMS, MARKET, CONFIG, local_vol)

        assert float(grid.S_grid[grid.ki_index]) == pytest.approx(TERMS.ki_barrier, rel=1e-12)
        assert float(grid.S_grid[grid.ko_index]) == pytest.approx(TERMS.ko_barrier, rel=1e-12)


class TestNoSpuriousVega:
    def test_bumping_market_sigma_does_not_move_the_price(self) -> None:
        """The structural guarantee behind the vega design.

        On the local-volatility path the surface supplies the volatility, so
        ``market.sigma`` must not be read at all. If it were -- and it would
        be, through the flat grid's padding rule -- a vega differenced by
        bumping it would come back not as a clean zero but as a small number
        produced by the grid changing shape. Plausible, meaningless, and
        exactly the kind of thing this project has spent two phases removing.
        """
        local_vol = ConstantLocalVol(0.20, MARKET.drift)
        base = price_snowball_pde(TERMS, MARKET, CONFIG, local_vol=local_vol)
        for bumped_sigma in (0.05, 0.60):
            bumped = price_snowball_pde(
                TERMS,
                EquityMarket(S=5000.0, r=0.03, sigma=bumped_sigma, q=0.0),
                CONFIG,
                local_vol=local_vol,
            )
            assert bumped.price == base.price
            assert bumped.space_nodes == base.space_nodes

    def test_but_the_rate_still_matters(self) -> None:
        """Discounting is the market's job even when volatility is not."""
        local_vol = ConstantLocalVol(0.20, MARKET.drift)
        base = price_snowball_pde(TERMS, MARKET, CONFIG, local_vol=local_vol)
        shifted = price_snowball_pde(
            TERMS, MARKET.with_shift(rate_abs=0.01), CONFIG, local_vol=local_vol
        )
        assert shifted.price != base.price


class TestCoefficientCache:
    def test_each_time_level_is_evaluated_once(self) -> None:
        """The roll cache: the new level of step k is the old level of step k+1.

        Two states are stepped per iteration, so a naive implementation would
        evaluate four levels where one suffices.
        """
        grid = _LocalVolGrid(TERMS, MARKET, CONFIG, ConstantLocalVol(0.2, MARKET.drift))
        calls = {"n": 0}
        original = grid._coefficients

        def counted(t: float):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            return original(t)

        grid._coefficients = counted  # type: ignore[method-assign]
        for k in range(10):
            grid._levels(k)  # once for V_ki
            grid._levels(k)  # once for V_no_ki, same step
        # One evaluation for the first old level, then one per new level.
        assert calls["n"] == 11

    def test_the_cache_is_correctness_neutral(self) -> None:
        """A miss recomputes from scratch, so it cannot go stale."""
        local_vol = ConstantLocalVol(0.2, MARKET.drift)
        warm = _LocalVolGrid(TERMS, MARKET, CONFIG, local_vol)
        cold = _LocalVolGrid(TERMS, MARKET, CONFIG, local_vol)

        for k in range(5):
            warm._levels(k)
        # Jump around, forcing misses.
        for k in (7, 3, 7, 0):
            rolled, _ = warm._levels(k)
            fresh, _ = cold._levels(k)
            for a, b in zip(rolled, fresh, strict=True):
                assert np.array_equal(a, b)


@pytest.mark.slow
@pytest.mark.parametrize("code", ["510050", "510300"])
class TestRealSurface:
    def test_prices_and_reports_its_basis(self, code: str) -> None:
        if not SNAPSHOT_DIR.exists():
            pytest.skip("no snapshots committed")
        fitted = fit_surface(code, source_kind="snapshot")
        local_vol = local_vol_from_fitted(fitted, spot=100.0, tail_drift=0.0)
        terms = SnowballTerms(S0=100.0, T=2.0, ko_pct=1.03, ki_pct=0.80, coupon_pa=0.20)
        market = EquityMarket(S=100.0, r=0.03, sigma=0.20, q=0.0)

        result = price_snowball_pde(terms, market, CONFIG, local_vol=local_vol)

        # A snowball is worth a fraction of notional, and a positive one here.
        assert 0.0 < result.price < 0.5
        assert np.all(np.isfinite(result.V_no_ki))
        assert np.all(np.isfinite(result.V_ki))

    def test_differs_materially_from_the_flat_price(self, code: str) -> None:
        """The point of the phase.

        Local volatility at the 80% knock-in is roughly twice the at-the-money
        level, so the short put leg is priced off a far wider distribution than
        a single at-the-money quote implies. A local-volatility price that
        matched the flat one would mean the surface was not reaching the
        engine.
        """
        if not SNAPSHOT_DIR.exists():
            pytest.skip("no snapshots committed")
        fitted = fit_surface(code, source_kind="snapshot")
        local_vol = local_vol_from_fitted(fitted, spot=100.0, tail_drift=0.0)
        terms = SnowballTerms(S0=100.0, T=1.0, ko_pct=1.03, ki_pct=0.80, coupon_pa=0.20)
        atm = local_vol.atm_total_vol(1.0)
        market = EquityMarket(S=100.0, r=0.03, sigma=atm, q=0.03 - local_vol.drift(0.5))

        flat = price_snowball_pde(terms, market, CONFIG).price
        local = price_snowball_pde(terms, market, CONFIG, local_vol=local_vol).price

        assert abs(local - flat) > 0.05 * abs(flat)
        # The knock-in region really does carry more volatility.
        assert float(local_vol.local_vol(80.0, 0.5)) > 1.5 * float(local_vol.local_vol(100.0, 0.5))


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------


class TestMonteCarloPath:
    def test_constant_local_vol_matches_the_exact_scheme(self) -> None:
        """Validates the loop mechanics, and cannot do more than that.

        For a *constant* volatility the log-Euler step is already exact -- the
        increment does not depend on the state -- so this agrees to 2.9e-15 and
        is silent about discretisation bias by construction. It is still worth
        having: it proves the sequential loop reproduces the cumulative sum,
        which is the part that was restructured.
        """
        config = MonteCarloConfig(paths=50_000, steps_per_year=252, seed=42)
        exact = price_snowball_mc(TERMS, MARKET, config)
        looped = price_snowball_mc(
            TERMS, MARKET, config, local_vol=ConstantLocalVol(MARKET.sigma, MARKET.drift)
        )
        assert looped.price == pytest.approx(exact.price, abs=1e-12)
        assert looped.std_error == pytest.approx(exact.std_error, abs=1e-12)

    def test_does_not_read_market_sigma(self) -> None:
        config = MonteCarloConfig(paths=20_000, steps_per_year=252, seed=1)
        local_vol = ConstantLocalVol(0.20, MARKET.drift)
        base = price_snowball_mc(TERMS, MARKET, config, local_vol=local_vol)
        bumped = price_snowball_mc(
            TERMS,
            EquityMarket(S=5000.0, r=0.03, sigma=0.55, q=0.0),
            config,
            local_vol=local_vol,
        )
        assert bumped.price == base.price

    def test_rejects_mismatched_draws(self) -> None:
        with pytest.raises(ValueError, match="draws must have shape"):
            price_snowball_mc(
                TERMS,
                MARKET,
                MonteCarloConfig(paths=1000, steps_per_year=252),
                draws=np.zeros((5, 5)),
                local_vol=ConstantLocalVol(0.2, MARKET.drift),
            )


@pytest.mark.slow
class TestEngineAgreement:
    """PDE against Monte Carlo under local volatility.

    A genuine cross-check: different discretisations of different formulations,
    and under local volatility the Monte Carlo additionally carries a bias the
    PDE does not.
    """

    def test_engines_agree_on_a_converged_grid(self) -> None:
        if not SNAPSHOT_DIR.exists():
            pytest.skip("no snapshots committed")
        local_vol = local_vol_from_fitted(
            fit_surface("510050", source_kind="snapshot"), spot=100.0, tail_drift=0.0
        )
        terms = SnowballTerms(S0=100.0, T=1.0, ko_pct=1.03, ki_pct=0.80, coupon_pa=0.20)
        market = EquityMarket(S=100.0, r=0.03, sigma=0.20, q=0.0)

        pde = price_snowball_pde(terms, market, LOCAL_VOL_PDE_CONFIG, local_vol=local_vol).price
        mc = price_snowball_mc(
            terms,
            market,
            MonteCarloConfig(paths=200_000, steps_per_year=504, seed=42),
            local_vol=local_vol,
        )
        # Phase 1's budget for the flat engines was three standard errors.
        assert abs(pde - mc.price) < 3.0 * mc.std_error

    def test_the_flat_defaults_would_not_have_been_enough(self) -> None:
        """Why LOCAL_VOL_PDE_CONFIG exists, asserted so it cannot be quietly dropped.

        The flat grid budget was validated for a constant operator. On a Dupire
        surface it is short by an order of magnitude in the measured gap, which
        is the kind of thing that would otherwise be discovered as a pricing
        discrepancy rather than as a discretisation one.
        """
        if not SNAPSHOT_DIR.exists():
            pytest.skip("no snapshots committed")
        local_vol = local_vol_from_fitted(
            fit_surface("510050", source_kind="snapshot"), spot=100.0, tail_drift=0.0
        )
        terms = SnowballTerms(S0=100.0, T=1.0, ko_pct=1.03, ki_pct=0.80, coupon_pa=0.20)
        market = EquityMarket(S=100.0, r=0.03, sigma=0.20, q=0.0)

        coarse = price_snowball_pde(terms, market, CONFIG, local_vol=local_vol).price
        fine = price_snowball_pde(terms, market, LOCAL_VOL_PDE_CONFIG, local_vol=local_vol).price
        # Measured 0.00349 on this trade. Against Monte Carlo the flat defaults
        # sit at 14.5 standard errors and this grid at 2.0, so the difference
        # is not a refinement nicety -- it is most of the disagreement.
        assert abs(fine - coarse) > 0.003


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------


class TestRiskUnderLocalVolatility:
    """Greeks on the local-volatility path, and what had to change for them.

    Two of the five do not care where the volatility came from: delta and gamma
    are differenced off the solution surface, and theta off two adjacent time
    levels. The other two had to be rebuilt, because both were bumping a market
    field that no longer drives anything.
    """

    CONFIG = PDEConfig(space_nodes=1600, steps_per_year=1008)

    def test_every_greek_matches_the_flat_engine_under_a_constant_surface(self) -> None:
        """The degenerate check, extended to risk.

        Each Greek agrees to machine precision, which is only possible because
        the surface shift and the market bump coincide for a constant surface:
        ``with_atm_shift`` adds to the one volatility, ``with_rate_shift`` adds
        to the one drift. Anything bumping the wrong quantity shows up here.
        """
        terms = SnowballTerms(S0=100.0, T=1.0, ko_pct=1.03, ki_pct=0.80, coupon_pa=0.20)
        market = EquityMarket(S=100.0, r=0.03, sigma=0.20, q=0.0)
        local_vol = ConstantLocalVol(0.20, market.drift)

        flat = snowball_risk_pde(terms, market, self.CONFIG)
        local = snowball_risk_pde(terms, market, self.CONFIG, local_vol=local_vol)

        for name in ("delta", "gamma", "vega", "theta", "rho"):
            assert getattr(local.greeks, name) == pytest.approx(
                getattr(flat.greeks, name), abs=1e-12
            ), name
        assert local.carry_rho == pytest.approx(flat.carry_rho, abs=1e-12)

    def test_carry_rho_moves_the_drift_without_moving_the_rate(self) -> None:
        """The same trap as rho, one layer down, and a sharper one.

        ``ForwardCurve`` stores a rate and a drift, and its carry is their
        difference. So ``with_rate_shift``, which moves both together, leaves
        the carry *exactly* unchanged -- using it here would return a carry_rho
        of zero while every other Greek looked right.

        Under a constant surface ``with_carry_shift`` and a market bump
        coincide, so the degenerate pins it: the local path must reproduce the
        flat path's carry_rho, which is nowhere near zero.
        """
        terms = SnowballTerms(S0=100.0, T=1.0, ko_pct=1.03, ki_pct=0.80, coupon_pa=0.20)
        market = EquityMarket(S=100.0, r=0.03, sigma=0.20, q=0.0)
        local_vol = ConstantLocalVol(0.20, market.drift)

        flat = snowball_risk_pde(terms, market, self.CONFIG).carry_rho
        local = snowball_risk_pde(terms, market, self.CONFIG, local_vol=local_vol).carry_rho

        assert flat is not None and local is not None
        assert abs(flat) > 1e-4, "the flat carry_rho must be large enough to be worth matching"
        assert local == pytest.approx(flat, abs=1e-12)

        # And the trap itself: a rate shift leaves the surface's carry alone.
        surface = ConstantLocalVol(0.20, market.drift)
        assert surface.with_rate_shift(0.01).drift(0.5) == pytest.approx(
            surface.drift(0.5) + 0.01, abs=1e-15
        )
        assert surface.with_carry_shift(0.01).drift(0.5) == pytest.approx(
            surface.drift(0.5) - 0.01, abs=1e-15
        )

    def test_rho_moves_the_drift_and_not_only_the_discounting(self) -> None:
        """The defect this caught, pinned so it cannot come back.

        On this path the drift lives on the surface's forward curve, so bumping
        only ``market.r`` leaves rho missing its drift half. Measured on this
        trade that turned +0.0036 into -0.0001 -- a sign change, not a small
        distortion. A rho that disagrees in sign with the flat engine under a
        constant surface is the signature.
        """
        terms = SnowballTerms(S0=100.0, T=1.0, ko_pct=1.03, ki_pct=0.80, coupon_pa=0.20)
        market = EquityMarket(S=100.0, r=0.03, sigma=0.20, q=0.0)
        local_vol = ConstantLocalVol(0.20, market.drift)

        flat = snowball_risk_pde(terms, market, self.CONFIG).greeks.rho
        local = snowball_risk_pde(terms, market, self.CONFIG, local_vol=local_vol).greeks.rho

        assert flat > 0.0
        assert local == pytest.approx(flat, abs=1e-12)

    def test_the_method_says_how_vega_was_taken(self) -> None:
        """A number is never read without knowing what produced it.

        The local-volatility vega is a *different derivative* from the flat
        one -- a parallel shift of the at-the-money level also changes
        curvature at every maturity, because SSVI's curvature scale depends on
        the variance level. Comparing the two as if they were the same quantity
        would be a mistake, so the label makes the difference visible.
        """
        terms = SnowballTerms(S0=100.0, T=1.0, ko_pct=1.03, ki_pct=0.80, coupon_pa=0.20)
        market = EquityMarket(S=100.0, r=0.03, sigma=0.20, q=0.0)

        assert snowball_risk_pde(terms, market, self.CONFIG).method == "pde-grid"
        assert (
            snowball_risk_pde(
                terms, market, self.CONFIG, local_vol=ConstantLocalVol(0.2, market.drift)
            ).method
            == "pde-grid+ssvi-parallel-vega"
        )

    def test_monte_carlo_risk_agrees_in_the_degenerate_case(self) -> None:
        config = MonteCarloConfig(paths=40_000, steps_per_year=252, seed=42)
        terms = SnowballTerms(S0=100.0, T=1.0, ko_pct=1.03, ki_pct=0.80, coupon_pa=0.20)
        market = EquityMarket(S=100.0, r=0.03, sigma=0.20, q=0.0)

        flat = snowball_risk_mc(terms, market, config)
        local = snowball_risk_mc(
            terms, market, config, local_vol=ConstantLocalVol(0.20, market.drift)
        )
        for name in ("delta", "gamma", "vega", "rho"):
            assert getattr(local.greeks, name) == pytest.approx(
                getattr(flat.greeks, name), abs=1e-12
            ), name

    @pytest.mark.slow
    def test_real_surface_vega_is_finite_and_signed_like_the_flat_one(self) -> None:
        """A snowball is short volatility, whichever surface it is priced off."""
        if not SNAPSHOT_DIR.exists():
            pytest.skip("no snapshots committed")
        local_vol = local_vol_from_fitted(
            fit_surface("510050", source_kind="snapshot"), spot=100.0, tail_drift=0.0
        )
        terms = SnowballTerms(S0=100.0, T=1.0, ko_pct=1.03, ki_pct=0.80, coupon_pa=0.20)
        atm = local_vol.atm_total_vol(1.0)
        market = EquityMarket(S=100.0, r=0.03, sigma=atm, q=0.03 - local_vol.drift(0.5))

        risk = snowball_risk_pde(terms, market, LOCAL_VOL_PDE_CONFIG, local_vol=local_vol)
        assert np.isfinite(risk.greeks.vega)
        assert risk.greeks.vega < 0.0
        assert np.isfinite(risk.greeks.delta)
        assert np.isfinite(risk.greeks.theta)

    def test_a_shift_that_breaks_no_arbitrage_is_refused(self) -> None:
        """The guard, firing.

        A surface whose variance term structure the shift would tip
        non-monotone must raise rather than return a vega computed off a
        calendar-arbitraged surface.
        """
        from otcpricer.marketdata.forward import ForwardCurve
        from otcpricer.models.local_vol import LocalVolSurface
        from otcpricer.models.svi import SSVIParams, SSVISurface

        inverted = LocalVolSurface(
            SSVISurface(
                SSVIParams(
                    rho=-0.2, eta=1.0, gamma=0.4, maturities=(0.1, 1.0), thetas=(0.025, 0.0256)
                )
            ),
            ForwardCurve(maturities=(1.0,), forwards=(100.0,), tail_drift=0.0, rate=0.0),
            100.0,
        )
        terms = SnowballTerms(S0=100.0, T=1.0, ko_pct=1.03, ki_pct=0.80, coupon_pa=0.20)
        market = EquityMarket(S=100.0, r=0.03, sigma=0.20, q=0.0)

        with pytest.raises(CalibrationError, match="non-monotone"):
            snowball_risk_pde(
                terms,
                market,
                PDEConfig(space_nodes=400, steps_per_year=252),
                local_vol=inverted,
                vol_bump=0.10,
            )
