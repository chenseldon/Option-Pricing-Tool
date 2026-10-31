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

from otcpricer.core.market import EquityMarket
from otcpricer.engines.pde import PDEConfig, _LocalVolGrid, _SnowballGrid, price_snowball_pde
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
