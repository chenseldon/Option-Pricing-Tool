"""Tests for the European local-volatility engines.

This file carries the decisive test of the whole phase: **the Dupire round
trip.** A local volatility is derived from an implied volatility surface, so
there is exactly one way to know it is right -- price the vanillas the surface
was built from and check the implied volatilities come back. Dupire's
construction guarantees they must, so any gap is an error somewhere in the
formula, the forward curve, the moneyness mapping, the operator coefficients,
the index alignment, or the discretisation.

Three tests before it exist only to localise a failure when it does happen,
and each is built to see something the others cannot:

``TestConstantVolatility``
    A constant local volatility must reproduce Black-Scholes. Because
    ``atm_total_vol`` and ``boundary_moneyness`` reduce exactly to the flat
    engine's rules, the stand-in lands on the identical grid -- so a
    discrepancy is an indexing or coefficient error, not a different
    discretisation.

``TestTimeVaryingVolatility``
    A *smooth, spot-independent* time-dependent volatility, where the price is
    Black-Scholes at the integrated variance. This is the only test that can
    see whether the theta scheme takes its coefficients from the right time
    levels: with constant coefficients the two levels coincide, so the error
    is invisible, and it is first-order in ``dt`` -- it reads as slow
    convergence rather than as a wrong answer.

``TestTimeDirection``
    Two maturities off the same time-varying surface. A single European price
    cannot detect a reversed time axis, because the integral of a
    time-dependent variance is reversal-invariant; two maturities can.
"""

from __future__ import annotations

import numpy as np
import pytest

from otcpricer.core.exceptions import OtcPricerError
from otcpricer.core.vol import LocalVolatility
from otcpricer.engines.european import price_european_mc, price_european_pde
from otcpricer.engines.mc import MonteCarloConfig
from otcpricer.engines.pde import PDEConfig
from otcpricer.models import bsm
from otcpricer.models.local_vol import local_vol_from_fitted
from otcpricer.models.surface import fit_surface
from tests.test_marketdata import SNAPSHOT_DIR

SPOT, RATE, CARRY = 100.0, 0.03, 0.01


class ConstantLocalVol:
    """A constant volatility wearing the local-volatility interface.

    Reduces to the flat model exactly, which is what makes it a usable
    reference: ``atm_total_vol`` returns ``sigma sqrt(T)`` and
    ``boundary_moneyness`` returns ``-/+ n sigma sqrt(T)``, the same rules the
    flat engine applies.
    """

    def __init__(self, sigma: float, mu: float) -> None:
        self.sigma, self.mu = float(sigma), float(mu)

    def local_vol(self, S: float | np.ndarray, t: float) -> np.ndarray:
        return np.full(np.shape(S), self.sigma, dtype=float)

    def drift(self, t: float) -> float:
        return self.mu

    def forward_growth(self, t_from: float, t_to: float) -> float:
        return float(np.exp(self.mu * (t_to - t_from)))

    def atm_total_vol(self, T: float) -> float:
        return self.sigma * float(np.sqrt(T))

    def boundary_moneyness(self, T: float, n_sigma: float) -> tuple[float, float]:
        pad = n_sigma * self.atm_total_vol(T)
        return -pad, pad

    def with_atm_shift(self, vol_abs: float) -> ConstantLocalVol:
        return ConstantLocalVol(self.sigma + float(vol_abs), self.mu)

    def with_rate_shift(self, rate_abs: float) -> ConstantLocalVol:
        return ConstantLocalVol(self.sigma, self.mu + float(rate_abs))

    def with_carry_shift(self, carry_abs: float) -> ConstantLocalVol:
        return ConstantLocalVol(self.sigma, self.mu - float(carry_abs))


class TimeVaryingVol:
    """``sigma^2(t) = s0^2 (1 + beta t)``: smooth, and independent of spot.

    The European price is then exactly Black-Scholes at the integrated
    variance, which is what makes the time discretisation measurable on its
    own.
    """

    def __init__(self, s0: float, beta: float, mu: float) -> None:
        self.s0, self.beta, self.mu = float(s0), float(beta), float(mu)

    def effective(self, T: float) -> float:
        return self.s0 * float(np.sqrt(1.0 + self.beta * T / 2.0))

    def local_vol(self, S: float | np.ndarray, t: float) -> np.ndarray:
        return np.full(np.shape(S), self.s0 * float(np.sqrt(1.0 + self.beta * t)), dtype=float)

    def drift(self, t: float) -> float:
        return self.mu

    def forward_growth(self, t_from: float, t_to: float) -> float:
        return float(np.exp(self.mu * (t_to - t_from)))

    def atm_total_vol(self, T: float) -> float:
        return self.effective(T) * float(np.sqrt(T))

    def boundary_moneyness(self, T: float, n_sigma: float) -> tuple[float, float]:
        pad = n_sigma * self.atm_total_vol(T)
        return -pad, pad

    def with_atm_shift(self, vol_abs: float) -> TimeVaryingVol:
        return TimeVaryingVol(self.s0 + float(vol_abs), self.beta, self.mu)

    def with_rate_shift(self, rate_abs: float) -> TimeVaryingVol:
        return TimeVaryingVol(self.s0, self.beta, self.mu + float(rate_abs))

    def with_carry_shift(self, carry_abs: float) -> TimeVaryingVol:
        return TimeVaryingVol(self.s0, self.beta, self.mu - float(carry_abs))


# ---------------------------------------------------------------------------
# Localising tests
# ---------------------------------------------------------------------------


class TestConstantVolatility:
    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(ConstantLocalVol(0.2, 0.02), LocalVolatility)

    @pytest.mark.parametrize("sigma", [0.12, 0.20, 0.45])
    @pytest.mark.parametrize(
        ("T", "strike", "kind"),
        [
            (0.08, 90.0, "p"),
            (0.08, 100.0, "c"),
            (0.08, 110.0, "c"),
            (0.5, 90.0, "p"),
            (0.5, 100.0, "c"),
            (2.0, 110.0, "c"),
            (2.0, 100.0, "p"),
        ],
    )
    def test_reproduces_black_scholes(
        self, sigma: float, T: float, strike: float, kind: str
    ) -> None:
        """The variable-coefficient path, with the variability removed.

        Any band-slicing error, node offset, or boundary mistake shows up here
        with nothing else to hide behind, because the grid is identical to the
        flat engine's by construction.
        """
        local_vol = ConstantLocalVol(sigma, RATE - CARRY)
        expected = float(bsm.price(kind, SPOT, strike, T, RATE, sigma, CARRY))
        result = price_european_pde(
            SPOT,
            strike,
            T,
            RATE,
            kind,
            local_vol,
            PDEConfig(space_nodes=800, steps_per_year=2000),
        )
        assert result.price == pytest.approx(expected, rel=1e-3, abs=1e-6)

    def test_strike_lands_exactly_on_a_node(self) -> None:
        """A payoff kink between nodes costs an order of convergence."""
        result = price_european_pde(
            SPOT, 107.0, 0.75, RATE, "c", ConstantLocalVol(0.25, RATE - CARRY)
        )
        assert float(result.S_grid[result.strike_index]) == pytest.approx(107.0, rel=1e-12)

    def test_converges_under_refinement(self) -> None:
        local_vol = ConstantLocalVol(0.2, RATE - CARRY)
        expected = float(bsm.price("c", SPOT, 100.0, 1.0, RATE, 0.2, CARRY))
        errors = [
            abs(
                price_european_pde(
                    SPOT,
                    100.0,
                    1.0,
                    RATE,
                    "c",
                    local_vol,
                    PDEConfig(space_nodes=n, steps_per_year=4 * n),
                ).price
                - expected
            )
            for n in (200, 400, 800)
        ]
        assert errors[-1] < errors[0]
        assert errors[-1] < 1e-4

    def test_monte_carlo_reproduces_black_scholes(self) -> None:
        local_vol = ConstantLocalVol(0.2, RATE - CARRY)
        expected = float(bsm.price("c", SPOT, 100.0, 1.0, RATE, 0.2, CARRY))
        result = price_european_mc(
            SPOT,
            100.0,
            1.0,
            RATE,
            "c",
            local_vol,
            MonteCarloConfig(paths=200_000, steps_per_year=252, seed=7),
        )
        assert abs(result.price - expected) < 4.0 * result.std_error + 0.01

    def test_put_call_parity(self) -> None:
        """Holds whatever the volatility, so it tests the drift and discounting."""
        local_vol = ConstantLocalVol(0.3, RATE - CARRY)
        config = PDEConfig(space_nodes=800, steps_per_year=2000)
        call = price_european_pde(SPOT, 95.0, 1.5, RATE, "c", local_vol, config).price
        put = price_european_pde(SPOT, 95.0, 1.5, RATE, "p", local_vol, config).price
        discount = float(np.exp(-RATE * 1.5))
        forward = SPOT * local_vol.forward_growth(0.0, 1.5)
        assert call - put == pytest.approx(discount * (forward - 95.0), abs=2e-3)


class TestTimeVaryingVolatility:
    """The only tests that can see a wrong time level in the theta scheme."""

    def test_reproduces_the_integrated_variance(self) -> None:
        local_vol = TimeVaryingVol(0.20, 1.5, RATE - CARRY)
        expected = float(bsm.price("c", SPOT, 100.0, 1.0, RATE, local_vol.effective(1.0), CARRY))
        result = price_european_pde(
            SPOT,
            100.0,
            1.0,
            RATE,
            "c",
            local_vol,
            PDEConfig(space_nodes=3000, steps_per_year=1000),
        )
        assert result.price == pytest.approx(expected, rel=2e-5)

    def test_time_discretisation_is_second_order(self) -> None:
        """Crank-Nicolson's order, which a wrong time level would halve.

        Taking the Dirichlet boundary folds from the old level instead of the
        implicit one, or the explicit stencil from the new, degrades this to
        first order. Measured ratios here are 3.67 and 3.03 against the
        theoretical 4.
        """
        local_vol = TimeVaryingVol(0.20, 1.5, RATE - CARRY)
        expected = float(bsm.price("c", SPOT, 100.0, 1.0, RATE, local_vol.effective(1.0), CARRY))
        errors = [
            abs(
                price_european_pde(
                    SPOT,
                    100.0,
                    1.0,
                    RATE,
                    "c",
                    local_vol,
                    PDEConfig(space_nodes=6000, steps_per_year=steps),
                    variance_graded=False,
                    # The step floor exists to protect short-dated accuracy;
                    # here it would override the very counts being refined.
                    min_time_steps=1,
                ).price
                - expected
            )
            for steps in (250, 500, 1000)
        ]
        # Halving dt must cut the error by appreciably more than half.
        assert errors[0] / errors[1] > 2.6
        assert errors[1] / errors[2] > 2.6

    def test_monte_carlo_reproduces_the_integrated_variance(self) -> None:
        local_vol = TimeVaryingVol(0.20, 1.5, RATE - CARRY)
        expected = float(bsm.price("c", SPOT, 100.0, 1.0, RATE, local_vol.effective(1.0), CARRY))
        result = price_european_mc(
            SPOT,
            100.0,
            1.0,
            RATE,
            "c",
            local_vol,
            MonteCarloConfig(paths=200_000, steps_per_year=504, seed=11),
        )
        assert abs(result.price - expected) < 4.0 * result.std_error + 0.02


class TestTimeDirection:
    def test_two_maturities_pin_the_time_axis(self) -> None:
        """A single price cannot detect a reversed time axis.

        Because the integral of a time-dependent variance over ``[0, T]`` is
        unchanged by reversal, one European price is invariant to passing time
        *remaining* where calendar time belongs. Two maturities off the same
        rising surface are not: under reversal the shorter one would pick up the
        variance from the far end of the longer one's window.
        """
        local_vol = TimeVaryingVol(0.20, 3.0, RATE - CARRY)
        config = PDEConfig(space_nodes=2000, steps_per_year=1500)
        for T in (0.5, 1.0):
            expected = float(bsm.price("c", SPOT, 100.0, T, RATE, local_vol.effective(T), CARRY))
            result = price_european_pde(SPOT, 100.0, T, RATE, "c", local_vol, config)
            assert result.price == pytest.approx(expected, rel=5e-5)

        # And the effective volatilities really do differ, so the test has teeth.
        assert local_vol.effective(0.5) < local_vol.effective(1.0) * 0.95


class TestInputHandling:
    @pytest.mark.parametrize(
        ("spot", "strike", "T"),
        [(0.0, 100.0, 1.0), (-1.0, 100.0, 1.0), (100.0, 0.0, 1.0), (100.0, 100.0, 0.0)],
    )
    def test_rejects_non_positive_inputs(self, spot: float, strike: float, T: float) -> None:
        with pytest.raises(OtcPricerError):
            price_european_pde(spot, strike, T, RATE, "c", ConstantLocalVol(0.2, 0.02))

    def test_monte_carlo_rejects_mismatched_draws(self) -> None:
        with pytest.raises(OtcPricerError, match="draws must have shape"):
            price_european_mc(
                SPOT,
                100.0,
                1.0,
                RATE,
                "c",
                ConstantLocalVol(0.2, 0.02),
                MonteCarloConfig(paths=1000, steps_per_year=252),
                draws=np.zeros((10, 10)),
            )

    def test_results_say_what_produced_them(self) -> None:
        pde = price_european_pde(SPOT, 100.0, 1.0, RATE, "c", ConstantLocalVol(0.2, 0.02))
        mc = price_european_mc(
            SPOT,
            100.0,
            1.0,
            RATE,
            "c",
            ConstantLocalVol(0.2, 0.02),
            MonteCarloConfig(paths=2000),
        )
        assert pde.as_dict()["vol_model"] == "dupire-local"
        assert mc.as_dict()["vol_model"] == "dupire-local"


# ---------------------------------------------------------------------------
# The Dupire round trip
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("code", ["510050", "510300"])
class TestDupireRoundTrip:
    """Price the vanillas the surface was fitted to, and get its own smile back.

    The decisive test. Dupire's construction guarantees the round trip in
    continuous time, so the residual is the engine's discretisation error and
    nothing else -- which makes the right budget "small compared with the
    surface's own calibration error". Measured at a production grid:

    ======  ===============  ==============  ============
    chain   worst round trip fit rms         share
    ======  ===============  ==============  ============
    510050  0.441 vp         0.587 vp        75%
    510300  0.0065 vp        0.654 vp        1.0%
    ======  ===============  ==============  ============

    The gap between the two is 510050's rougher variance term structure, which
    costs time-step resolution; it converges away cleanly (0.874, 0.401,
    0.175, 0.0735, 0.0298 volatility points for 400 to 6400 steps).
    """

    CONFIG = PDEConfig(space_nodes=1600, steps_per_year=1008)

    @staticmethod
    def _setup(code: str) -> tuple[object, object, float]:
        if not SNAPSHOT_DIR.exists():
            pytest.skip("no snapshots committed")
        fitted = fit_surface(code, source_kind="snapshot")
        return fitted, local_vol_from_fitted(fitted, tail_drift=0.02), fitted.surface_input.rate

    def test_reproduces_the_implied_surface(self, code: str) -> None:
        fitted, local_vol, rate = self._setup(code)
        budget = fitted.calibration.rms_vol_error * 100  # type: ignore[attr-defined]

        worst = 0.0
        for slice_input in fitted.surface_input.usable_slices:  # type: ignore[attr-defined]
            T = slice_input.year_fraction
            forward = float(local_vol.forward(T))  # type: ignore[attr-defined]
            for moneyness in (0.90, 0.95, 1.00, 1.05, 1.10):
                strike = forward * moneyness
                kind = "c" if moneyness >= 1.0 else "p"
                target = float(local_vol.surface.implied_vol(np.log(moneyness), T))  # type: ignore[attr-defined]
                price = price_european_pde(
                    local_vol.spot,
                    strike,
                    T,
                    rate,
                    kind,
                    local_vol,
                    self.CONFIG,  # type: ignore[attr-defined,arg-type]
                ).price
                # Invert at that expiry's own forward -- S = F with q = r is
                # Black-76. Passing spot here instead would measure the carry,
                # which on this data is -4% a year, and report it as a model
                # error.
                recovered = float(
                    bsm.implied_volatility(kind, price, forward, strike, T, rate, rate)
                )
                worst = max(worst, abs(recovered - target) * 100)

        assert worst < budget, (
            f"round trip errs by {worst:.4f} vol points, above the surface's own "
            f"calibration rms of {budget:.4f}"
        )

    def test_monte_carlo_agrees_with_the_pde(self, code: str) -> None:
        """Different discretisations of the same local volatility.

        The Monte Carlo carries a discretisation bias the PDE does not, so this
        is a genuine cross-check rather than a restatement.
        """
        fitted, local_vol, rate = self._setup(code)
        T = float(fitted.surface_input.usable_slices[-1].year_fraction)  # type: ignore[attr-defined]
        strike = float(local_vol.forward(T))  # type: ignore[attr-defined]

        pde = price_european_pde(
            local_vol.spot,
            strike,
            T,
            rate,
            "c",
            local_vol,
            self.CONFIG,  # type: ignore[attr-defined,arg-type]
        ).price
        mc = price_european_mc(
            local_vol.spot,
            strike,
            T,
            rate,
            "c",
            local_vol,  # type: ignore[attr-defined,arg-type]
            MonteCarloConfig(paths=100_000, steps_per_year=1008, seed=3),
        )
        assert abs(pde - mc.price) < 4.0 * mc.std_error + 0.002 * strike

    def test_skew_survives_the_round_trip(self, code: str) -> None:
        """Not just the level: the shape has to come back too.

        A bug that collapsed the smile to its at-the-money level would still
        pass a test written only at the money.
        """
        _, local_vol, rate = self._setup(code)
        T = 0.4055
        forward = float(local_vol.forward(T))  # type: ignore[attr-defined]

        recovered = []
        for moneyness in (0.90, 1.10):
            strike = forward * moneyness
            kind = "c" if moneyness >= 1.0 else "p"
            price = price_european_pde(
                local_vol.spot,
                strike,
                T,
                rate,
                kind,
                local_vol,
                self.CONFIG,  # type: ignore[attr-defined,arg-type]
            ).price
            recovered.append(
                float(bsm.implied_volatility(kind, price, forward, strike, T, rate, rate))
            )

        # Downside implied volatility above the upside: the equity skew.
        assert recovered[0] > recovered[1]
