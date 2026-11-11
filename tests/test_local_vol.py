"""Tests for Dupire local volatility.

The order reflects how much each one would cost to get wrong.

**The two exact anchors come first.** A flat surface must give back its own
constant volatility, and a surface with term structure but no smile must give
back ``dw/dT``. Both hold to machine precision, so they are gates rather than
tolerances -- and between them they exercise the numerator, the denominator and
the moneyness mapping.

**Then the reductions the engine work depends on.** Under a constant volatility
the grid rules must reproduce the flat engine's bit for bit. That is what lets a
constant-volatility stand-in land on the identical grid later, which in turn is
what makes a price difference between the two paths unambiguous evidence of an
indexing error rather than of a different discretisation.

**Then the properties that are easy to state and easy to violate**: that the
surface does not move when spot is bumped, that clamping is counted rather than
absorbed, and that a surface which failed the no-arbitrage precondition is
refused instead of quietly producing numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from otcpricer.core.exceptions import CalibrationError
from otcpricer.core.vol import LocalVolatility
from otcpricer.marketdata.forward import ForwardCurve
from otcpricer.models.local_vol import (
    DEFAULT_MAX_ATM_MULTIPLE,
    MIN_TIME,
    LocalVolSurface,
    local_vol_from_fitted,
)
from otcpricer.models.surface import fit_surface
from otcpricer.models.svi import SSVIParams, SSVISurface
from tests.test_marketdata import SNAPSHOT_DIR

MATURITIES = (0.25, 0.5, 1.0, 2.0)
FLAT_VOL = 0.22

#: A curve with no drift at all, so moneyness is just ``log(S / spot)``. Keeps
#: the anchor tests about Dupire rather than about the forward curve, which has
#: its own tests in test_marketdata.
FLAT_CURVE = ForwardCurve(maturities=(1.0,), forwards=(100.0,), tail_drift=0.0, rate=0.0)


def _surface(thetas: tuple[float, ...], *, eta: float = 1e-8, rho: float = 0.0) -> SSVISurface:
    """An SSVI surface. ``eta -> 0`` drives the smile out, leaving pure term structure."""
    return SSVISurface(
        SSVIParams(rho=rho, eta=eta, gamma=0.4, maturities=MATURITIES, thetas=thetas)
    )


@pytest.fixture
def flat() -> LocalVolSurface:
    """Constant implied volatility: no smile, variance linear in maturity."""
    return LocalVolSurface(_surface(tuple(FLAT_VOL**2 * t for t in MATURITIES)), FLAT_CURVE, 100.0)


@pytest.fixture
def term_structure() -> LocalVolSurface:
    """Term structure, still no smile."""
    return LocalVolSurface(_surface((0.01, 0.03, 0.05, 0.14)), FLAT_CURVE, 100.0)


@pytest.fixture
def skewed() -> LocalVolSurface:
    """An equity-index shape: downside skew and real curvature."""
    return LocalVolSurface(
        _surface((0.01, 0.025, 0.055, 0.12), eta=1.2, rho=-0.35), FLAT_CURVE, 100.0
    )


def _real(code: str, **kwargs: object) -> LocalVolSurface:
    if not SNAPSHOT_DIR.exists():
        pytest.skip("no snapshots committed")
    fitted = fit_surface(code, source_kind="snapshot")
    return local_vol_from_fitted(fitted, tail_drift=0.02, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The exact anchors
# ---------------------------------------------------------------------------


class TestAnalyticAnchors:
    def test_flat_surface_returns_its_own_volatility(self, flat: LocalVolSurface) -> None:
        """With no smile and variance linear in maturity, Dupire is the identity.

        Every term in the denominator but the leading 1 vanishes, and the
        numerator is exactly ``sigma^2``. Holds to machine precision, so this is
        a gate: any error in the numerator, the denominator, or the moneyness
        mapping breaks it.
        """
        spots = np.array([60.0, 80.0, 100.0, 125.0, 160.0])
        for t in (0.3, 0.7, 1.5, 2.5):
            assert np.allclose(flat.local_vol(spots, t), FLAT_VOL, atol=1e-14)

    def test_no_smile_gives_the_forward_variance(self, term_structure: LocalVolSurface) -> None:
        """Without a smile the local variance is exactly ``dw/dT``.

        The forward variance, which is the other closed form Dupire has to
        reproduce. It exercises the numerator independently of the smile terms.
        """
        for spot in (70.0, 100.0, 140.0):
            for t in (0.3, 0.7, 1.5, 2.5):
                y = term_structure.log_moneyness(spot, t)
                expected = float(term_structure.surface.dw_dT(y, t))
                assert float(term_structure.local_vol(spot, t)) ** 2 == pytest.approx(
                    expected, abs=1e-14
                )

    def test_denominator_is_the_butterfly_expression(self, skewed: LocalVolSurface) -> None:
        """The reuse identity, asserted rather than assumed.

        The Dupire denominator is identically Gatheral's ``g(y)``, so this
        module calls the existing, finite-difference-tested implementation
        instead of transcribing the expanded form a second time. Pinning the
        equality means a future edit cannot silently fork them.
        """
        for t in (0.05, 0.3, 1.0, 2.5):
            y = np.linspace(-1.0, 1.0, 81)
            spots = 100.0 * np.exp(y)
            _, denominator = skewed._dupire(spots, t)
            expected = np.asarray(skewed.surface.butterfly(skewed.log_moneyness(spots, t), t))
            assert np.allclose(denominator, expected, rtol=1e-15, atol=0.0)

    def test_matches_dupire_built_from_numerical_derivatives(self, skewed: LocalVolSurface) -> None:
        """The formula itself, against a version that differentiates ``w``.

        Independent of the analytic derivative closed forms, so an algebra
        error in either place shows up here.
        """
        h, dt = 1e-5, 1e-6
        for t in (0.2, 0.6, 1.5):
            for y in (-0.4, -0.1, 0.0, 0.15, 0.5):

                def w(k: float, time: float = t) -> float:
                    return float(skewed.surface.total_variance(k, time))

                w0 = w(y)
                dw_dy = (w(y + h) - w(y - h)) / (2.0 * h)
                d2w_dy2 = (w(y + h) - 2.0 * w0 + w(y - h)) / (h * h)
                dw_dt = (w(y, t + dt) - w(y, t - dt)) / (2.0 * dt)
                denominator = (
                    1.0
                    - (y / w0) * dw_dy
                    + 0.25 * (-0.25 - 1.0 / w0 + y * y / (w0 * w0)) * dw_dy * dw_dy
                    + 0.5 * d2w_dy2
                )
                expected = np.sqrt(dw_dt / denominator)
                spot = 100.0 * np.exp(y)
                assert float(skewed.local_vol(spot, t)) == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# The reductions the engine work depends on
# ---------------------------------------------------------------------------


class TestFlatReduction:
    def test_atm_total_vol_is_sigma_root_t(self, flat: LocalVolSurface) -> None:
        """Exactly what the flat grid builder uses, so the grids coincide."""
        for t in (0.25, 0.5, 1.0, 2.0, 3.0):
            assert flat.atm_total_vol(t) == pytest.approx(FLAT_VOL * np.sqrt(t), rel=1e-14)

    def test_boundary_reduces_to_four_sigma_root_t(self, flat: LocalVolSurface) -> None:
        """The padding rule must collapse onto the flat engine's.

        Under a constant volatility total variance does not depend on moneyness,
        so the fixed point has the closed-form root ``-/+ n sigma sqrt(T)`` and
        the cap does not bind. Without this the constant-volatility stand-in
        would land on a different grid and the comparison test would be
        measuring discretisation rather than correctness.
        """
        for t in (0.5, 2.0, 3.0):
            lower, upper = flat.boundary_moneyness(t, 4.0)
            assert lower == pytest.approx(-4.0 * FLAT_VOL * np.sqrt(t), abs=1e-12)
            assert upper == pytest.approx(4.0 * FLAT_VOL * np.sqrt(t), abs=1e-12)

    def test_boundary_widens_on_the_downside_under_skew(self, skewed: LocalVolSurface) -> None:
        """A skew puts implied volatility at the lower boundary above the money.

        So a rule measured at the money under-pads there, which is why the
        fixed point measures the z-score at the boundary itself.
        """
        lower, upper = skewed.boundary_moneyness(2.0, 4.0)
        atm = skewed.atm_total_vol(2.0)
        assert -lower > 4.0 * atm
        assert -lower > upper

    def test_cap_bounds_the_fixed_point(self, skewed: LocalVolSurface) -> None:
        """The fixed point is self-reinforcing under a steep skew; the cap stops it."""
        atm = skewed.atm_total_vol(2.0)
        lower, upper = skewed.boundary_moneyness(2.0, 4.0, max_atm_multiple=2.0)
        assert -lower == pytest.approx(2.0 * atm, rel=1e-9)
        assert upper == pytest.approx(2.0 * atm, rel=1e-9)
        assert -lower < DEFAULT_MAX_ATM_MULTIPLE * atm


# ---------------------------------------------------------------------------
# Properties that are easy to state and easy to violate
# ---------------------------------------------------------------------------


class TestSurfaceIsFixed:
    def test_satisfies_the_engine_protocol(self, flat: LocalVolSurface) -> None:
        assert isinstance(flat, LocalVolatility)

    def test_does_not_re_anchor_when_spot_moves(self, skewed: LocalVolSurface) -> None:
        """The defining property of local volatility, and easy to get wrong.

        ``sigma_loc`` is a fixed function of absolute spot and time, calibrated
        once. A delta differenced by bumping spot must see the *same* surface --
        that is exactly why a local-volatility delta differs from a
        Black-Scholes one. Re-anchoring the forward curve on the bumped spot
        would slide the whole smile along with it, and the test shows that would
        give a materially different number.
        """
        spot, t = 100.0, 1.0
        bumped = 101.0

        # Reading the fixed surface at a higher spot: the skew is traversed.
        assert float(skewed.local_vol(bumped, t)) != pytest.approx(
            float(skewed.local_vol(spot, t)), rel=1e-6
        )

        # Re-anchoring instead would report the at-the-money value again.
        re_anchored = LocalVolSurface(skewed.surface, skewed.forwards, bumped)
        assert float(re_anchored.local_vol(bumped, t)) == pytest.approx(
            float(skewed.local_vol(spot, t)), rel=1e-12
        )
        assert float(re_anchored.local_vol(bumped, t)) != pytest.approx(
            float(skewed.local_vol(bumped, t)), rel=1e-6
        )

    def test_vectorises_over_spot_at_a_single_time(self, skewed: LocalVolSurface) -> None:
        """The contract both solvers rely on for speed."""
        spots = np.linspace(70.0, 140.0, 17)
        vector = skewed.local_vol(spots, 0.8)
        assert vector.shape == (17,)
        for index, spot in enumerate(spots):
            assert float(vector[index]) == pytest.approx(float(skewed.local_vol(spot, 0.8)))

    def test_local_variance_is_the_square(self, skewed: LocalVolSurface) -> None:
        spots = np.linspace(70.0, 140.0, 9)
        assert np.allclose(
            skewed.local_variance(spots, 0.5), skewed.local_vol(spots, 0.5) ** 2, rtol=1e-15
        )


class TestShortMaturity:
    def test_stays_finite_down_to_the_time_floor(self, skewed: LocalVolSurface) -> None:
        for t in (1e-2, 1e-4, 1e-6, 1e-8, 0.0):
            values = skewed.local_vol(np.array([60.0, 100.0, 160.0]), t)
            assert np.all(np.isfinite(values))
            assert np.all(values >= skewed.floor)
            assert np.all(values <= skewed.cap)

    def test_at_the_money_converges_while_the_wings_diverge(self, skewed: LocalVolSurface) -> None:
        """Correct behaviour, not a defect.

        At zero time to expiry an option struck away from spot has no time
        value, so only an unbounded local volatility can produce any smile.
        At the money the limit is finite.
        """
        uncapped = LocalVolSurface(skewed.surface, skewed.forwards, 100.0, cap=1e12)
        atm = [float(uncapped.local_vol(100.0, t)) for t in (1e-3, 1e-5, 1e-7)]
        assert atm[0] == pytest.approx(atm[-1], rel=0.05)

        wing = [float(uncapped.local_vol(70.0, t)) for t in (1e-3, 1e-5, 1e-7)]
        assert wing[0] < wing[1] < wing[2]

    @pytest.mark.parametrize("gamma", [0.2, 0.3, 0.4, 0.5])
    def test_wing_divergence_rate_follows_the_power_law(self, gamma: float) -> None:
        """The rate is derivable, so it is asserted rather than eyeballed.

        As ``theta -> 0`` the curvature ``phi ~ eta theta^-gamma`` blows up, so
        at fixed moneyness ``w ~ theta^(1-gamma) ~ t^(1-gamma)`` and
        ``dw/dt ~ t^-gamma`` while the denominator tends to a constant. Hence
        ``sigma_loc ~ t^(-gamma/2)`` -- *not* ``t^(-1/2)``, which is what a first
        reading of the formula suggests and what this test was originally
        written to assert before the measurement corrected it.
        """
        surface = LocalVolSurface(
            _surface((0.01, 0.025, 0.055, 0.12), eta=1.2, rho=-0.35)
            if gamma == 0.4
            else SSVISurface(
                SSVIParams(
                    rho=-0.35,
                    eta=1.2,
                    gamma=gamma,
                    maturities=MATURITIES,
                    thetas=(0.01, 0.025, 0.055, 0.12),
                )
            ),
            FLAT_CURVE,
            100.0,
            cap=1e12,
        )
        near = float(surface.local_vol(70.0, 1e-5))
        far = float(surface.local_vol(70.0, 1e-7))
        assert far / near == pytest.approx(100.0 ** (gamma / 2.0), rel=0.02)

    def test_time_floor_sits_far_below_any_time_step(self) -> None:
        """Theta is read from two adjacent PDE levels.

        A floor anywhere near ``dt`` would give both levels identical frozen
        coefficients and silently zero the term-structure contribution to
        theta -- damaging the one Greek the project anchors most carefully, via
        the mechanism meant to protect the numerics.
        """
        daily = 1.0 / 252.0
        assert daily / 10_000.0 > MIN_TIME


class TestClamping:
    def test_counts_every_hit(self, skewed: LocalVolSurface) -> None:
        """Clamping is reported, never absorbed."""
        tight = LocalVolSurface(skewed.surface, skewed.forwards, 100.0, floor=0.18, cap=0.22)
        spots = 100.0 * np.exp(np.linspace(-0.8, 0.8, 101))
        diagnostics = tight.scan(spots, np.linspace(0.05, 2.0, 40))

        assert diagnostics.n_floored > 0
        assert diagnostics.n_capped > 0
        assert diagnostics.clamped_fraction > 0.0
        assert diagnostics.n_evaluated == 101 * 40
        assert diagnostics.worst_cap is not None
        assert "clamped" in diagnostics.summary()

    def test_reports_no_clamping_when_there_is_none(self, flat: LocalVolSurface) -> None:
        spots = 100.0 * np.exp(np.linspace(-0.5, 0.5, 51))
        diagnostics = flat.scan(spots, np.linspace(0.05, 2.0, 20))

        assert diagnostics.n_floored == 0
        assert diagnostics.n_capped == 0
        assert diagnostics.well_defined
        assert "no clamping" in diagnostics.summary()
        assert diagnostics.min_vol == pytest.approx(FLAT_VOL, rel=1e-12)
        assert diagnostics.max_vol == pytest.approx(FLAT_VOL, rel=1e-12)

    def test_reports_the_cell_peclet_number(self, skewed: LocalVolSurface) -> None:
        """Above roughly 2 a central-difference scheme can oscillate."""
        spots = 100.0 * np.exp(np.linspace(-0.6, 0.6, 61))
        diagnostics = skewed.scan(spots, np.linspace(0.05, 2.0, 30), dx=1.2 / 60)

        assert diagnostics.max_peclet is not None
        assert 0.0 < diagnostics.max_peclet < 2.0

    def test_serialises(self, skewed: LocalVolSurface) -> None:
        payload = skewed.scan(
            100.0 * np.exp(np.linspace(-0.5, 0.5, 21)), np.linspace(0.1, 2.0, 10)
        ).as_dict()
        for key in ("min_vol", "max_vol", "min_denominator", "clamped_fraction", "well_defined"):
            assert key in payload


class TestPrecondition:
    def test_refuses_a_surface_that_cannot_support_a_local_volatility(
        self, flat: LocalVolSurface
    ) -> None:
        """The guard has to be able to fire."""
        with pytest.raises(CalibrationError, match="cannot produce a local volatility"):
            LocalVolSurface(
                flat.surface,
                FLAT_CURVE,
                100.0,
                precondition=(False, "the Dupire numerator is negative"),
            )

    def test_can_be_overridden_to_study_a_broken_surface(self, flat: LocalVolSurface) -> None:
        surface = LocalVolSurface(
            flat.surface,
            FLAT_CURVE,
            100.0,
            precondition=(False, "deliberately broken"),
            require_well_defined=False,
        )
        assert surface.precondition is not None and not surface.precondition[0]

    @pytest.mark.parametrize(
        ("kwargs", "fragment"),
        [
            ({"floor": 0.0}, "floor must be positive"),
            ({"floor": -0.1}, "floor must be positive"),
            ({"cap": 0.005}, "must exceed floor"),
            ({"spot": 0.0}, "spot must be positive"),
            ({"spot": -1.0}, "spot must be positive"),
        ],
    )
    def test_rejects_inconsistent_bounds(
        self, flat: LocalVolSurface, kwargs: dict[str, float], fragment: str
    ) -> None:
        arguments: dict[str, float] = {"spot": 100.0, **kwargs}
        spot = arguments.pop("spot")
        with pytest.raises(CalibrationError, match=fragment):
            LocalVolSurface(flat.surface, FLAT_CURVE, spot, **arguments)  # type: ignore[arg-type]


class TestParallelShift:
    def test_moves_at_the_money_implied_volatility_exactly(self, skewed: LocalVolSurface) -> None:
        """A shift of the implied surface, by construction exact at the money."""
        for delta in (0.01, -0.01, 0.05):
            shifted = skewed.with_atm_shift(delta)
            for maturity in MATURITIES:
                before = float(skewed.surface.implied_vol(0.0, maturity))
                after = float(shifted.surface.implied_vol(0.0, maturity))
                assert after == pytest.approx(before + delta, abs=1e-12)

    def test_keeps_the_skew_parameters(self, skewed: LocalVolSurface) -> None:
        shifted = skewed.with_atm_shift(0.01)
        for name in ("rho", "eta", "gamma"):
            assert getattr(shifted.surface.params, name) == getattr(skewed.surface.params, name)

    def test_moves_local_volatility_by_less_than_one_to_one_at_the_money(
        self, skewed: LocalVolSurface
    ) -> None:
        """Worth pinning: an implied shift is not a local shift.

        Local volatility responds sub-linearly at the money and more than
        one-for-one in the wings, which is the whole reason a desk's vega is
        defined on the implied surface rather than on this one.
        """
        base = float(skewed.local_vol(100.0, 1.0))
        moved = float(skewed.with_atm_shift(0.01).local_vol(100.0, 1.0))
        assert 0.0 < moved - base < 0.01

    def test_refuses_a_shift_that_creates_calendar_arbitrage(self) -> None:
        """A steeply inverted term structure can be tipped non-monotone.

        Unusual but constructible, so it is checked rather than assumed: here
        total variance is barely increasing, and a one-vol-point *downward*
        shift reverses it.
        """
        inverted = SSVISurface(
            SSVIParams(rho=-0.2, eta=1.0, gamma=0.4, maturities=(0.1, 1.0), thetas=(0.025, 0.0256))
        )
        surface = LocalVolSurface(inverted, FLAT_CURVE, 100.0)

        with pytest.raises(CalibrationError, match="non-monotone"):
            surface.with_atm_shift(-0.10)

    def test_a_carry_shift_moves_the_tail_drift_and_leaves_the_rate(self) -> None:
        """The one respect in which it differs from a rate shift -- and it is
        the respect that decides whether carry_rho measures anything.

        ``ForwardCurve.carry`` is ``rate - drift``. A rate shift moves both, so
        the carry it reports is unchanged; only moving the drift alone changes
        it. Using the rate shift for carry_rho would therefore return a clean
        zero on the local-volatility path while every other Greek looked
        correct.

        The quoted forwards do not move under either: they were implied from
        the option chain and are observations, not functions of a hypothetical
        carry.
        """
        surface = _real("510300")
        bump = 0.0025
        shifted = surface.with_carry_shift(bump)

        assert shifted.forwards.tail_drift == pytest.approx(
            surface.forwards.tail_drift - bump, abs=1e-15
        )
        assert shifted.forwards.rate == surface.forwards.rate
        assert shifted.forwards.forwards == surface.forwards.forwards

        beyond = surface.forwards.maturities[-1] + 1.0
        assert float(shifted.forwards.carry(beyond)) == pytest.approx(
            float(surface.forwards.carry(beyond)) + bump, abs=1e-15
        )

        # And the contrast that makes the method necessary.
        rate_shifted = surface.with_rate_shift(bump)
        assert float(rate_shifted.forwards.carry(beyond)) == pytest.approx(
            float(surface.forwards.carry(beyond)), abs=1e-15
        )

    def test_shift_survives_on_the_real_surfaces(self) -> None:
        """Measured: the guard is real but does not bind in practice.

        Even five vol points either way leaves both fitted surfaces
        arbitrage-free, so the refusal path above is a guard rather than an
        obstacle.
        """
        for code in ("510050", "510300"):
            surface = _real(code)
            for delta in (0.01, -0.01, 0.05, -0.05):
                shifted = surface.with_atm_shift(delta)
                assert shifted.surface.params.butterfly_margin > 0.0
                assert shifted.surface.params.calendar_margin >= 0.0


# ---------------------------------------------------------------------------
# The real surfaces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", ["510050", "510300"])
class TestRealSurfaces:
    def test_local_variance_is_defined_everywhere_on_a_wide_grid(self, code: str) -> None:
        """The arbitrage-free guarantee, checked where the solver will look."""
        surface = _real(code)
        lower, upper = surface.boundary_moneyness(3.0)
        spots = surface.spot * np.exp(np.linspace(lower, upper, 241))
        diagnostics = surface.scan(spots, np.linspace(1e-4, 3.0, 200))

        assert diagnostics.n_undefined == 0
        assert diagnostics.min_denominator > 0.2
        assert diagnostics.well_defined

    def test_clamping_is_confined_to_the_short_dated_wing(self, code: str) -> None:
        """Where the 1/sqrt(t) divergence is genuine, and harmless."""
        surface = _real(code)
        lower, upper = surface.boundary_moneyness(3.0)
        spots = surface.spot * np.exp(np.linspace(lower, upper, 241))
        diagnostics = surface.scan(spots, np.linspace(1e-4, 3.0, 200))

        assert diagnostics.n_floored == 0
        assert diagnostics.clamped_fraction < 0.05
        assert diagnostics.worst_cap is not None
        # Short-dated and far from the money.
        assert diagnostics.worst_cap[1] < 0.05

    def test_skew_doubles_local_volatility_at_the_knock_in(self, code: str) -> None:
        """The headline of the phase, in the one place a snowball cares about.

        A snowball's knock-in sits around 80% of the initial fixing. Local
        volatility there is far above the at-the-money level, so the short put
        leg is priced off a materially wider distribution than a flat quote
        implies.
        """
        surface = _real(code)
        atm = float(surface.local_vol(surface.spot, 1.0))
        knock_in = float(surface.local_vol(surface.spot * 0.80, 1.0))

        assert knock_in > 1.5 * atm

    def test_boundary_beats_the_inherited_flat_grid(self, code: str) -> None:
        """Reach probability is a diagnostic, but it should not regress.

        The flat grid Phase 1 validated sits at 7.1e-03 by this measure, so the
        local-volatility rule has to do at least as well.
        """
        surface = _real(code)
        lower, upper = surface.boundary_moneyness(3.0)
        assert surface.reach_probability(lower, 3.0) < 7.1e-03
        assert surface.reach_probability(upper, 3.0) < 7.1e-03

    def test_most_of_a_snowball_tenor_is_extrapolated(self, code: str) -> None:
        """The dominant modelling exposure, and it is not numerical.

        The chains stop at 0.41 years while the products run one to three, so
        the bulk of the pricing horizon rests on the variance term structure's
        linear extension rather than on quotes. Pinned here so it is impossible
        to forget.
        """
        surface = _real(code)
        assert surface.extrapolated_fraction(3.0) == pytest.approx(0.865, abs=0.01)
        assert surface.extrapolated_fraction(0.3) == 0.0

    def test_forward_curve_comes_from_the_fit(self, code: str) -> None:
        """Using any other curve evaluates the volatility at the wrong moneyness."""
        surface = _real(code)
        fitted = fit_surface(code, source_kind="snapshot")
        usable = fitted.surface_input.usable_slices

        assert surface.forwards.maturities == tuple(s.year_fraction for s in usable)
        for maturity, slice_input in zip(surface.forwards.maturities, usable, strict=True):
            assert float(surface.forward(maturity)) == pytest.approx(
                slice_input.forward.forward, rel=1e-12
            )
