"""Tests for the SVI/SSVI surface, its calibration, and the arbitrage checker.

Three things are being established here, in order of how much they matter.

**The analytic derivatives are right.** Phase 2b's Dupire formula is built
entirely out of ``dw/dk``, ``d2w/dk2`` and ``dw/dT``. One wrong derivative and
every local volatility is wrong, with nothing downstream to catch it, so each is
checked against a high-order finite difference across the whole surface.

**The calibration recovers what it should.** A surface is generated from known
SSVI parameters, sampled, and refitted; the fit has to come back to the same
place. Without that, a plausible-looking fit to real data proves nothing.

**The checker can fail.** A checker that cannot fail is worthless. Surfaces with
deliberate butterfly and calendar arbitrage are constructed here, and the
checker is required to find them -- including the specific degenerate case that
once passed it.
"""

from __future__ import annotations

import numpy as np
import pytest

from otcpricer.core.exceptions import CalibrationError, ParameterError
from otcpricer.marketdata.snapshots import load_snapshot
from otcpricer.models import bsm
from otcpricer.models.arbitrage import (
    DEFAULT_DEGENERACY_FLOOR,
    DEFAULT_MAX_MATURITY,
    check_surface,
    local_variance_is_well_defined,
)
from otcpricer.models.surface import FittedSurface, fit_surface, fit_surface_from_chain
from otcpricer.models.svi import (
    SSVIParams,
    SSVISurface,
    SVISliceParams,
    butterfly_g,
    power_law_phi,
    power_law_phi_derivative,
    raw_svi_total_variance,
)
from otcpricer.models.svi_calibration import calibrate_ssvi
from otcpricer.reports.surface_report import surface_report
from tests.test_marketdata import SNAPSHOT_DIR

# ---------------------------------------------------------------------------
# Reference surfaces
# ---------------------------------------------------------------------------

#: A healthy surface with an equity-index shape: downward skew, curvature
#: decaying with maturity, variance rising throughout.
HEALTHY = SSVIParams(
    rho=-0.35,
    eta=1.2,
    gamma=0.4,
    maturities=(0.08, 0.25, 0.5, 1.0, 2.0),
    thetas=(0.0032, 0.0100, 0.0200, 0.0400, 0.0800),
)

#: The real fits, pinned as regression anchors. Recorded from the committed
#: snapshots; a refactor that moves these has changed the model, not the code.
REAL_ANCHORS = {
    "510050": {"n_slices": 4, "n_points": 39, "rms_vp": 0.645, "max_vp": 1.350},
    "510300": {"n_slices": 3, "n_points": 42, "rms_vp": 0.654, "max_vp": 1.977},
}


@pytest.fixture
def healthy() -> SSVISurface:
    return SSVISurface(HEALTHY)


def _grid(
    moneyness: float = 0.6, n_k: int = 25, t_min: float = 0.05, t_max: float = 3.0, n_t: int = 13
) -> tuple[np.ndarray, np.ndarray]:
    return np.linspace(-moneyness, moneyness, n_k), np.linspace(t_min, t_max, n_t)


# ---------------------------------------------------------------------------
# Raw SVI slice
# ---------------------------------------------------------------------------


class TestRawSVI:
    def test_zero_b_gives_a_flat_smile(self) -> None:
        """``b = 0`` removes every ``k``-dependent term, so variance is constant."""
        params = SVISliceParams(a=0.04, b=0.0, rho=-0.3, m=0.1, sigma=0.2)
        k = np.linspace(-0.8, 0.8, 41)
        variance = np.asarray(raw_svi_total_variance(k, params))

        assert np.allclose(variance, 0.04, atol=1e-15)

    def test_zero_rho_is_symmetric_about_m(self) -> None:
        params = SVISliceParams(a=0.02, b=0.3, rho=0.0, m=0.15, sigma=0.25)
        offset = np.linspace(0.0, 0.5, 11)
        left = np.asarray(raw_svi_total_variance(params.m - offset, params))
        right = np.asarray(raw_svi_total_variance(params.m + offset, params))

        assert np.allclose(left, right, atol=1e-15)

    def test_minimum_sits_at_m_when_rho_is_zero(self) -> None:
        params = SVISliceParams(a=0.02, b=0.3, rho=0.0, m=0.15, sigma=0.25)
        k = np.linspace(-0.5, 0.8, 2001)
        variance = np.asarray(raw_svi_total_variance(k, params))

        assert k[int(np.argmin(variance))] == pytest.approx(params.m, abs=1e-3)

    def test_vertex_value_is_a_plus_b_sigma_sqrt_one_minus_rho_squared(self) -> None:
        """At ``k = m`` the square root collapses to ``sigma``."""
        params = SVISliceParams(a=0.02, b=0.3, rho=-0.4, m=0.15, sigma=0.25)
        assert float(raw_svi_total_variance(params.m, params)) == pytest.approx(
            params.a + params.b * params.sigma, rel=1e-14
        )

    def test_wings_are_asymptotically_linear(self) -> None:
        """Slopes approach ``b(rho -/+ 1)``, SVI's defining wing behaviour."""
        params = SVISliceParams(a=0.02, b=0.3, rho=-0.4, m=0.0, sigma=0.25)
        k = np.array([-400.0, -399.0, 399.0, 400.0])
        w = np.asarray(raw_svi_total_variance(k, params))

        assert (w[1] - w[0]) == pytest.approx(params.b * (params.rho - 1.0), rel=1e-4)
        assert (w[3] - w[2]) == pytest.approx(params.b * (params.rho + 1.0), rel=1e-4)


class TestPowerLawPhi:
    def test_derivative_matches_a_finite_difference(self) -> None:
        """``phi'`` enters ``dw/dT``; an error here is an error in Dupire."""
        theta = np.array([0.001, 0.01, 0.05, 0.2, 0.8])
        for eta in (0.5, 1.2, 4.0):
            for gamma in (0.05, 0.25, 0.5):
                analytic = power_law_phi_derivative(theta, eta, gamma)
                h = 1e-7 * np.maximum(theta, 1e-3)
                numeric = (
                    power_law_phi(theta + h, eta, gamma) - power_law_phi(theta - h, eta, gamma)
                ) / (2.0 * h)
                assert np.allclose(analytic, numeric, rtol=1e-6)

    def test_phi_decays_with_theta(self) -> None:
        """Curvature falls as variance accumulates -- the smile flattens out."""
        theta = np.linspace(0.001, 1.0, 200)
        phi = power_law_phi(theta, 1.5, 0.4)
        assert np.all(np.diff(phi) < 0.0)


# ---------------------------------------------------------------------------
# SSVI: parameters and term structure
# ---------------------------------------------------------------------------


class TestSSVIParams:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("rho", -1.0),
            ("rho", 1.0),
            ("eta", 0.0),
            ("eta", -1.0),
            ("gamma", 0.0),
            ("gamma", 0.75),
        ],
    )
    def test_rejects_parameters_outside_the_analysed_range(self, field: str, value: float) -> None:
        kwargs: dict[str, object] = {
            "rho": -0.3,
            "eta": 1.0,
            "gamma": 0.4,
            "maturities": (0.25, 1.0),
            "thetas": (0.01, 0.04),
        }
        kwargs[field] = value
        with pytest.raises(ParameterError):
            SSVIParams(**kwargs)  # type: ignore[arg-type]

    def test_rejects_mismatched_maturities_and_thetas(self) -> None:
        with pytest.raises(ParameterError, match="2 maturities but 3 thetas"):
            SSVIParams(
                rho=-0.3, eta=1.0, gamma=0.4, maturities=(0.25, 1.0), thetas=(0.01, 0.02, 0.04)
            )

    def test_rejects_unordered_maturities(self) -> None:
        with pytest.raises(ParameterError, match="increasing"):
            SSVIParams(rho=-0.3, eta=1.0, gamma=0.4, maturities=(1.0, 0.25), thetas=(0.04, 0.01))

    def test_butterfly_margin_is_positive_for_a_healthy_surface(self) -> None:
        assert HEALTHY.butterfly_margin > 0.0

    def test_butterfly_margin_goes_negative_when_eta_is_cranked(self) -> None:
        params = SSVIParams(
            rho=-0.7, eta=8.0, gamma=0.5, maturities=(0.25, 1.0), thetas=(0.05, 0.10)
        )
        assert params.butterfly_margin < 0.0

    def test_calendar_margin_is_the_smallest_theta_increment(self) -> None:
        increments = np.diff(np.array(HEALTHY.thetas))
        assert HEALTHY.calendar_margin == pytest.approx(float(np.min(increments)))
        assert HEALTHY.calendar_margin == pytest.approx(0.0100 - 0.0032)

    def test_calendar_margin_is_infinite_for_a_single_maturity(self) -> None:
        params = SSVIParams(rho=0.0, eta=1.0, gamma=0.4, maturities=(0.5,), thetas=(0.02,))
        assert params.calendar_margin == float("inf")


class TestTermStructure:
    def test_theta_passes_through_every_fitted_point(self, healthy: SSVISurface) -> None:
        for maturity, theta in zip(HEALTHY.maturities, HEALTHY.thetas, strict=True):
            assert float(healthy.theta(maturity)) == pytest.approx(theta, rel=1e-12)

    def test_theta_vanishes_at_the_origin(self, healthy: SSVISurface) -> None:
        """No time, no variance. The interpolant is anchored there."""
        assert float(healthy.theta(1e-12)) < 1e-9

    def test_theta_is_monotone(self, healthy: SSVISurface) -> None:
        """Monotonicity *is* the calendar-arbitrage condition."""
        maturities = np.linspace(1e-6, 5.0, 20001)
        assert np.all(np.diff(np.asarray(healthy.theta(maturities))) >= -1e-18)

    def test_dtheta_dT_matches_a_finite_difference(self, healthy: SSVISurface) -> None:  # noqa: N802
        maturities = np.linspace(0.02, 4.0, 400)
        h = 1e-6
        numeric = (
            np.asarray(healthy.theta(maturities + h)) - np.asarray(healthy.theta(maturities - h))
        ) / (2.0 * h)
        assert np.allclose(np.asarray(healthy.dtheta_dT(maturities)), numeric, atol=1e-6)

    def test_no_kink_at_the_last_quoted_maturity(self, healthy: SSVISurface) -> None:
        """Where the extrapolated tail joins the interpolant.

        This is the seam that matters: ``dw/dT`` is Dupire's numerator, so a
        kink here is a jump in local volatility at the edge of the quoted
        surface. Extrapolating at the average rate ``theta/T`` instead of the
        matched slope left a 1.04e-03 discrepancy exactly here while matching
        to 1e-9 everywhere else.
        """
        last = HEALTHY.maturities[-1]
        h = 1e-6
        numeric = (float(healthy.theta(last + h)) - float(healthy.theta(last - h))) / (2.0 * h)
        assert float(healthy.dtheta_dT(last)) == pytest.approx(numeric, abs=1e-7)

    def test_tail_extends_at_the_last_quoted_secant(self, healthy: SSVISurface) -> None:
        secant = (HEALTHY.thetas[-1] - HEALTHY.thetas[-2]) / (
            HEALTHY.maturities[-1] - HEALTHY.maturities[-2]
        )
        beyond = HEALTHY.maturities[-1] + 1.5
        assert float(healthy.dtheta_dT(beyond)) == pytest.approx(secant, rel=1e-12)
        assert float(healthy.theta(beyond)) == pytest.approx(
            HEALTHY.thetas[-1] + 1.5 * secant, rel=1e-12
        )

    def test_concave_term_structure_still_accumulates_variance(self) -> None:
        """The regression test for the flat-tail defect.

        PCHIP clamps its endpoint derivative to zero when the variance term
        structure is concave enough at the end -- which is what a downward
        sloping volatility term structure looks like, and is ordinary in a
        stressed market. The tail then went flat, so total variance stopped
        growing and local variance was identically zero past the last quoted
        expiry. Nothing flagged it, because a flat surface is
        calendar-arbitrage-free.

        Measured on exactly these inputs: PCHIP gave 0.000000 where the
        realised rate is 0.066667.
        """
        params = SSVIParams(
            rho=-0.3, eta=1.0, gamma=0.4, maturities=(0.25, 1.0), thetas=(0.05, 0.10)
        )
        surface = SSVISurface(params)

        assert float(surface.dtheta_dT(2.0)) == pytest.approx(0.0666667, rel=1e-4)
        # Variance keeps accumulating out where a snowball actually lives.
        assert float(surface.theta(3.0)) > float(surface.theta(1.0))
        assert float(surface.dw_dT(0.0, 3.0)) > 0.0

    def test_concave_tail_stays_monotone(self) -> None:
        """The fix must not buy positivity at the cost of the calendar condition."""
        params = SSVIParams(
            rho=-0.3, eta=1.0, gamma=0.4, maturities=(0.1, 0.25, 1.0), thetas=(0.06, 0.08, 0.10)
        )
        surface = SSVISurface(params)
        maturities = np.linspace(1e-6, 4.0, 20001)
        assert np.all(np.diff(np.asarray(surface.theta(maturities))) >= -1e-18)


# ---------------------------------------------------------------------------
# Analytic derivatives -- what Phase 2b's Dupire formula is built from
# ---------------------------------------------------------------------------


class TestAnalyticDerivatives:
    """Every derivative against a high-order finite difference.

    Richardson-extrapolated central differences, so the numeric reference is
    good to roughly 1e-10 and a genuine 1e-6 error in an analytic form cannot
    hide inside the truncation error.
    """

    @staticmethod
    def _richardson(f, x: np.ndarray, h: float) -> np.ndarray:  # type: ignore[no-untyped-def]
        """Fourth-order central first derivative."""
        return (
            -np.asarray(f(x + 2 * h))
            + 8 * np.asarray(f(x + h))
            - 8 * np.asarray(f(x - h))
            + np.asarray(f(x - 2 * h))
        ) / (12 * h)

    def test_dw_dk(self, healthy: SSVISurface) -> None:
        k, maturities = _grid()
        for maturity in maturities:
            numeric = self._richardson(lambda x, t=maturity: healthy.total_variance(x, t), k, 1e-5)
            analytic = np.asarray(healthy.dw_dk(k, maturity))
            assert np.allclose(analytic, numeric, rtol=1e-8, atol=1e-12)

    def test_d2w_dk2(self, healthy: SSVISurface) -> None:
        k, maturities = _grid()
        h = 1e-4
        for maturity in maturities:
            numeric = (
                -np.asarray(healthy.total_variance(k + 2 * h, maturity))
                + 16 * np.asarray(healthy.total_variance(k + h, maturity))
                - 30 * np.asarray(healthy.total_variance(k, maturity))
                + 16 * np.asarray(healthy.total_variance(k - h, maturity))
                - np.asarray(healthy.total_variance(k - 2 * h, maturity))
            ) / (12 * h * h)
            analytic = np.asarray(healthy.d2w_dk2(k, maturity))
            assert np.allclose(analytic, numeric, rtol=1e-6, atol=1e-10)

    def test_dw_dT(self, healthy: SSVISurface) -> None:  # noqa: N802
        k, maturities = _grid(t_min=0.05, t_max=3.0, n_t=19)
        for kk in k:
            numeric = self._richardson(
                lambda t, k0=kk: healthy.total_variance(k0, t), maturities, 1e-5
            )
            analytic = np.asarray(healthy.dw_dT(kk, maturities))
            assert np.allclose(analytic, numeric, rtol=1e-6, atol=1e-9)

    def test_dw_dT_across_the_tail_boundary(self, healthy: SSVISurface) -> None:  # noqa: N802
        """Straddling the seam, where the two expressions must agree."""
        last = HEALTHY.maturities[-1]
        maturities = np.linspace(last - 0.2, last + 0.2, 81)
        for kk in (-0.4, 0.0, 0.3):
            numeric = self._richardson(
                lambda t, k0=kk: healthy.total_variance(k0, t), maturities, 1e-6
            )
            analytic = np.asarray(healthy.dw_dT(kk, maturities))
            assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-8)

    def test_second_derivative_is_strictly_positive(self, healthy: SSVISurface) -> None:
        """An SSVI slice is always convex for ``|rho| < 1``; the closed form says so."""
        k, maturities = _grid(moneyness=2.0, n_k=81)
        for maturity in maturities:
            assert np.all(np.asarray(healthy.d2w_dk2(k, maturity)) > 0.0)

    def test_butterfly_matches_the_standalone_formula(self, healthy: SSVISurface) -> None:
        """``surface.butterfly`` must be exactly ``butterfly_g`` of its own pieces."""
        k, maturities = _grid()
        for maturity in maturities:
            expected = butterfly_g(
                healthy.total_variance(k, maturity),
                healthy.dw_dk(k, maturity),
                healthy.d2w_dk2(k, maturity),
                k,
            )
            assert np.allclose(np.asarray(healthy.butterfly(k, maturity)), expected, rtol=1e-14)


# ---------------------------------------------------------------------------
# Degenerate and limiting cases
# ---------------------------------------------------------------------------


class TestDegenerateCases:
    def test_zero_rho_gives_a_symmetric_smile(self) -> None:
        params = SSVIParams(
            rho=0.0, eta=1.2, gamma=0.4, maturities=(0.25, 1.0), thetas=(0.01, 0.04)
        )
        surface = SSVISurface(params)
        offset = np.linspace(0.0, 0.7, 15)
        for maturity in (0.25, 0.6, 1.0):
            left = np.asarray(surface.total_variance(-offset, maturity))
            right = np.asarray(surface.total_variance(offset, maturity))
            assert np.allclose(left, right, atol=1e-15)

    def test_at_the_money_variance_is_theta_exactly(self, healthy: SSVISurface) -> None:
        """At ``k = 0`` the SSVI expression collapses to ``theta``.

        ``w(0) = (theta/2)(1 + rho*0 + sqrt(rho^2 + 1 - rho^2)) = theta``,
        independent of ``eta`` and ``gamma``. A clean algebraic anchor.
        """
        for maturity, theta in zip(HEALTHY.maturities, HEALTHY.thetas, strict=True):
            assert float(healthy.total_variance(0.0, maturity)) == pytest.approx(theta, rel=1e-14)

    def test_at_the_money_vol_is_sqrt_theta_over_T(self, healthy: SSVISurface) -> None:  # noqa: N802
        for maturity, theta in zip(HEALTHY.maturities, HEALTHY.thetas, strict=True):
            assert float(healthy.implied_vol(0.0, maturity)) == pytest.approx(
                np.sqrt(theta / maturity), rel=1e-14
            )

    def test_negative_rho_puts_the_skew_to_the_downside(self, healthy: SSVISurface) -> None:
        """Equity index convention: puts bid over calls."""
        for maturity in (0.25, 1.0, 2.0):
            low = float(healthy.implied_vol(-0.2, maturity))
            high = float(healthy.implied_vol(0.2, maturity))
            assert low > high

    def test_implied_vol_at_strike_agrees_with_log_moneyness(self, healthy: SSVISurface) -> None:
        forward = 2.85
        for strike in (2.4, 2.85, 3.3):
            assert float(healthy.implied_vol_at_strike(strike, forward, 0.5)) == pytest.approx(
                float(healthy.implied_vol(np.log(strike / forward), 0.5)), rel=1e-14
            )

    def test_vectorises_over_both_arguments(self, healthy: SSVISurface) -> None:
        k = np.linspace(-0.3, 0.3, 7)
        maturities = np.linspace(0.1, 2.0, 7)
        paired = np.asarray(healthy.total_variance(k, maturities))
        assert paired.shape == (7,)
        for index in range(7):
            assert paired[index] == pytest.approx(
                float(healthy.total_variance(k[index], maturities[index])), rel=1e-14
            )


# ---------------------------------------------------------------------------
# The arbitrage checker -- it has to be able to fail
# ---------------------------------------------------------------------------


class TestArbitrageChecker:
    def test_passes_a_healthy_surface(self, healthy: SSVISurface) -> None:
        report = check_surface(healthy)
        assert report.arbitrage_free
        assert report.butterfly_free
        assert report.calendar_free
        assert not report.calendar_degenerate
        assert not report.violations
        assert report.min_butterfly > 0.0
        assert report.n_checked > 1000

    @pytest.mark.parametrize("eta", [4.0, 6.0, 8.0])
    def test_catches_butterfly_arbitrage(self, eta: float) -> None:
        """Cranking curvature past the Gatheral-Jacquier bound.

        The sufficient condition ``theta*phi*(1+|rho|) < 4`` is violated by
        construction here, and ``g(k)`` must actually go negative -- not merely
        the parameter margin, which is a different claim.
        """
        params = SSVIParams(
            rho=-0.7, eta=eta, gamma=0.5, maturities=(0.25, 1.0), thetas=(0.05, 0.10)
        )
        assert params.butterfly_margin < 0.0

        report = check_surface(SSVISurface(params))

        assert not report.arbitrage_free
        assert not report.butterfly_free
        assert report.min_butterfly < 0.0
        assert any(v.condition == "butterfly" for v in report.violations)
        assert "ARBITRAGE" in report.summary()

    def test_catches_calendar_arbitrage(self) -> None:
        """A variance term structure that falls: total variance decreasing in T."""
        params = SSVIParams(
            rho=-0.2,
            eta=1.0,
            gamma=0.4,
            maturities=(0.25, 0.5, 1.0),
            thetas=(0.04, 0.03, 0.02),
        )
        assert params.calendar_margin < 0.0

        report = check_surface(SSVISurface(params))

        assert not report.arbitrage_free
        assert not report.calendar_free
        assert report.min_calendar < 0.0
        assert any(v.condition == "calendar" for v in report.violations)

    def test_catches_the_degenerate_flat_tail(self) -> None:
        """The case that once passed, and the reason degeneracy is reported.

        A flat variance term structure satisfies ``dw/dT >= 0`` and leaves local
        variance at exactly zero. It is arbitrage-free and unusable, so the
        report must say both.
        """
        surface = SSVISurface(
            SSVIParams(rho=-0.3, eta=1.0, gamma=0.4, maturities=(0.25, 1.0), thetas=(0.05, 0.10))
        )
        surface._tail_rate = 0.0  # reproduce PCHIP's clamped endpoint slope

        report = check_surface(surface, maturities=np.linspace(0.125, 3.0, 120))

        assert report.arbitrage_free, "a flat surface genuinely admits no arbitrage"
        assert report.calendar_free
        assert report.calendar_degenerate
        assert report.min_calendar_relative < DEFAULT_DEGENERACY_FLOOR

        usable, reason = local_variance_is_well_defined(report)
        assert not usable
        assert "degenerate" in reason
        assert "No arbitrage is present" in reason

    def test_violations_are_worst_first(self) -> None:
        """A truncated list has to be the informative one."""
        params = SSVIParams(
            rho=-0.8, eta=8.0, gamma=0.5, maturities=(0.25, 1.0), thetas=(0.08, 0.20)
        )
        report = check_surface(SSVISurface(params))
        values = [v.value for v in report.violations]

        assert values == sorted(values)
        assert report.min_butterfly == pytest.approx(values[0])

    def test_checks_beyond_the_last_quoted_maturity(self, healthy: SSVISurface) -> None:
        """The grid has to cover where the surface is used, not where it was fitted.

        The fitted constraints hold at the quoted maturities; between and past
        them the interpolated term structure decides, which is precisely where
        a defect can hide.
        """
        report = check_surface(healthy)
        assert report.maturity_range[0] < HEALTHY.maturities[0]
        assert report.maturity_range[1] > HEALTHY.maturities[-1]

    def test_report_serialises(self, healthy: SSVISurface) -> None:
        payload = check_surface(healthy).as_dict()
        assert payload["arbitrage_free"] is True
        assert payload["calendar_degenerate"] is False
        assert "min_calendar_relative" in payload
        assert payload["n_violations"] == 0

    def test_local_variance_precondition_reports_each_failure(self) -> None:
        params = SSVIParams(
            rho=-0.7, eta=8.0, gamma=0.5, maturities=(0.25, 1.0), thetas=(0.05, 0.04)
        )
        report = check_surface(SSVISurface(params))
        usable, reason = local_variance_is_well_defined(report)

        assert not usable
        assert "denominator" in reason
        assert "numerator" in reason


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _sample(
    surface: SSVISurface,
    maturities: tuple[float, ...],
    *,
    n_strikes: int = 13,
    width: float = 1.5,
) -> tuple[list[float], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Sample a surface the way a real chain would be observed.

    Strikes are placed in standard deviations of the slice's own at-the-money
    volatility, so each expiry is covered over a comparable moneyness range
    rather than the short end being sampled far out in the wing.
    """
    T: list[float] = []
    k_slices: list[np.ndarray] = []
    vol_slices: list[np.ndarray] = []
    vega_slices: list[np.ndarray] = []

    for maturity in maturities:
        atm = float(surface.implied_vol(0.0, maturity))
        reach = width * atm * np.sqrt(maturity)
        k = np.linspace(-reach, reach, n_strikes)
        vols = np.asarray(surface.implied_vol(k, maturity))
        vegas = np.array(
            [
                float(bsm.vega("c", 1.0, float(np.exp(kk)), maturity, 0.0, float(v), 0.0))
                for kk, v in zip(k, vols, strict=True)
            ]
        )
        T.append(maturity)
        k_slices.append(k)
        vol_slices.append(vols)
        vega_slices.append(vegas)

    return T, k_slices, vol_slices, vega_slices


class TestCalibrationSelfRecovery:
    def test_recovers_the_generating_surface(self, healthy: SSVISurface) -> None:
        """The end-to-end check on the surface implementation.

        Generate volatilities from known parameters, refit, and require the fit
        back at the same place. Without this, a tidy-looking fit to real data
        establishes nothing -- the objective could be minimised by the wrong
        surface.
        """
        T, k, vols, vegas = _sample(healthy, HEALTHY.maturities)
        result = calibrate_ssvi(T, k, vols, vegas)

        assert result.converged
        # Noise-free data: the fit should be essentially exact.
        assert result.rms_vol_error < 1e-4
        assert result.max_vol_error < 1e-3
        assert result.params.rho == pytest.approx(HEALTHY.rho, abs=0.02)
        assert result.params.eta == pytest.approx(HEALTHY.eta, rel=0.1)
        assert result.params.gamma == pytest.approx(HEALTHY.gamma, abs=0.05)
        for fitted, expected in zip(result.params.thetas, HEALTHY.thetas, strict=True):
            assert fitted == pytest.approx(expected, rel=2e-3)

    def test_recovered_surface_reprices_the_samples(self, healthy: SSVISurface) -> None:
        """Parameter recovery and surface recovery are different claims."""
        T, k, vols, _ = _sample(healthy, HEALTHY.maturities)
        result = calibrate_ssvi(T, k, vols)

        for maturity, k_slice, vol_slice in zip(T, k, vols, strict=True):
            refitted = np.asarray(result.surface.implied_vol(k_slice, maturity))
            assert np.allclose(refitted, vol_slice, atol=5e-4)

    def test_recovery_survives_quote_noise(self, healthy: SSVISurface) -> None:
        """Half a vol point of noise must not move the surface more than that."""
        T, k, vols, vegas = _sample(healthy, HEALTHY.maturities)
        rng = np.random.default_rng(20261027)
        noisy = [v + rng.normal(0.0, 0.005, size=v.shape) for v in vols]

        result = calibrate_ssvi(T, k, noisy, vegas)

        assert result.converged
        assert result.rms_vol_error < 0.01
        for maturity, k_slice, clean in zip(T, k, vols, strict=True):
            refitted = np.asarray(result.surface.implied_vol(k_slice, maturity))
            assert np.max(np.abs(refitted - clean)) < 0.01

    def test_fit_is_arbitrage_free_by_construction(self, healthy: SSVISurface) -> None:
        T, k, vols, vegas = _sample(healthy, HEALTHY.maturities)
        result = calibrate_ssvi(T, k, vols, vegas)

        assert result.butterfly_margin > 0.0
        assert result.calendar_margin >= 0.0
        report = check_surface(result.surface)
        assert report.arbitrage_free
        assert not report.calendar_degenerate

    def test_calendar_condition_holds_structurally(self, healthy: SSVISurface) -> None:
        """Theta is parameterised as a level plus non-negative increments.

        So monotonicity cannot be violated even transiently, whatever the
        optimiser does. Fitting deliberately inconsistent data must therefore
        still produce a monotone term structure.
        """
        T, k, vols, vegas = _sample(healthy, HEALTHY.maturities)
        # Inconsistent: drag the long end's volatility below the short end's.
        broken = [v * (1.0 if index < 2 else 0.4) for index, v in enumerate(vols)]

        result = calibrate_ssvi(T, k, broken, vegas)

        assert result.calendar_margin >= 0.0
        assert list(result.params.thetas) == sorted(result.params.thetas)

    def test_per_slice_quality_is_reported(self, healthy: SSVISurface) -> None:
        T, k, vols, vegas = _sample(healthy, HEALTHY.maturities)
        result = calibrate_ssvi(T, k, vols, vegas)

        assert len(result.slices) == len(HEALTHY.maturities)
        for quality, maturity in zip(result.slices, HEALTHY.maturities, strict=True):
            assert quality.maturity == pytest.approx(maturity)
            assert quality.n_points == 13
            assert quality.rms_vol_error <= quality.max_vol_error
            assert quality.atm_vol_fitted == pytest.approx(quality.atm_vol_observed, abs=1e-3)
        assert result.n_points == sum(q.n_points for q in result.slices)


class TestCalibrationInputHandling:
    def test_rejects_no_maturities(self) -> None:
        with pytest.raises(CalibrationError, match="no maturities"):
            calibrate_ssvi([], [], [])

    def test_rejects_unordered_maturities(self) -> None:
        k = [np.linspace(-0.2, 0.2, 5)] * 2
        v = [np.full(5, 0.2)] * 2
        with pytest.raises(CalibrationError, match="strictly increasing"):
            calibrate_ssvi([1.0, 0.5], k, v)

    def test_rejects_mismatched_slice_counts(self) -> None:
        with pytest.raises(CalibrationError, match="2 maturities but 1"):
            calibrate_ssvi([0.5, 1.0], [np.zeros(5)], [np.zeros(5), np.zeros(5)])

    def test_rejects_a_slice_whose_arrays_disagree(self) -> None:
        with pytest.raises(CalibrationError, match="5 moneyness values but 4"):
            calibrate_ssvi([0.5], [np.zeros(5)], [np.zeros(4)])

    def test_rejects_an_empty_slice(self) -> None:
        with pytest.raises(CalibrationError, match="no observations"):
            calibrate_ssvi([0.5], [np.zeros(0)], [np.zeros(0)])

    def test_refuses_an_underdetermined_fit(self) -> None:
        """Three global parameters plus one theta per expiry need that many points.

        Four expiries need seven observations, so one quote each is six short.
        """
        maturities = [0.1, 0.25, 0.5, 1.0]
        k = [np.array([0.0])] * 4
        v = [np.array([0.2])] * 4
        with pytest.raises(CalibrationError, match="4 observations cannot identify 7"):
            calibrate_ssvi(maturities, k, v)

    def test_accepts_a_fit_that_is_exactly_determined(self) -> None:
        """The boundary itself must not be refused: seven points, seven parameters."""
        maturities = [0.1, 0.25, 0.5, 1.0]
        k = [
            np.array([-0.05, 0.0, 0.05]),
            np.array([-0.05, 0.0]),
            np.array([0.0]),
            np.array([0.0]),
        ]
        v = [
            np.array([0.22, 0.20, 0.19]),
            np.array([0.21, 0.20]),
            np.array([0.20]),
            np.array([0.20]),
        ]
        result = calibrate_ssvi(maturities, k, v)

        assert result.n_points == 7
        assert result.butterfly_margin > 0.0

    def test_rejects_mismatched_weights(self) -> None:
        with pytest.raises(CalibrationError, match="weights do not match"):
            calibrate_ssvi([0.5], [np.zeros(5)], [np.zeros(5)], [np.zeros(4)])

    def test_single_maturity_fits(self) -> None:
        """One expiry is a legitimate degenerate case: no calendar condition."""
        surface = SSVISurface(
            SSVIParams(rho=-0.3, eta=1.2, gamma=0.4, maturities=(0.5,), thetas=(0.02,))
        )
        T, k, vols, vegas = _sample(surface, (0.5,))
        result = calibrate_ssvi(T, k, vols, vegas)

        assert result.rms_vol_error < 1e-4
        assert result.calendar_margin == float("inf")


# ---------------------------------------------------------------------------
# End to end, on the committed real snapshots
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", ["510050", "510300"])
class TestRealSurfaces:
    def test_fits_within_the_recorded_error(self, code: str) -> None:
        """Regression anchors from the captured chains.

        Pinned loosely enough to survive a different SLSQP build, tightly
        enough that a model change cannot pass unnoticed.
        """
        if not SNAPSHOT_DIR.exists():
            pytest.skip("no snapshots committed")
        anchor = REAL_ANCHORS[code]
        fitted = fit_surface(code, source_kind="snapshot", snapshot=None)

        assert fitted.calibration.converged
        assert fitted.calibration.n_points == anchor["n_points"]
        assert len(fitted.calibration.slices) == anchor["n_slices"]
        assert fitted.calibration.rms_vol_error * 100 == pytest.approx(anchor["rms_vp"], abs=0.05)
        assert fitted.calibration.max_vol_error * 100 == pytest.approx(anchor["max_vp"], abs=0.15)
        # Under one vol point, the range an ETF surface should fit within.
        assert fitted.calibration.rms_vol_error < 0.01

    def test_is_arbitrage_free_and_usable_for_local_volatility(self, code: str) -> None:
        if not SNAPSHOT_DIR.exists():
            pytest.skip("no snapshots committed")
        fitted = fit_surface(code, source_kind="snapshot")

        assert fitted.arbitrage.arbitrage_free
        assert not fitted.arbitrage.calendar_degenerate
        assert fitted.arbitrage.min_butterfly > 0.1
        usable, reason = fitted.usable_for_local_volatility
        assert usable, reason

    def test_reproduces_the_quoted_at_the_money_level(self, code: str) -> None:
        """The level is the one thing a surface must not get wrong."""
        if not SNAPSHOT_DIR.exists():
            pytest.skip("no snapshots committed")
        fitted = fit_surface(code, source_kind="snapshot")

        for quality in fitted.calibration.slices:
            assert quality.atm_vol_fitted == pytest.approx(quality.atm_vol_observed, abs=0.02)

    def test_query_by_strike_uses_that_expirys_forward(self, code: str) -> None:
        if not SNAPSHOT_DIR.exists():
            pytest.skip("no snapshots committed")
        fitted = fit_surface(code, source_kind="snapshot")
        tag = fitted.calibration and fitted.surface_input.usable_slices[0].expiry_tag
        forward = fitted.forwards[tag]

        at_forward = fitted.implied_vol(forward, tag)
        maturity = fitted.surface_input.usable_slices[0].year_fraction
        assert at_forward == pytest.approx(float(fitted.surface.implied_vol(0.0, maturity)))

    def test_unknown_expiry_is_refused_with_the_available_ones(self, code: str) -> None:
        if not SNAPSHOT_DIR.exists():
            pytest.skip("no snapshots committed")
        fitted = fit_surface(code, source_kind="snapshot")
        with pytest.raises(CalibrationError, match="available:"):
            fitted.implied_vol(3.0, "9912")

    def test_serialises_with_its_evidence(self, code: str) -> None:
        """A surface without its provenance and fit quality is not a deliverable."""
        if not SNAPSHOT_DIR.exists():
            pytest.skip("no snapshots committed")
        payload = fit_surface(code, source_kind="snapshot").as_dict()

        assert payload["underlying"] == code
        assert payload["as_of"] == "2026-10-27"
        assert str(payload["source"]).startswith("snapshot:")
        assert payload["local_volatility_ready"] is True
        for key in ("input", "calibration", "arbitrage", "forwards"):
            assert payload[key]
        assert payload["calibration"]["slices"]  # type: ignore[index]
        assert payload["input"]["rejections"]  # type: ignore[index]


class TestSurfaceOrchestration:
    def test_too_few_usable_expiries_fails_with_the_reasons(self) -> None:
        """A refusal has to carry the diagnosis, not just the verdict."""
        if not SNAPSHOT_DIR.exists():
            pytest.skip("no snapshots committed")
        chain = load_snapshot(SNAPSHOT_DIR, "510050")
        # Keep only the one-day expiry, which screening rejects on its own.
        trimmed = type(chain)(
            underlying=chain.underlying,
            as_of=chain.as_of,
            slices=(chain.slices[0],),
            source=chain.source,
        )

        with pytest.raises(CalibrationError) as excinfo:
            fit_surface_from_chain(trimmed)

        message = str(excinfo.value)
        assert "survived screening" in message
        assert "variance term structure" in message
        assert chain.slices[0].expiry_tag in message

    def test_fit_from_chain_matches_fit_from_source(self) -> None:
        if not SNAPSHOT_DIR.exists():
            pytest.skip("no snapshots committed")
        chain = load_snapshot(SNAPSHOT_DIR, "510050")
        direct = fit_surface_from_chain(chain)
        viaSource = fit_surface("510050", source_kind="snapshot")

        assert direct.calibration.rms_vol_error == pytest.approx(
            viaSource.calibration.rms_vol_error, rel=1e-9
        )
        assert direct.calibration.params.rho == pytest.approx(
            viaSource.calibration.params.rho, rel=1e-9
        )


# ---------------------------------------------------------------------------
# The validation report
# ---------------------------------------------------------------------------


class TestSurfaceReport:
    @pytest.fixture
    def report_text(self) -> str:
        if not SNAPSHOT_DIR.exists():
            pytest.skip("no snapshots committed")
        return surface_report(fit_surface("510300", source_kind="snapshot"))

    def test_states_its_provenance(self, report_text: str) -> None:
        """A number whose source is unstated is not reviewable."""
        assert "510300" in report_text
        assert "2026-10-27" in report_text
        assert "snapshot:" in report_text
        assert "2.000%" in report_text

    def test_carries_every_required_section(self, report_text: str) -> None:
        for heading in (
            "FITTED PARAMETERS",
            "CALIBRATION ERROR, IN VOLATILITY POINTS",
            "FORWARDS AND PUT-CALL PARITY RESIDUALS",
            "DISCARDED QUOTES",
            "STATIC ARBITRAGE CHECK",
            "LOCAL VOLATILITY PRECONDITION",
        ):
            assert heading in report_text, heading

    def test_reports_a_row_per_fitted_expiry(self, report_text: str) -> None:
        fitted = fit_surface("510300", source_kind="snapshot")
        for slice_input in fitted.surface_input.usable_slices:
            assert slice_input.expiry_tag in report_text

    def test_names_the_excluded_expiry_and_why(self, report_text: str) -> None:
        """An expiry dropped without explanation looks like a bug."""
        assert "EXPIRIES EXCLUDED FROM THE FIT" in report_text
        assert "survived screening" in report_text

    def test_rejection_counts_reconcile_in_the_text(self, report_text: str) -> None:
        fitted = fit_surface("510300", source_kind="snapshot")
        assert f"{fitted.surface_input.n_rejected:5d}  total discarded" in report_text
        assert f"{fitted.surface_input.n_points:5d}  retained and fitted" in report_text

    def test_states_the_arbitrage_verdict(self, report_text: str) -> None:
        assert "PASS" in report_text
        assert "verdict: USABLE" in report_text
        assert "arbitrage-free" in report_text

    def test_a_failing_surface_reads_as_failing(self) -> None:
        """The report has to be able to say no.

        Built by hand from an arbitraged surface, because a report that only
        renders healthy cases is untested where it matters most.
        """
        if not SNAPSHOT_DIR.exists():
            pytest.skip("no snapshots committed")
        healthy_fit = fit_surface("510300", source_kind="snapshot")
        broken = SSVISurface(
            SSVIParams(rho=-0.7, eta=8.0, gamma=0.5, maturities=(0.25, 1.0), thetas=(0.05, 0.10))
        )
        broken_report = check_surface(broken)
        fitted = FittedSurface(
            underlying_code=healthy_fit.underlying_code,
            as_of=healthy_fit.as_of,
            source=healthy_fit.source,
            surface_input=healthy_fit.surface_input,
            calibration=healthy_fit.calibration,
            arbitrage=broken_report,
            forwards=healthy_fit.forwards,
        )

        text = surface_report(fitted)

        assert "FAIL" in text
        assert "verdict: NOT USABLE" in text
        assert "ARBITRAGE:" in text
        assert "worst violations" in text
        assert "butterfly" in text

    def test_lines_stay_terminal_width(self, report_text: str) -> None:
        """It has to be readable where it is read."""
        overlong = [line for line in report_text.splitlines() if len(line) > 110]
        assert not overlong, overlong


class TestArbitrageGridHorizon:
    def test_default_grid_reaches_the_products_horizon(self) -> None:
        """The grid has to cover where the surface is used.

        The listed options stop at roughly five months; the snowballs priced
        off this surface run one to three years. A grid bounded by the quoted
        expiries would leave that entire range unchecked -- and that range is
        where the flat-tail defect lived.
        """
        surface = SSVISurface(
            SSVIParams(rho=-0.2, eta=1.2, gamma=0.4, maturities=(0.08, 0.41), thetas=(0.003, 0.016))
        )
        report = check_surface(surface)

        assert report.maturity_range[1] >= DEFAULT_MAX_MATURITY

    def test_a_longer_dated_surface_is_not_truncated(self) -> None:
        """The horizon is a floor, not a cap."""
        surface = SSVISurface(
            SSVIParams(rho=-0.2, eta=1.2, gamma=0.4, maturities=(1.0, 5.0), thetas=(0.04, 0.20))
        )
        report = check_surface(surface)

        assert report.maturity_range[1] == pytest.approx(7.5)
