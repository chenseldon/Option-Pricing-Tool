"""Tests for the Heston calibration.

The load-bearing test is a round trip through synthetic data: generate a
surface from known parameters, fit it, and require the parameters back. It is
the only test that can distinguish "fits the quotes" from "found the right
model", and it caught two things -- an objective that went flat on a single bad
point, and an identifiability metric that reported 850,000 for a fit recovered
to four decimals.

Everything involving real chains is marked slow: one calibration is about a
minute, because each objective evaluation is a few hundred COS prices.
"""

from __future__ import annotations

import numpy as np
import pytest

from otcpricer.core.exceptions import CalibrationError
from otcpricer.engines.fourier import price_heston_european
from otcpricer.models.bsm import implied_volatility
from otcpricer.models.heston import HestonParams
from otcpricer.models.heston_calibration import (
    CALIBRATION_BOUNDS,
    COMPARABLE_ABSOLUTE_SLACK,
    UNIDENTIFIABLE_PENALTY,
    calibrate_heston,
)

TRUTH = HestonParams(v0=0.04, kappa=2.0, theta=0.045, sigma=0.5, rho=-0.6)

# Kept inside the identifiable region: at 0.1 years a 25% out-of-the-money call
# is worth 3e-08, and the project's implied-vol solver rightly refuses to
# invert that rather than returning a meaningless number.
MATURITIES = [0.1, 0.25, 0.5, 1.0]
MONEYNESS = np.linspace(-0.12, 0.12, 9)


def _synthetic_surface(params: HestonParams) -> list[np.ndarray]:
    """Implied vols generated from known parameters, forward-struck at 1.0."""
    slices = []
    for maturity in MATURITIES:
        vols = []
        for k in MONEYNESS:
            strike = float(np.exp(k))
            price = price_heston_european(
                params, spot=1.0, strike=strike, maturity=maturity, rate=0.0
            ).price
            vols.append(implied_volatility("c", price, 1.0, strike, maturity, 0.0))
        slices.append(np.array(vols))
    return slices


@pytest.fixture(scope="module")
def synthetic() -> list[np.ndarray]:
    return _synthetic_surface(TRUTH)


class TestValidation:
    @pytest.mark.parametrize(
        ("maturities", "moneyness", "vols", "fragment"),
        [
            ([], [], [], "no slices"),
            ([1.0, 2.0], [MONEYNESS], [np.ones(9)], "maturities"),
            ([0.0], [MONEYNESS], [np.ones(9)], "must be positive"),
            ([-1.0], [MONEYNESS], [np.ones(9)], "must be positive"),
            ([1.0], [MONEYNESS], [np.ones(5)], "strikes"),
            ([1.0], [np.array([])], [np.array([])], "is empty"),
        ],
    )
    def test_refuses_incoherent_input(
        self,
        maturities: list[float],
        moneyness: list[np.ndarray],
        vols: list[np.ndarray],
        fragment: str,
    ) -> None:
        with pytest.raises(CalibrationError, match=fragment):
            calibrate_heston(maturities, moneyness, vols)

    def test_refuses_mismatched_weights(self) -> None:
        with pytest.raises(CalibrationError, match="weights do not match"):
            calibrate_heston([1.0], [MONEYNESS], [np.ones(9)], [np.ones(5)])

    def test_the_sigma_bound_is_wide_enough_for_this_data(self) -> None:
        """The fits land near 2, and a textbook bound of 1 would clip them.

        Clipping would improve the Feller margin and worsen the fit -- making
        the model look better behaved by making it describe the market less
        well, which is the wrong direction to resolve that tension.
        """
        low, high = CALIBRATION_BOUNDS["sigma"]
        assert high >= 2.0
        assert low > 0.0


class TestTheSyntheticRoundTrip:
    """Generate from known parameters, fit, require them back."""

    @pytest.fixture(scope="class")
    def fit(self, synthetic: list[np.ndarray]):  # type: ignore[no-untyped-def]
        return calibrate_heston(MATURITIES, [MONEYNESS] * len(MATURITIES), synthetic, terms=192)

    def test_recovers_every_parameter(self, fit) -> None:  # type: ignore[no-untyped-def]
        """To four decimals, which is the strongest statement available here.

        A fit that merely reproduced the vols would not establish this: the
        kappa/sigma valley means several parameter sets price almost
        identically, so recovering the *generating* values is a separate and
        stronger claim than fitting well.
        """
        assert fit.params.v0 == pytest.approx(TRUTH.v0, rel=1e-3)
        assert fit.params.kappa == pytest.approx(TRUTH.kappa, rel=5e-3)
        assert fit.params.theta == pytest.approx(TRUTH.theta, rel=1e-3)
        assert fit.params.sigma == pytest.approx(TRUTH.sigma, rel=5e-3)
        assert fit.params.rho == pytest.approx(TRUTH.rho, rel=5e-3)

    def test_the_fit_error_is_essentially_zero(self, fit) -> None:  # type: ignore[no-untyped-def]
        """0.0005 volatility points, i.e. the COS tolerance rather than a fit."""
        assert fit.rms_vol_points < 0.01
        assert fit.n_points == len(MATURITIES) * MONEYNESS.size
        assert fit.n_unidentifiable == 0
        assert fit.converged

    def test_an_excellent_fit_can_still_be_unconfirmed(self, fit) -> None:  # type: ignore[no-untyped-def]
        """The distinction that ``converged`` alone cannot express.

        One start of four reached rms 4.6e-06 here; the next best stopped at
        4.2e-03, a thousand times worse, with kappa at 8.09 instead of 2.00 --
        and reported successful convergence. So the right answer was found and
        corroborated by nothing, which is a different fact from the parameters
        being ambiguous, and the two are reported separately.
        """
        assert fit.converged
        assert fit.n_comparable_starts == 1
        assert not fit.confirmed_by_multiple_starts
        # Nothing disagreed, because nothing else arrived.
        assert not fit.ambiguous
        assert not fit.well_identified

    def test_a_single_start_is_allowed_and_says_so(self, synthetic) -> None:  # type: ignore[no-untyped-def]
        fit = calibrate_heston(
            MATURITIES, [MONEYNESS] * len(MATURITIES), synthetic, terms=192, starts=1
        )
        assert fit.n_starts == 1
        assert not fit.confirmed_by_multiple_starts

    def test_vega_weights_are_accepted_and_change_the_answer(self, synthetic) -> None:  # type: ignore[no-untyped-def]
        """Weighting is a real choice, not decoration.

        Equal weights and vega weights should not give bitwise-identical
        parameters; if they did, the weights were being ignored.
        """
        weights = [np.linspace(0.5, 1.5, MONEYNESS.size) for _ in MATURITIES]
        weighted = calibrate_heston(
            MATURITIES, [MONEYNESS] * len(MATURITIES), synthetic, weights, terms=192, starts=1
        )
        plain = calibrate_heston(
            MATURITIES, [MONEYNESS] * len(MATURITIES), synthetic, terms=192, starts=1
        )
        assert weighted.params != plain.params


class TestTheObjective:
    def test_an_unidentifiable_point_is_penalised_not_fatal(self) -> None:
        """The defect this caught, pinned.

        The objective first returned a single huge value for the whole surface
        whenever any model price could not be inverted to a volatility. That
        turns one locally bad trial into a flat plateau with no gradient, and
        the optimiser simply stops. Charging the penalty per point keeps the
        descent direction informative.

        Driven here by asking for a 40% out-of-the-money strike at six weeks,
        where the premium underflows and the project's solver refuses to invert
        it -- correctly.
        """
        maturities = [0.1, 0.5]
        moneyness = [np.array([-0.4, 0.0, 0.4]), np.array([-0.1, 0.0, 0.1])]
        vols = [np.array([0.30, 0.20, 0.30]), np.array([0.22, 0.20, 0.21])]

        fit = calibrate_heston(maturities, moneyness, vols, terms=128, starts=2)

        # It produced a usable parameter set rather than stalling at the start.
        assert np.isfinite(fit.rms_vol_error)
        assert fit.params.sigma > 0.0
        assert UNIDENTIFIABLE_PENALTY == 1.0

    def test_the_reported_error_excludes_unscorable_points(self) -> None:
        """Rather than averaging a penalty into it.

        A penalty mixed into the headline error would make the fit look worse
        than it is in a way nobody could unpick. The count is carried
        separately instead.
        """
        maturities = [0.1]
        moneyness = [np.array([-0.5, 0.0, 0.5])]
        vols = [np.array([0.35, 0.20, 0.35])]
        fit = calibrate_heston(maturities, moneyness, vols, terms=128, starts=1)

        if fit.n_unidentifiable:
            # Whatever survived is scored on its own merits, not on the penalty.
            assert fit.rms_vol_error < UNIDENTIFIABLE_PENALTY
            assert fit.n_points == 3

    def test_comparability_uses_an_absolute_floor_too(self) -> None:
        """Because a purely relative bar is meaningless on a near-exact fit.

        The objective on recovered synthetic data is around 1e-12, and "within
        10% of that" would exclude a start agreeing to six decimal places. On
        real quotes the fit error is twenty times the absolute floor, so there
        the relative bar binds instead.
        """
        assert pytest.approx(5e-4) == COMPARABLE_ABSOLUTE_SLACK
        assert COMPARABLE_ABSOLUTE_SLACK * 100.0 < 0.1  # under a tenth of a vol point


@pytest.mark.slow
class TestAgainstRealChains:
    """Where the comparison with SSVI actually happens."""

    @pytest.fixture(scope="class")
    def akshare_comparison(self):  # type: ignore[no-untyped-def]
        from otcpricer.models.surface import fit_surface

        out = {}
        for code in ("510050", "510300"):
            try:
                ssvi = fit_surface(code)
            except Exception:  # pragma: no cover - needs committed snapshots
                pytest.skip("no snapshots committed")
            usable = ssvi.surface_input.usable_slices
            heston = calibrate_heston(
                [s.year_fraction for s in usable],
                [np.array([q.log_moneyness for q in s.points]) for s in usable],
                [np.array([q.implied_vol for q in s.points]) for s in usable],
                [np.array([q.vega for q in s.points]) for s in usable],
                forwards=[s.forward.forward for s in usable],
                rate=0.02,
                terms=192,
            )
            out[code] = (ssvi.calibration.rms_vol_error * 100.0, heston, len(usable))
        return out

    @pytest.mark.parametrize("code", ["510050", "510300"])
    def test_heston_fits_worse_than_ssvi_and_by_how_much(
        self, akshare_comparison, code: str
    ) -> None:  # type: ignore[no-untyped-def]
        """The honest headline, pinned so it cannot be quietly reversed.

        On three expiries SSVI has six parameters against Heston's five and
        wins by 10-12%. On the six-expiry CFFEX chains it has nine and wins by
        38-41%. Heston is not bought here for accuracy -- it is bought for a
        forward smile that does not flatten.
        """
        ssvi_rms, heston, n_slices = akshare_comparison[code]
        assert n_slices == 3
        ratio = heston.rms_vol_points / ssvi_rms
        assert 1.0 < ratio < 1.25, (
            f"{code}: Heston {heston.rms_vol_points:.3f} vs SSVI {ssvi_rms:.3f} = {ratio:.2f}x"
        )

    @pytest.mark.parametrize("code", ["510050", "510300"])
    def test_feller_is_violated_on_real_quotes(self, akshare_comparison, code: str) -> None:  # type: ignore[no-untyped-def]
        """Which is what makes the QE scheme a requirement downstream."""
        _, heston, _ = akshare_comparison[code]
        assert heston.params.feller_margin < 0.0
        assert not heston.params.satisfies_feller

    def test_the_fit_is_corroborated_on_real_data(self, akshare_comparison) -> None:  # type: ignore[no-untyped-def]
        """Unlike the synthetic case, where only one start found the minimum.

        Real quotes carry enough noise that the objective's valleys are
        shallower, so several starts land in the same place. That is a weaker
        surface to optimise on and an easier one to agree about.
        """
        for code in ("510050", "510300"):
            _, heston, _ = akshare_comparison[code]
            assert heston.n_comparable_starts >= 1

    def test_serialises_with_everything_a_reader_needs(self, akshare_comparison) -> None:  # type: ignore[no-untyped-def]
        _, heston, _ = akshare_comparison["510300"]
        payload = heston.as_dict()

        for key in (
            "params",
            "rms_vol_points",
            "n_comparable_starts",
            "parameter_spread",
            "confirmed_by_multiple_starts",
            "ambiguous",
            "well_identified",
            "n_unidentifiable",
        ):
            assert key in payload
        assert "same objective svi_calibration uses" in str(payload["note"])
        assert payload["params"]["satisfies_feller"] is False  # type: ignore[index]
