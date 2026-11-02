"""Tests for the QE scheme: Heston paths that reproduce Heston prices.

The decisive test is the round trip -- QE-simulated European prices against the
COS prices of the same parameters. It is this model's equivalent of Phase 2b's
Dupire round trip: two engines, one analytic and one simulated, sharing nothing
but the model, so agreement is real evidence and disagreement localises to one
of them.

A methodological note that cost a false conclusion here
-------------------------------------------------------
A single run priced three strikes and all three came out 1.7 to 1.9 standard
errors high. That looks exactly like bias and is not: the three strikes share
one set of paths, so their errors are strongly correlated and a common sign is
what *one seed's* noise looks like. Averaging six seeds per configuration put
the bias at ``-0.0018 +/- 0.0150`` -- indistinguishable from zero -- and showed
the apparent convergence trend in the single-seed study was noise too.

So the bias tests here average across seeds. The slow ones are marked.
"""

from __future__ import annotations

import numpy as np
import pytest

from otcpricer.core.market import EquityMarket
from otcpricer.engines.fourier import price_heston_european
from otcpricer.engines.mc import (
    QE_PSI_CRITICAL,
    MonteCarloConfig,
    heston_variance_paths,
    simulate_log_paths,
)
from otcpricer.models.heston import HestonParams

TEXTBOOK = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
CALIBRATED_510050 = HestonParams(v0=0.0272, kappa=4.857, theta=0.0496, sigma=1.960, rho=-0.208)
CALIBRATED_510300 = HestonParams(v0=0.0459, kappa=1.528, theta=0.0913, sigma=1.343, rho=-0.283)

# Feller margin -0.348, and a variance that sits at zero 70% of the time -- the
# regime Euler cannot represent at all.
FELLER_BROKEN = HestonParams(v0=0.01, kappa=0.3, theta=0.02, sigma=0.6, rho=-0.5)

MARKET = EquityMarket(S=100.0, r=0.03, sigma=0.2, q=0.0)


def _european(params: HestonParams, strike: float, maturity: float, config: MonteCarloConfig):  # type: ignore[no-untyped-def]
    """Antithetic pair-mean price and its standard error."""
    _, paths = simulate_log_paths(MARKET, maturity, config, heston=params)
    payoff = np.maximum(paths[-1, :] - strike, 0.0) * np.exp(-MARKET.r * maturity)
    half = payoff.size // 2
    pairs = 0.5 * (payoff[:half] + payoff[half : 2 * half])
    return float(pairs.mean()), float(pairs.std(ddof=1) / np.sqrt(pairs.size))


class TestPathProperties:
    @pytest.mark.parametrize(
        ("name", "params"),
        [
            ("textbook", TEXTBOOK),
            ("510050", CALIBRATED_510050),
            ("510300", CALIBRATED_510300),
            ("feller broken", FELLER_BROKEN),
        ],
    )
    def test_paths_stay_positive(self, name: str, params: HestonParams) -> None:
        """Which an Euler scheme on the variance cannot promise.

        When Feller is violated the variance reaches zero, an Euler step takes
        it negative, and ``sqrt(v)`` is undefined. QE samples from a
        distribution supported on the non-negatives instead, so there is
        nothing to truncate.
        """
        config = MonteCarloConfig(paths=20_000, steps_per_year=126, seed=5, antithetic=True)
        _, paths = simulate_log_paths(MARKET, 1.0, config, heston=params)

        assert np.all(np.isfinite(paths))
        assert np.all(paths > 0.0)
        assert paths[0, 0] == pytest.approx(MARKET.S)

    def test_the_shape_matches_the_schedule(self) -> None:
        config = MonteCarloConfig(paths=1_000, steps_per_year=100, seed=1, antithetic=False)
        times, paths = simulate_log_paths(MARKET, 2.0, config, heston=TEXTBOOK)

        assert times.shape == (201,)
        assert paths.shape == (201, 1_000)
        assert times[0] == 0.0
        assert times[-1] == pytest.approx(2.0)

    def test_the_same_seed_gives_the_same_paths(self) -> None:
        """Needed for common random numbers in finite-difference Greeks."""
        config = MonteCarloConfig(paths=5_000, steps_per_year=63, seed=9, antithetic=True)
        _, first = simulate_log_paths(MARKET, 1.0, config, heston=TEXTBOOK)
        _, second = simulate_log_paths(MARKET, 1.0, config, heston=TEXTBOOK)
        assert np.array_equal(first, second)

    def test_a_different_seed_gives_different_paths(self) -> None:
        a = MonteCarloConfig(paths=5_000, steps_per_year=63, seed=9, antithetic=True)
        b = MonteCarloConfig(paths=5_000, steps_per_year=63, seed=10, antithetic=True)
        _, first = simulate_log_paths(MARKET, 1.0, a, heston=TEXTBOOK)
        _, second = simulate_log_paths(MARKET, 1.0, b, heston=TEXTBOOK)
        assert not np.array_equal(first, second)

    def test_heston_and_local_vol_are_refused_together(self) -> None:
        """They are different models, not different settings.

        Silently preferring one would turn a caller's mistake into a plausible
        number from the wrong model.
        """
        from otcpricer.models.local_vol import local_vol_from_fitted
        from otcpricer.models.surface import fit_surface

        try:
            surface = local_vol_from_fitted(fit_surface("510300"), spot=100.0, tail_drift=0.02)
        except Exception:  # pragma: no cover - needs the committed snapshots
            pytest.skip("no snapshots committed")

        config = MonteCarloConfig(paths=1_000, steps_per_year=63, seed=1, antithetic=False)
        with pytest.raises(ValueError, match="different models"):
            simulate_log_paths(MARKET, 1.0, config, heston=TEXTBOOK, local_vol=surface)

    def test_the_switching_threshold_is_andersens(self) -> None:
        assert QE_PSI_CRITICAL == 1.5


class TestTheRoundTrip:
    """QE prices against COS prices: the correctness evidence for this path."""

    @pytest.mark.parametrize(
        ("name", "params"),
        [
            ("textbook", TEXTBOOK),
            ("510050", CALIBRATED_510050),
            ("510300", CALIBRATED_510300),
        ],
    )
    @pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
    def test_reproduces_the_fourier_price(
        self, name: str, params: HestonParams, strike: float
    ) -> None:
        """Within three standard errors, which is the honest tolerance.

        Not two: with three strikes per parameter set and three parameter sets
        this is nine correlated comparisons, so a two-sigma band would fail on
        noise alone often enough to be useless as a regression test.
        """
        maturity = 1.0
        reference = price_heston_european(
            params, spot=MARKET.S, strike=strike, maturity=maturity, rate=MARKET.r, flag="c"
        ).price
        config = MonteCarloConfig(paths=200_000, steps_per_year=252, seed=11, antithetic=True)
        simulated, standard_error = _european(params, strike, maturity, config)

        assert abs(simulated - reference) < 3.0 * standard_error, (
            f"{name} K={strike}: QE {simulated:.6f} vs COS {reference:.6f}, "
            f"{abs(simulated - reference) / standard_error:.2f} standard errors"
        )

    @pytest.mark.parametrize(
        ("name", "params"),
        [("textbook", TEXTBOOK), ("510050", CALIBRATED_510050)],
    )
    def test_the_underlying_is_a_martingale_under_simulation(
        self, name: str, params: HestonParams
    ) -> None:
        """``E[S_T] = S_0 e^{rT}``, to within sampling error.

        The sampling error matters: at 200,000 paths and this volatility it is
        roughly 3 basis points on the ratio, so a tolerance tighter than that
        would be testing the random number generator.
        """
        config = MonteCarloConfig(paths=200_000, steps_per_year=252, seed=11, antithetic=True)
        _, paths = simulate_log_paths(MARKET, 1.0, config, heston=params)
        ratio = float(paths[-1, :].mean() / MARKET.S)
        assert ratio == pytest.approx(float(np.exp(MARKET.r * 1.0)), rel=1.5e-3)


@pytest.mark.slow
class TestBiasAcrossSeeds:
    """Where the bias is actually measured, because one seed cannot.

    Six seeds per configuration, following the same design Phase 2b used for
    the local-volatility Monte Carlo bias.
    """

    SEEDS = (11, 23, 37, 51, 67, 83)

    def _bias(self, params: HestonParams, steps_per_year: int) -> tuple[float, float]:
        reference = price_heston_european(
            params, spot=MARKET.S, strike=100.0, maturity=1.0, rate=MARKET.r, flag="c"
        ).price
        errors = []
        for seed in self.SEEDS:
            config = MonteCarloConfig(
                paths=200_000, steps_per_year=steps_per_year, seed=seed, antithetic=True
            )
            simulated, _ = _european(params, 100.0, 1.0, config)
            errors.append(simulated - reference)
        values = np.array(errors)
        return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))

    @pytest.mark.parametrize("steps_per_year", [63, 252])
    def test_no_measurable_bias(self, steps_per_year: int) -> None:
        """Measured on 510300, the set whose single-seed run looked biased.

        Averaged over seeds the bias is ``-0.014 +/- 0.014`` at 63 steps a year
        and ``-0.002 +/- 0.015`` at 252 -- both inside one standard error of
        zero, on a price of 9.78. The step count is therefore an accuracy
        parameter whose effect is below measurement here, not a free lunch: it
        is bounded, not shown to be absent.
        """
        bias, standard_error = self._bias(CALIBRATED_510300, steps_per_year)
        assert abs(bias) < 2.0 * standard_error, (
            f"bias {bias:+.6f} is {abs(bias) / standard_error:.2f} standard errors "
            f"from zero at {steps_per_year} steps/year"
        )
        # And bounded in absolute terms: 2 standard errors is 0.3% of price.
        assert abs(bias) + 2.0 * standard_error < 0.05

    def test_the_apparent_single_seed_bias_was_noise(self) -> None:
        """Pinning the methodological trap, not just the number.

        One run, three strikes, all three high by 1.7-1.9 standard errors. The
        strikes share paths, so their errors are correlated and a common sign
        is ordinary noise rather than evidence. This test asserts that the
        spread *across seeds* is large enough to produce such a run, which is
        what makes the single-seed reading uninformative.
        """
        reference = price_heston_european(
            CALIBRATED_510300, spot=MARKET.S, strike=100.0, maturity=1.0, rate=MARKET.r
        ).price
        errors = []
        for seed in self.SEEDS:
            config = MonteCarloConfig(paths=200_000, steps_per_year=252, seed=seed, antithetic=True)
            simulated, standard_error = _european(CALIBRATED_510300, 100.0, 1.0, config)
            errors.append((simulated - reference) / standard_error)

        spread = np.array(errors)
        assert spread.max() - spread.min() > 1.5, (
            "seed-to-seed spread should comfortably span the apparent effect"
        )
        assert abs(spread.mean()) < 1.5


@pytest.mark.slow
class TestFellerBrokenRegime:
    @pytest.mark.parametrize(
        ("name", "params", "zero_share", "paths_touching"),
        [
            # Feller satisfied: the variance is strictly positive, always.
            ("textbook", TEXTBOOK, 0.0, 0.0),
            # And the fits to this project's own data, which are not.
            ("510050", CALIBRATED_510050, 0.458, 1.00),
            ("510300", CALIBRATED_510300, 0.316, 0.99),
            ("broken", FELLER_BROKEN, 0.699, 0.98),
        ],
    )
    def test_the_variance_really_does_reach_zero(
        self, name: str, params: HestonParams, zero_share: float, paths_touching: float
    ) -> None:
        """The measurement that makes QE a requirement rather than a refinement.

        At the parameters calibrated to this project's own data, **45.8% of all
        variance cells are exactly zero and every single path touches zero** on
        510050; 31.6% and 99% on 510300. That is not an edge case to be guarded
        against -- it is the typical state of the process. A truncated Euler
        scheme would be clamping a negative value on roughly half its cells,
        and that bias does not vanish as the step shrinks, because the event
        being clamped is not a discretisation artefact.

        With Feller satisfied the share is exactly zero, which is the control:
        it confirms the atom is the model's behaviour and not the scheme's.

        Read off the variance directly, because it is **not** recoverable from
        the spot paths -- the log-spot step mixes ``v_t`` with ``v_{t+dt}`` and
        adds a deterministic drift, so a zero variance does not freeze the
        return. An earlier version of this test looked for frozen returns and
        found none, which said nothing about the variance.
        """
        config = MonteCarloConfig(paths=50_000, steps_per_year=252, seed=3, antithetic=True)
        _, variance = heston_variance_paths(params, 2.0, config)

        assert np.all(np.isfinite(variance))
        assert np.all(variance >= 0.0)

        at_zero = float((variance == 0.0).mean())
        touching = float((variance == 0.0).any(axis=0).mean())

        assert at_zero == pytest.approx(zero_share, abs=0.02)
        assert touching == pytest.approx(paths_touching, abs=0.02)

    def test_the_long_run_variance_is_still_respected(self) -> None:
        """Despite sitting at zero half the time, which is the reassuring part.

        A scheme that reached zero too eagerly would pull the mean variance
        below theta. It does not: 0.047 against a theta of 0.0496.
        """
        config = MonteCarloConfig(paths=50_000, steps_per_year=252, seed=3, antithetic=True)
        _, variance = heston_variance_paths(CALIBRATED_510050, 2.0, config)
        assert float(variance.mean()) == pytest.approx(CALIBRATED_510050.theta, rel=0.1)

    def test_still_reproduces_the_fourier_price(self) -> None:
        """The regime where the scheme has to earn its place."""
        reference = price_heston_european(
            FELLER_BROKEN, spot=MARKET.S, strike=100.0, maturity=1.0, rate=MARKET.r
        ).price
        config = MonteCarloConfig(paths=200_000, steps_per_year=252, seed=7, antithetic=True)
        simulated, standard_error = _european(FELLER_BROKEN, 100.0, 1.0, config)

        assert abs(simulated - reference) < 3.0 * standard_error
