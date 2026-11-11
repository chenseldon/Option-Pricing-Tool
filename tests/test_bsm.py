"""Tests for the self-built Black-Scholes-Merton model.

The suite is layered from strongest evidence to weakest:

1. **Model-free identities** -- put-call parity, no-arbitrage bounds, Greek
   sign and monotonicity relations. These must hold for *any* correct
   implementation, so they catch errors without trusting a second
   implementation to be right.
2. **Independent benchmark** -- QuantLib, a different codebase entirely.
3. **Finite-difference agreement** -- each analytic Greek against a numerical
   derivative of our own price function. This catches a wrong closed form even
   where the price itself is right.
4. **Regression anchors** -- the measured v7.0.0 values, so a refactor cannot
   silently move a published number.
"""

from __future__ import annotations

import numpy as np
import pytest

from otcpricer.core.exceptions import ArbitrageError, ConvergenceError, ParameterError
from otcpricer.core.types import OptionType
from otcpricer.models import bsm

from .conftest import (
    BASELINE,
    BASELINE_EXPECTED,
    MARKET_GRID,
    MARKET_SAMPLE,
    Market,
    on_day_grid,
    quantlib_european,
    requires_quantlib,
)

# ---------------------------------------------------------------------------
# 1. Model-free identities
# ---------------------------------------------------------------------------


class TestPutCallParity:
    """``C - P = S e^{-qT} - K e^{-rT}`` must hold exactly."""

    @pytest.mark.parametrize("m", MARKET_SAMPLE, ids=repr)
    def test_parity_holds_on_sample(self, m: Market) -> None:
        call = bsm.price("c", m.S, m.K, m.T, m.r, m.sigma, m.q)
        put = bsm.price("p", m.S, m.K, m.T, m.r, m.sigma, m.q)
        expected = m.S * np.exp(-m.q * m.T) - m.K * np.exp(-m.r * m.T)
        assert call - put == pytest.approx(expected, abs=1e-9, rel=1e-12)

    def test_parity_holds_across_full_grid(self) -> None:
        """Vectorised over the whole grid, so this stays fast."""
        worst = 0.0
        for m in MARKET_GRID:
            call = bsm.price("c", m.S, m.K, m.T, m.r, m.sigma, m.q)
            put = bsm.price("p", m.S, m.K, m.T, m.r, m.sigma, m.q)
            expected = m.S * np.exp(-m.q * m.T) - m.K * np.exp(-m.r * m.T)
            # Scale by the spot: an absolute tolerance is meaningless when S
            # ranges over two orders of magnitude.
            worst = max(worst, abs((call - put) - expected) / m.S)
        assert worst < 1e-13, f"worst scaled parity violation {worst:.3e}"

    @pytest.mark.parametrize("m", MARKET_SAMPLE, ids=repr)
    def test_delta_parity(self, m: Market) -> None:
        """``delta_call - delta_put = e^{-qT}``."""
        dc = bsm.delta("c", m.S, m.K, m.T, m.r, m.sigma, m.q)
        dp = bsm.delta("p", m.S, m.K, m.T, m.r, m.sigma, m.q)
        assert dc - dp == pytest.approx(np.exp(-m.q * m.T), abs=1e-14)

    @pytest.mark.parametrize("m", MARKET_SAMPLE, ids=repr)
    def test_gamma_and_vega_are_type_independent(self, m: Market) -> None:
        """Parity is linear in S, so its curvature and vol sensitivity vanish."""
        for greek in (bsm.gamma, bsm.vega):
            call_value = greek("c", m.S, m.K, m.T, m.r, m.sigma, m.q)
            put_value = greek("p", m.S, m.K, m.T, m.r, m.sigma, m.q)
            assert call_value == pytest.approx(put_value, rel=1e-15)


class TestNoArbitrageBounds:
    """Prices must stay inside their static no-arbitrage band."""

    @pytest.mark.parametrize("m", MARKET_SAMPLE, ids=repr)
    def test_price_within_bounds(self, m: Market) -> None:
        price = bsm.price(m.option_type, m.S, m.K, m.T, m.r, m.sigma, m.q)
        lower, upper = bsm.no_arbitrage_bounds(m.option_type, m.S, m.K, m.T, m.r, m.q)
        assert lower - 1e-10 <= price <= upper + 1e-10

    def test_price_within_bounds_across_full_grid(self) -> None:
        for m in MARKET_GRID:
            price = bsm.price(m.option_type, m.S, m.K, m.T, m.r, m.sigma, m.q)
            lower, upper = bsm.no_arbitrage_bounds(m.option_type, m.S, m.K, m.T, m.r, m.q)
            tol = 1e-9 * m.S
            assert lower - tol <= price <= upper + tol, (
                f"{m} price {price} outside [{lower}, {upper}]"
            )

    def test_price_is_never_negative(self) -> None:
        for m in MARKET_GRID:
            assert bsm.price(m.option_type, m.S, m.K, m.T, m.r, m.sigma, m.q) >= 0.0


class TestGreekSigns:
    """Sign and range conventions that any correct implementation satisfies."""

    def test_call_delta_in_unit_interval(self) -> None:
        for m in MARKET_GRID:
            d = bsm.delta("c", m.S, m.K, m.T, m.r, m.sigma, m.q)
            assert 0.0 <= d <= 1.0, f"{m} call delta {d}"

    def test_put_delta_in_negative_unit_interval(self) -> None:
        for m in MARKET_GRID:
            d = bsm.delta("p", m.S, m.K, m.T, m.r, m.sigma, m.q)
            assert -1.0 <= d <= 0.0, f"{m} put delta {d}"

    def test_gamma_and_vega_non_negative(self) -> None:
        for m in MARKET_GRID:
            assert bsm.gamma(m.option_type, m.S, m.K, m.T, m.r, m.sigma, m.q) >= 0.0
            assert bsm.vega(m.option_type, m.S, m.K, m.T, m.r, m.sigma, m.q) >= 0.0

    def test_call_rho_positive_put_rho_negative(self) -> None:
        for m in MARKET_GRID:
            assert bsm.rho("c", m.S, m.K, m.T, m.r, m.sigma, m.q) >= 0.0
            assert bsm.rho("p", m.S, m.K, m.T, m.r, m.sigma, m.q) <= 0.0

    def test_long_option_theta_is_negative_without_carry(self) -> None:
        """With zero rates and dividends, time decay is unambiguously a loss.

        Theta can legitimately turn positive for a deep in-the-money put (the
        strike earns interest) or with a large dividend yield, which is why
        this pins down only the no-carry case.
        """
        for option_type in ("c", "p"):
            for ratio in (0.8, 1.0, 1.25):
                value = bsm.theta(option_type, 100.0, 100.0 * ratio, 0.5, 0.0, 0.25, 0.0)
                assert value < 0.0, f"{option_type} ratio {ratio} theta {value}"


class TestMonotonicity:
    """Directional responses the model must respect."""

    def test_call_price_increases_with_spot(self) -> None:
        spots = np.linspace(50.0, 200.0, 300)
        prices = bsm.price("c", spots, 100.0, 1.0, 0.02, 0.25, 0.0)
        assert np.all(np.diff(prices) > 0.0)

    def test_put_price_decreases_with_spot(self) -> None:
        spots = np.linspace(50.0, 200.0, 300)
        prices = bsm.price("p", spots, 100.0, 1.0, 0.02, 0.25, 0.0)
        assert np.all(np.diff(prices) < 0.0)

    @pytest.mark.parametrize("option_type", ["c", "p"])
    def test_price_increases_with_volatility(self, option_type: str) -> None:
        """Vega is non-negative, so price must be non-decreasing in vol."""
        vols = np.linspace(0.01, 2.0, 300)
        prices = bsm.price(option_type, 100.0, 105.0, 1.0, 0.02, vols, 0.0)
        assert np.all(np.diff(prices) >= -1e-12)

    def test_call_price_decreases_with_strike(self) -> None:
        strikes = np.linspace(50.0, 200.0, 300)
        prices = bsm.price("c", 100.0, strikes, 1.0, 0.02, 0.25, 0.0)
        assert np.all(np.diff(prices) < 0.0)

    def test_call_vertical_spread_is_not_wider_than_strike_gap(self) -> None:
        """``-dC/dK <= e^{-rT}``: a call spread cannot be worth more than its width."""
        r, T = 0.03, 1.0
        strikes = np.linspace(60.0, 160.0, 400)
        prices = bsm.price("c", 100.0, strikes, T, r, 0.3, 0.0)
        slope = -np.diff(prices) / np.diff(strikes)
        assert np.all(slope <= np.exp(-r * T) + 1e-9)

    def test_butterfly_is_non_negative(self) -> None:
        """Convexity in strike: the price must be a convex function of K."""
        strikes = np.linspace(60.0, 160.0, 400)
        prices = bsm.price("c", 100.0, strikes, 1.0, 0.03, 0.3, 0.0)
        second_difference = np.diff(prices, n=2)
        assert np.all(second_difference >= -1e-10)


# ---------------------------------------------------------------------------
# 2. Independent benchmark: QuantLib
# ---------------------------------------------------------------------------


@requires_quantlib
class TestQuantLibAgreement:
    """Agreement with an entirely separate implementation.

    Maturities are placed on exact day boundaries so our year-fraction clock
    and QuantLib's date calendar line up; otherwise the test would measure the
    calendar mismatch rather than the pricing maths.
    """

    CASES = [
        ("c", 5500.0, 5500.0, 91, 0.02, 0.20, 0.01),
        ("p", 5500.0, 5000.0, 182, 0.02, 0.22, 0.01),
        ("c", 100.0, 120.0, 365, 0.03, 0.35, 0.00),
        ("p", 100.0, 80.0, 730, 0.05, 0.15, 0.02),
        ("c", 100.0, 100.0, 7, 0.01, 0.80, 0.00),
        ("p", 100.0, 130.0, 1095, -0.01, 0.12, 0.00),
    ]

    @pytest.mark.quantlib
    @pytest.mark.parametrize(("option_type", "S", "K", "T_days", "r", "sigma", "q"), CASES)
    def test_price_matches_quantlib(
        self, option_type: str, S: float, K: float, T_days: int, r: float, sigma: float, q: float
    ) -> None:
        T = on_day_grid(T_days)
        reference = quantlib_european(option_type, S, K, T, r, sigma, q)
        ours = bsm.price(option_type, S, K, T, r, sigma, q)
        assert ours == pytest.approx(reference["price"], rel=1e-11, abs=1e-11)

    @pytest.mark.quantlib
    @pytest.mark.parametrize(("option_type", "S", "K", "T_days", "r", "sigma", "q"), CASES)
    def test_greeks_match_quantlib(
        self, option_type: str, S: float, K: float, T_days: int, r: float, sigma: float, q: float
    ) -> None:
        T = on_day_grid(T_days)
        reference = quantlib_european(option_type, S, K, T, r, sigma, q)
        ours = bsm.greeks(option_type, S, K, T, r, sigma, q)
        assert ours.delta == pytest.approx(reference["delta"], rel=1e-11, abs=1e-12)
        assert ours.gamma == pytest.approx(reference["gamma"], rel=1e-11, abs=1e-14)
        assert ours.vega == pytest.approx(reference["vega"], rel=1e-11, abs=1e-12)
        assert ours.rho == pytest.approx(reference["rho"], rel=1e-11, abs=1e-12)


# ---------------------------------------------------------------------------
# 3. Analytic Greeks against finite differences
# ---------------------------------------------------------------------------


class TestGreeksAgainstFiniteDifference:
    """Each closed-form Greek must match a numerical derivative of the price.

    This is the check that catches a transcription error in a Greek formula
    even when the price function itself is correct -- the two are independent
    code paths here.
    """

    @pytest.mark.parametrize("m", MARKET_SAMPLE, ids=repr)
    def test_delta(self, m: Market) -> None:
        h = m.S * 1e-5
        up = bsm.price(m.option_type, m.S + h, m.K, m.T, m.r, m.sigma, m.q)
        down = bsm.price(m.option_type, m.S - h, m.K, m.T, m.r, m.sigma, m.q)
        numerical = (up - down) / (2.0 * h)
        analytic = bsm.delta(m.option_type, m.S, m.K, m.T, m.r, m.sigma, m.q)
        assert analytic == pytest.approx(numerical, rel=1e-6, abs=1e-9)

    @pytest.mark.parametrize("m", MARKET_SAMPLE, ids=repr)
    def test_gamma(self, m: Market) -> None:
        # Gamma needs a larger step: the second difference loses roughly twice
        # as many digits to cancellation as the first.
        h = m.S * 1e-3
        up = bsm.price(m.option_type, m.S + h, m.K, m.T, m.r, m.sigma, m.q)
        mid = bsm.price(m.option_type, m.S, m.K, m.T, m.r, m.sigma, m.q)
        down = bsm.price(m.option_type, m.S - h, m.K, m.T, m.r, m.sigma, m.q)
        numerical = (up - 2.0 * mid + down) / (h * h)
        analytic = bsm.gamma(m.option_type, m.S, m.K, m.T, m.r, m.sigma, m.q)
        assert analytic == pytest.approx(numerical, rel=1e-4, abs=1e-10)

    @pytest.mark.parametrize("m", MARKET_SAMPLE, ids=repr)
    def test_vega(self, m: Market) -> None:
        h = 1e-6
        up = bsm.price(m.option_type, m.S, m.K, m.T, m.r, m.sigma + h, m.q)
        down = bsm.price(m.option_type, m.S, m.K, m.T, m.r, m.sigma - h, m.q)
        # Analytic vega is per percentage point; the FD derivative is per unit.
        numerical = (up - down) / (2.0 * h) * bsm.VEGA_SCALE
        analytic = bsm.vega(m.option_type, m.S, m.K, m.T, m.r, m.sigma, m.q)
        assert analytic == pytest.approx(numerical, rel=1e-6, abs=1e-10)

    @pytest.mark.parametrize("m", MARKET_SAMPLE, ids=repr)
    def test_rho(self, m: Market) -> None:
        h = 1e-7
        up = bsm.price(m.option_type, m.S, m.K, m.T, m.r + h, m.sigma, m.q)
        down = bsm.price(m.option_type, m.S, m.K, m.T, m.r - h, m.sigma, m.q)
        numerical = (up - down) / (2.0 * h) * bsm.RHO_SCALE
        analytic = bsm.rho(m.option_type, m.S, m.K, m.T, m.r, m.sigma, m.q)
        assert analytic == pytest.approx(numerical, rel=1e-5, abs=1e-9)

    @pytest.mark.parametrize("m", MARKET_SAMPLE, ids=repr)
    def test_theta(self, m: Market) -> None:
        """Theta is ``-dV/dT`` per day: time to maturity shrinks as time passes."""
        h = min(1e-6, m.T * 1e-3)
        later = bsm.price(m.option_type, m.S, m.K, m.T - h, m.r, m.sigma, m.q)
        earlier = bsm.price(m.option_type, m.S, m.K, m.T + h, m.r, m.sigma, m.q)
        numerical = (later - earlier) / (2.0 * h) * bsm.THETA_SCALE
        analytic = bsm.theta(m.option_type, m.S, m.K, m.T, m.r, m.sigma, m.q)
        assert analytic == pytest.approx(numerical, rel=1e-5, abs=1e-9)

    @pytest.mark.parametrize("m", MARKET_SAMPLE, ids=repr)
    def test_combined_greeks_match_individual_functions(self, m: Market) -> None:
        """``greeks()`` shares one d1/d2 evaluation; it must still agree exactly."""
        combined = bsm.greeks(m.option_type, m.S, m.K, m.T, m.r, m.sigma, m.q)
        args = (m.option_type, m.S, m.K, m.T, m.r, m.sigma, m.q)
        assert combined.delta == pytest.approx(bsm.delta(*args), rel=1e-14, abs=1e-15)
        assert combined.gamma == pytest.approx(bsm.gamma(*args), rel=1e-14, abs=1e-18)
        assert combined.vega == pytest.approx(bsm.vega(*args), rel=1e-14, abs=1e-15)
        assert combined.theta == pytest.approx(bsm.theta(*args), rel=1e-13, abs=1e-15)
        assert combined.rho == pytest.approx(bsm.rho(*args), rel=1e-14, abs=1e-15)


# ---------------------------------------------------------------------------
# 4. Implied volatility
# ---------------------------------------------------------------------------


class TestImpliedVolatility:
    """Round-trip behaviour and the handling of quotes that carry no vol info."""

    @pytest.mark.parametrize("m", MARKET_SAMPLE, ids=repr)
    def test_round_trip_recovers_input_volatility(self, m: Market) -> None:
        premium = bsm.price(m.option_type, m.S, m.K, m.T, m.r, m.sigma, m.q)
        if bsm.vol_uncertainty(m.option_type, m.S, m.K, m.T, m.r, m.sigma, m.q) > 1e-3:
            pytest.skip("quote does not identify a volatility; covered separately")
        recovered = bsm.implied_volatility(m.option_type, premium, m.S, m.K, m.T, m.r, m.q)
        assert recovered == pytest.approx(m.sigma, abs=1e-6)

    def test_round_trip_across_full_grid(self) -> None:
        """Sweep the grid; every quote either round-trips or is flagged.

        A silent third outcome -- returning a wrong volatility without
        complaint -- is what this is really guarding against.
        """
        checked = flagged = 0
        worst = 0.0
        for m in MARKET_GRID:
            premium = bsm.price(m.option_type, m.S, m.K, m.T, m.r, m.sigma, m.q)
            if premium < 1e-12:
                continue
            try:
                recovered = bsm.implied_volatility(m.option_type, premium, m.S, m.K, m.T, m.r, m.q)
            except ConvergenceError:
                flagged += 1
                continue
            checked += 1
            worst = max(worst, abs(recovered - m.sigma))
        assert checked > 1000, f"only {checked} identifiable quotes exercised"
        # The solver promises vol resolvable to within max_vol_uncertainty=1e-3.
        assert worst <= 1e-3, f"worst |IV - sigma| = {worst:.3e} exceeds the declared limit"
        assert flagged > 0, "expected some degenerate quotes in a grid this wide"

    def test_degenerate_quote_is_rejected_not_guessed(self) -> None:
        """A deep-ITM short-dated quote prices at intrinsic for any vol.

        Measured: a 1-day 100/60 call is identical to 12 decimal places for
        volatility anywhere between 1% and 100%. Returning the first value the
        solver lands on would be a plausible-looking number with no content.
        """
        T = 1.0 / 365.0
        premium = bsm.price("c", 100.0, 60.0, T, 0.06, 0.05, 0.0)
        # Confirm the premise: price is flat in vol here.
        assert bsm.price("c", 100.0, 60.0, T, 0.06, 1.00, 0.0) == pytest.approx(premium, abs=1e-10)
        with pytest.raises(ConvergenceError, match="does not pin down a volatility"):
            bsm.implied_volatility("c", premium, 100.0, 60.0, T, 0.06, 0.0)

    def test_degenerate_quote_accepted_when_screen_disabled(self) -> None:
        T = 1.0 / 365.0
        premium = bsm.price("c", 100.0, 60.0, T, 0.06, 0.05, 0.0)
        recovered = bsm.implied_volatility(
            "c", premium, 100.0, 60.0, T, 0.06, 0.0, max_vol_uncertainty=float("inf")
        )
        # The returned vol reproduces the price even though it is not the input.
        assert bsm.price("c", 100.0, 60.0, T, 0.06, recovered, 0.0) == pytest.approx(
            premium, abs=1e-7
        )

    @pytest.mark.parametrize(
        ("premium", "S", "K"),
        [
            (5600.0, 5500.0, 5500.0),  # above the discounted forward
            (1.0, 5500.0, 3000.0),  # below intrinsic
        ],
    )
    def test_price_outside_no_arbitrage_band_raises(
        self, premium: float, S: float, K: float
    ) -> None:
        with pytest.raises(ArbitrageError, match="no-arbitrage"):
            bsm.implied_volatility("c", premium, S, K, 0.25, 0.02, 0.01)

    @pytest.mark.parametrize("premium", [0.0, -1.0])
    def test_non_positive_premium_raises(self, premium: float) -> None:
        with pytest.raises(ParameterError):
            bsm.implied_volatility("c", premium, 5500.0, 5500.0, 0.25, 0.02, 0.01)

    def test_zero_maturity_raises(self) -> None:
        with pytest.raises(ParameterError):
            bsm.implied_volatility("c", 100.0, 5500.0, 5500.0, 0.0, 0.02, 0.01)

    def test_solver_survives_far_initial_guess(self) -> None:
        """Newton from a bad start must still land, via the Brent fallback."""
        premium = bsm.price("c", 100.0, 100.0, 1.0, 0.02, 0.30, 0.0)
        for guess in (0.001, 0.05, 1.0, 4.9):
            recovered = bsm.implied_volatility(
                "c", premium, 100.0, 100.0, 1.0, 0.02, 0.0, initial_guess=guess
            )
            assert recovered == pytest.approx(0.30, abs=1e-8)


class TestVolUncertainty:
    """The screening helper used to filter quote files before a surface fit."""

    def test_atm_quote_is_well_identified(self) -> None:
        u = bsm.vol_uncertainty("c", 5500.0, 5500.0, 0.25, 0.02, 0.20, 0.01)
        assert u < 1e-6

    def test_deep_itm_short_dated_is_unidentified(self) -> None:
        u = bsm.vol_uncertainty("c", 100.0, 60.0, 1.0 / 365.0, 0.06, 0.05, 0.0)
        assert not np.isfinite(u) or u > 1e-3

    def test_far_otm_is_unidentified(self) -> None:
        u = bsm.vol_uncertainty("c", 100.0, 160.0, 0.08, 0.02, 0.20, 0.0)
        assert u > 1e-3

    def test_uncertainty_scales_inversely_with_vega(self) -> None:
        """Halving the price tolerance must halve the vol uncertainty."""
        args = ("c", 100.0, 100.0, 1.0, 0.02, 0.25, 0.0)
        coarse = bsm.vol_uncertainty(*args, price_tolerance=1e-8)
        fine = bsm.vol_uncertainty(*args, price_tolerance=5e-9)
        assert fine == pytest.approx(coarse / 2.0, rel=1e-12)


# ---------------------------------------------------------------------------
# 5. Vectorisation, edge cases, regression anchors
# ---------------------------------------------------------------------------


class TestVectorisation:
    """Array inputs must agree with scalar calls element by element."""

    def test_array_price_matches_scalar_loop(self) -> None:
        spots = np.linspace(3000.0, 8000.0, 97)
        vector = bsm.price("c", spots, 5500.0, 0.25, 0.02, 0.20, 0.01)
        scalar = np.array([bsm.price("c", float(s), 5500.0, 0.25, 0.02, 0.20, 0.01) for s in spots])
        np.testing.assert_allclose(vector, scalar, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "greek", [bsm.delta, bsm.gamma, bsm.vega, bsm.theta, bsm.rho], ids=lambda f: f.__name__
    )
    def test_array_greeks_match_scalar_loop(self, greek) -> None:  # noqa: ANN001
        spots = np.linspace(3000.0, 8000.0, 53)
        vector = greek("c", spots, 5500.0, 0.25, 0.02, 0.20, 0.01)
        scalar = np.array([greek("c", float(s), 5500.0, 0.25, 0.02, 0.20, 0.01) for s in spots])
        np.testing.assert_allclose(vector, scalar, rtol=0, atol=0)

    def test_scalar_input_returns_float_not_array(self) -> None:
        result = bsm.price("c", 5500.0, 5500.0, 0.25, 0.02, 0.20, 0.01)
        assert isinstance(result, float)

    def test_array_input_returns_array(self) -> None:
        result = bsm.price("c", np.array([5400.0, 5500.0]), 5500.0, 0.25, 0.02, 0.20, 0.01)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_broadcasting_over_strikes_and_vols(self) -> None:
        """A strike x vol grid in one call, as a surface fit needs."""
        strikes = np.array([4500.0, 5000.0, 5500.0, 6000.0])[:, None]
        vols = np.array([0.15, 0.20, 0.25])[None, :]
        grid = bsm.price("c", 5500.0, strikes, 0.25, 0.02, vols, 0.01)
        assert grid.shape == (4, 3)
        # Prices fall across strikes and rise across vols.
        assert np.all(np.diff(grid, axis=0) < 0.0)
        assert np.all(np.diff(grid, axis=1) > 0.0)


class TestEdgeCases:
    """Degenerate limits must give the right answer rather than NaN."""

    def test_vanishing_volatility_gives_discounted_forward_intrinsic(self) -> None:
        S, K, T, r, q = 100.0, 90.0, 1.0, 0.03, 0.01
        call = bsm.price("c", S, K, T, r, 1e-14, q)
        expected = max(0.0, S * np.exp(-q * T) - K * np.exp(-r * T))
        assert call == pytest.approx(expected, abs=1e-9)
        assert np.isfinite(call)

    def test_vanishing_volatility_out_of_the_money_is_worthless(self) -> None:
        call = bsm.price("c", 100.0, 200.0, 1.0, 0.0, 1e-14, 0.0)
        assert call == pytest.approx(0.0, abs=1e-12)

    def test_very_short_maturity_is_finite(self) -> None:
        for option_type in ("c", "p"):
            value = bsm.price(option_type, 100.0, 100.0, 1e-10, 0.02, 0.2, 0.0)
            assert np.isfinite(value)
            assert value >= 0.0

    def test_degenerate_inputs_produce_no_nan_in_greeks(self) -> None:
        for greek in (bsm.delta, bsm.gamma, bsm.vega, bsm.theta, bsm.rho):
            value = greek("c", 100.0, 100.0, 1e-12, 0.02, 1e-12, 0.0)
            assert np.isfinite(value), f"{greek.__name__} produced {value}"

    def test_negative_rate_is_accepted(self) -> None:
        """Negative rates are real and must not be rejected."""
        value = bsm.price("c", 100.0, 100.0, 1.0, -0.005, 0.2, 0.0)
        assert value > 0.0

    @pytest.mark.parametrize("bad_input", [{"S": -1.0}, {"K": 0.0}, {"T": -0.5}, {"sigma": 0.0}])
    def test_invalid_inputs_rejected(self, bad_input: dict[str, float]) -> None:
        args = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.02, "sigma": 0.2, "q": 0.0}
        args.update(bad_input)
        with pytest.raises(ParameterError):
            bsm.price("c", **args)

    def test_unknown_option_type_rejected(self) -> None:
        with pytest.raises(ParameterError, match="not recognised"):
            bsm.price("x", 100.0, 100.0, 1.0, 0.02, 0.2, 0.0)

    @pytest.mark.parametrize(
        ("spelling", "expected"),
        [
            ("c", OptionType.CALL),
            ("C", OptionType.CALL),
            ("call", OptionType.CALL),
            ("p", OptionType.PUT),
            ("PUT", OptionType.PUT),
        ],
    )
    def test_legacy_option_type_spellings_accepted(
        self, spelling: str, expected: OptionType
    ) -> None:
        """Stored JSON and old call sites use the bare 'c'/'p' strings."""
        assert OptionType.parse(spelling) is expected


class TestBaselineRegression:
    """Anchors measured on the v7.0.0 tag, before any of this refactoring.

    These exist so a future change cannot quietly move a number the project
    has already published in its README and screenshots.
    """

    def test_baseline_price(self) -> None:
        value = bsm.price(*BASELINE)
        assert value == pytest.approx(BASELINE_EXPECTED["price"], abs=1e-4)

    @pytest.mark.parametrize("greek", ["delta", "gamma", "vega", "theta", "rho"])
    def test_baseline_greeks(self, greek: str) -> None:
        computed = bsm.greeks(*BASELINE)
        assert getattr(computed, greek) == pytest.approx(BASELINE_EXPECTED[greek], abs=1e-6)

    def test_baseline_implied_vol_round_trip(self) -> None:
        recovered = bsm.implied_volatility(
            BASELINE.option_type,
            BASELINE_EXPECTED["price"],
            BASELINE.S,
            BASELINE.K,
            BASELINE.T,
            BASELINE.r,
            BASELINE.q,
        )
        assert recovered == pytest.approx(BASELINE.sigma, abs=1e-5)
