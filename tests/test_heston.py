"""Tests for the Heston model and the COS engine.

Four things carry the weight here, in order of how much they would cost to get
wrong silently:

1. **The martingale identity.** ``phi(-i) = E[S_T/S_0]`` must equal
   ``exp((r-q)T)`` exactly. It is one evaluation, it exercises every term of
   the characteristic function at once, and almost any algebra slip breaks it.
2. **COS against a Black-Scholes characteristic function.** The engine is
   model-agnostic, so feeding it a Gaussian transform checks the quadrature and
   the payoff coefficients against a closed form, with no stochastic volatility
   in the way.
3. **The Little Heston Trap, demonstrated rather than asserted.** The classic
   formulation is kept in the module for this purpose, and the test requires it
   to actually fail -- a hazard nobody can reproduce is one nobody believes.
4. **The truncation range floors, not converges.** Too narrow an ``L`` does not
   converge slowly to the right answer; it converges quickly to a wrong one.
   That distinction is the difference between a tolerance and a defect.
"""

from __future__ import annotations

import numpy as np
import pytest

from otcpricer.core.exceptions import EngineError, ParameterError
from otcpricer.engines.fourier import (
    DEFAULT_L,
    DEFAULT_TERMS,
    price_european_cos,
    price_heston_european,
)
from otcpricer.models import bsm
from otcpricer.models.heston import (
    HestonParams,
    _characteristic_function_classic,
    characteristic_function,
    cumulants,
)

# A well-behaved set, Feller satisfied.
TEXTBOOK = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)

# The fits this project's own data produces. Feller is violated on both.
CALIBRATED_510050 = HestonParams(v0=0.0272, kappa=4.857, theta=0.0496, sigma=1.960, rho=-0.208)
CALIBRATED_510300 = HestonParams(v0=0.0459, kappa=1.528, theta=0.0913, sigma=1.343, rho=-0.283)

# Deliberately extreme: a wide three-year density, which is what broke the
# call-coefficient route.
HIGH_VOL = HestonParams(v0=0.25, kappa=0.5, theta=0.30, sigma=1.5, rho=-0.9)

ALL_PARAMS = [
    ("textbook", TEXTBOOK),
    ("510050", CALIBRATED_510050),
    ("510300", CALIBRATED_510300),
    ("high vol", HIGH_VOL),
]


class TestParameters:
    @pytest.mark.parametrize(
        ("kwargs", "fragment"),
        [
            ({"v0": 0.0}, "v0 must be positive"),
            ({"v0": -0.01}, "v0 must be positive"),
            ({"kappa": 0.0}, "kappa must be positive"),
            ({"theta": -0.1}, "theta must be positive"),
            ({"sigma": 0.0}, "sigma must be positive"),
            ({"rho": 1.0}, "rho must lie in"),
            ({"rho": -1.0}, "rho must lie in"),
            ({"rho": 1.5}, "rho must lie in"),
        ],
    )
    def test_refuses_an_inadmissible_parameter(
        self, kwargs: dict[str, float], fragment: str
    ) -> None:
        base = {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma": 0.3, "rho": -0.7}
        base.update(kwargs)
        with pytest.raises(ParameterError, match=fragment):
            HestonParams(**base)  # type: ignore[arg-type]

    def test_rho_uses_the_open_interval(self) -> None:
        """Same reason SSVI's rho does: at |rho| = 1 the model degenerates.

        The two Brownian motions become the same one, and the stochastic
        volatility collapses into a single driver.
        """
        HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=0.999)
        with pytest.raises(ParameterError):
            HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=1.0)

    def test_feller_is_reported_not_enforced(self) -> None:
        """Because the calibrated fits violate it.

        A constructor that rejected a negative Feller margin would reject this
        project's own best fit to its own data, which is a sign the constraint
        does not belong there. What it does mean is that the QE scheme is
        mandatory for simulation.
        """
        assert CALIBRATED_510050.feller_margin == pytest.approx(-3.3598, abs=1e-3)
        assert CALIBRATED_510300.feller_margin == pytest.approx(-1.5246, abs=1e-3)
        assert not CALIBRATED_510050.satisfies_feller
        assert not CALIBRATED_510300.satisfies_feller

        assert TEXTBOOK.feller_margin == pytest.approx(0.07, abs=1e-9)
        assert TEXTBOOK.satisfies_feller

    def test_reads_as_volatilities_too(self) -> None:
        """v0 = 0.04 is a 20% vol, and confusing the two is a factor-of-25 error."""
        assert TEXTBOOK.initial_vol == pytest.approx(0.2)
        assert TEXTBOOK.long_run_vol == pytest.approx(0.2)

    def test_serialises_with_the_feller_caveat(self) -> None:
        payload = CALIBRATED_510050.as_dict()
        assert payload["satisfies_feller"] is False
        assert "QE scheme is required" in str(payload["feller_note"])


class TestCharacteristicFunction:
    @pytest.mark.parametrize(("name", "params"), ALL_PARAMS)
    @pytest.mark.parametrize("maturity", [0.1, 1.0, 3.0, 10.0])
    def test_is_one_at_zero(self, name: str, params: HestonParams, maturity: float) -> None:
        value = characteristic_function(params, np.array([0.0], dtype=complex), maturity)
        assert complex(value[0]) == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize(("name", "params"), ALL_PARAMS)
    @pytest.mark.parametrize("maturity", [0.1, 1.0, 3.0, 10.0])
    @pytest.mark.parametrize("drift", [0.0, 0.03, -0.02])
    def test_the_discounted_underlying_is_a_martingale(
        self, name: str, params: HestonParams, maturity: float, drift: float
    ) -> None:
        """``phi(-i) = E[S_T/S_0] = exp(drift * T)``, exactly.

        The single most informative check available on a characteristic
        function: one evaluation touches every term, and almost any algebra
        slip -- a sign, a factor of two, the wrong root -- breaks it. If this
        holds across maturities and drifts, the function is almost certainly
        right.
        """
        value = characteristic_function(
            params, np.array([-1j], dtype=complex), maturity, drift=drift
        )
        assert complex(value[0]).real == pytest.approx(np.exp(drift * maturity), rel=1e-11)
        assert complex(value[0]).imag == pytest.approx(0.0, abs=1e-11)

    @pytest.mark.parametrize(("name", "params"), ALL_PARAMS)
    def test_is_hermitian(self, name: str, params: HestonParams) -> None:
        """``phi(-u) = conj(phi(u))``, since the log return is real-valued."""
        u = np.array([0.5, 2.0, 10.0, 50.0], dtype=complex)
        forward = np.asarray(characteristic_function(params, u, 1.0))
        backward = np.asarray(characteristic_function(params, -u, 1.0))
        assert np.allclose(backward, np.conj(forward), rtol=1e-11)

    def test_decays_so_the_cosine_sum_converges(self) -> None:
        """Which is why COS works at all."""
        u = np.array([1.0, 10.0, 100.0, 1000.0], dtype=complex)
        magnitude = np.abs(np.asarray(characteristic_function(TEXTBOOK, u, 1.0)))
        assert np.all(np.diff(magnitude) < 0.0)
        assert magnitude[-1] < 1e-10

    def test_a_zero_maturity_is_the_degenerate_identity(self) -> None:
        u = np.array([0.0, 1.0, 10.0], dtype=complex)
        assert np.allclose(np.asarray(characteristic_function(TEXTBOOK, u, 0.0)), 1.0)

    def test_refuses_a_negative_maturity(self) -> None:
        with pytest.raises(ParameterError, match="non-negative"):
            characteristic_function(TEXTBOOK, np.array([1.0], dtype=complex), -1.0)


class TestTheLittleHestonTrap:
    """The two formulations, and the one that breaks.

    Kept as a demonstration rather than a claim. The classic form lives in the
    module for exactly this: a numerical hazard that cannot be reproduced is
    one nobody will believe, and someone will eventually "simplify" the root
    choice back.
    """

    U_GRID = np.linspace(0.01, 200.0, 4000)

    def test_the_two_forms_agree_where_both_are_stable(self) -> None:
        """To machine precision, which is the point: this is not an accuracy
        question but a branch of the complex logarithm."""
        for params in (TEXTBOOK, CALIBRATED_510050):
            for maturity in (0.25, 0.5, 1.0):
                stable = np.asarray(characteristic_function(params, self.U_GRID, maturity))
                classic = np.asarray(
                    _characteristic_function_classic(params, self.U_GRID, maturity)
                )
                assert np.all(np.isfinite(classic))
                assert np.abs(stable - classic).max() < 1e-14

    def test_the_classic_form_breaks_and_the_stable_one_does_not(self) -> None:
        """Required to fail. A test that tolerated success here would be vacuous.

        At the textbook parameters the classic form stays finite but develops
        discontinuities -- a largest adjacent jump of 7.2e-02 at two years
        against 7.9e-03 for the stable form -- because the logarithm has
        crossed a branch cut. At twenty years it is not finite at all.
        """
        with np.errstate(all="ignore"):
            stable = np.asarray(characteristic_function(TEXTBOOK, self.U_GRID, 2.0))
            classic = np.asarray(_characteristic_function_classic(TEXTBOOK, self.U_GRID, 2.0))

            stable_jump = np.abs(np.diff(stable)).max()
            classic_jump = np.abs(np.diff(classic)).max()

            assert stable_jump < 1e-2
            assert classic_jump > 5e-2
            assert classic_jump > 5.0 * stable_jump

            far = np.asarray(_characteristic_function_classic(TEXTBOOK, self.U_GRID, 20.0))
            assert not np.all(np.isfinite(far))

    def test_at_the_calibrated_parameters_it_overflows_inside_the_products_life(
        self,
    ) -> None:
        """Which makes the trap a hard requirement here, not a footnote.

        With ``sigma = 1.96`` the classic form stops being finite at about 1.85
        years -- evaluated out to the ``u`` that COS actually reaches at these
        settings, roughly 240. A snowball runs one to three years, so the
        unstable formulation would fail on the product this project exists to
        price.
        """
        with np.errstate(all="ignore"):
            at_one = np.asarray(
                _characteristic_function_classic(CALIBRATED_510050, self.U_GRID, 1.0)
            )
            assert np.all(np.isfinite(at_one))

            at_two = np.asarray(
                _characteristic_function_classic(CALIBRATED_510050, self.U_GRID, 2.0)
            )
            assert not np.all(np.isfinite(at_two))

            # And the stable form is fine an order of magnitude further out.
            far = np.asarray(characteristic_function(CALIBRATED_510050, self.U_GRID, 20.0))
            assert np.all(np.isfinite(far))

    def test_the_threshold_depends_on_how_far_out_u_goes(self) -> None:
        """Worth knowing, because it is why a coarse engine might not notice.

        ``d`` grows with ``u``, so a method evaluating only small ``u`` would
        survive longer. COS reaches about 240 at the default settings, which is
        why the failure is reached here.
        """
        with np.errstate(all="ignore"):
            narrow = np.linspace(0.01, 20.0, 400)
            assert np.all(
                np.isfinite(
                    np.asarray(_characteristic_function_classic(CALIBRATED_510050, narrow, 2.0))
                )
            )


class TestCumulants:
    @pytest.mark.parametrize(("name", "params"), ALL_PARAMS)
    @pytest.mark.parametrize("maturity", [0.25, 1.0, 3.0])
    def test_match_the_characteristic_function_they_came_from(
        self, name: str, params: HestonParams, maturity: float
    ) -> None:
        """They are derivatives of ``log phi`` at zero, so this is self-consistency.

        It is the property that makes the numerical route safe: whatever the
        engine integrates, the truncation range is derived from the same object.

        Differenced at the module's own step, deliberately. Comparing against a
        *different* step would just measure the difference between two
        truncation errors -- which is real but is not a defect, and at 1e-3
        against 1e-5 it is about 8e-06 relative. Since these numbers only size
        a truncation window that carries thirty standard deviations of margin,
        that distinction matters not at all for pricing and would make this
        test about finite differencing rather than about the cumulants.
        """
        c1, c2 = cumulants(params, maturity, drift=0.03, step=1e-3)

        h = 1e-3
        grid = np.array([-h, 0.0, h], dtype=complex)
        log_phi = np.log(np.asarray(characteristic_function(params, grid, maturity, drift=0.03)))
        c1_direct = float(np.real((log_phi[2] - log_phi[0]) / (2.0 * h) / 1j))
        c2_direct = float(np.real(-(log_phi[2] - 2.0 * log_phi[1] + log_phi[0]) / (h * h)))

        assert c1 == pytest.approx(c1_direct, rel=1e-12, abs=1e-14)
        assert c2 == pytest.approx(c2_direct, rel=1e-12)

    @pytest.mark.parametrize(("name", "params"), ALL_PARAMS)
    @pytest.mark.parametrize("maturity", [0.25, 1.0, 3.0])
    def test_the_mean_matches_its_closed_form(
        self, name: str, params: HestonParams, maturity: float
    ) -> None:
        """An independent check on ``c1``, which does have a short exact formula.

        ``c1 = drift*T + (1 - e^{-kappa T})(theta - v0) / (2 kappa) - theta T / 2``
        is three terms and verifiable by eye, unlike ``c2``. Checking the
        numerical route against it confirms the differencing is sound without
        putting a long transcription back into the module -- where the
        five-term ``c2`` is precisely what went wrong.
        """
        drift = 0.03
        kappa, theta, v0 = params.kappa, params.theta, params.v0
        expected = (
            drift * maturity
            + (1.0 - np.exp(-kappa * maturity)) * (theta - v0) / (2.0 * kappa)
            - 0.5 * theta * maturity
        )
        c1, _ = cumulants(params, maturity, drift=drift)
        assert c1 == pytest.approx(expected, rel=1e-5, abs=1e-9)

    @pytest.mark.parametrize(
        ("params", "maturity", "expected_c2"),
        [
            (TEXTBOOK, 1.0, 0.042470),
            (CALIBRATED_510050, 1.0, 0.049196),
            (HIGH_VOL, 1.0, 0.442434),
        ],
    )
    def test_agree_with_a_simulation(
        self, params: HestonParams, maturity: float, expected_c2: float
    ) -> None:
        """Pinned against a 400,000-path Euler run, which is the independent check.

        The high-volatility row is the one that matters. A transcribed
        closed-form ``c2`` gave **0.0888** there against this 0.4424 -- wrong by
        a factor of five, which halved the COS truncation width and produced
        finite, plausible, badly wrong prices. Differentiating the
        characteristic function instead makes that class of error impossible.
        """
        _, c2 = cumulants(params, maturity)
        assert c2 == pytest.approx(expected_c2, rel=1e-4)

    def test_the_variance_is_floored_not_absolute_valued(self) -> None:
        """An absolute value would hide exactly the defect described above."""
        c1, c2 = cumulants(TEXTBOOK, 1.0)
        assert c2 > 0.0
        assert cumulants(TEXTBOOK, 0.0) == (0.0, 0.0)

    def test_the_mean_carries_the_drift(self) -> None:
        c1_zero, _ = cumulants(TEXTBOOK, 2.0, drift=0.0)
        c1_drift, _ = cumulants(TEXTBOOK, 2.0, drift=0.05)
        assert c1_drift - c1_zero == pytest.approx(0.05 * 2.0, rel=1e-8)


class TestCosAgainstBlackScholes:
    """The engine is model-agnostic, so a Gaussian transform is an exact anchor."""

    @staticmethod
    def _gaussian_cf(vol: float, maturity: float, drift: float):  # type: ignore[no-untyped-def]
        def char_fn(u):  # type: ignore[no-untyped-def]
            return np.exp(
                1j * u * (drift - 0.5 * vol * vol) * maturity - 0.5 * vol * vol * maturity * u * u
            )

        return char_fn

    @pytest.mark.parametrize("flag", ["c", "p"])
    @pytest.mark.parametrize("strike", [70.0, 100.0, 130.0])
    @pytest.mark.parametrize(("maturity", "vol"), [(0.25, 0.2), (1.0, 0.25), (3.0, 0.4)])
    def test_reproduces_the_closed_form(
        self, flag: str, strike: float, maturity: float, vol: float
    ) -> None:
        spot, rate = 100.0, 0.03
        c1 = (rate - 0.5 * vol * vol) * maturity
        half = DEFAULT_L * vol * np.sqrt(maturity)

        result = price_european_cos(
            self._gaussian_cf(vol, maturity, rate),
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            flag=flag,
            lower=c1 - half,
            upper=c1 + half,
            terms=DEFAULT_TERMS,
        )
        exact = float(bsm.price(flag, spot, strike, maturity, rate, vol))
        assert result.price == pytest.approx(exact, abs=1e-10)

    def test_a_narrow_range_is_wrong_rather_than_slow(self) -> None:
        """The distinction that matters when choosing a default.

        Clipping the density does not slow convergence down; it moves the
        answer. The error at a too-narrow range is identical at 256 terms and
        at 4096, because the missing tail is simply not in the integral.
        """
        spot, strike, maturity, vol, rate = 100.0, 100.0, 1.0, 0.25, 0.03
        exact = float(bsm.price("c", spot, strike, maturity, rate, vol))
        c1 = (rate - 0.5 * vol * vol) * maturity

        errors = []
        for terms in (256, 1024, 4096):
            result = price_european_cos(
                self._gaussian_cf(vol, maturity, rate),
                spot=spot,
                strike=strike,
                maturity=maturity,
                rate=rate,
                flag="c",
                lower=c1 - 1.5 * vol,
                upper=c1 + 1.5 * vol,
                terms=terms,
            )
            errors.append(abs(result.price - exact))

        assert min(errors) > 1e-4, "a 1.5-sigma range should be visibly clipped"
        assert max(errors) / min(errors) < 1.05, "more terms must not help"


class TestCosOnHeston:
    @pytest.mark.parametrize(("name", "params"), ALL_PARAMS)
    @pytest.mark.parametrize("maturity", [0.25, 1.0, 3.0])
    def test_put_call_parity(self, name: str, params: HestonParams, maturity: float) -> None:
        """Holds to machine precision -- but note *why*, which is not a check.

        Calls are derived from puts by parity, so for calls this identity is
        satisfied by construction and proves nothing about the model. It is
        kept because it still guards the discounting and the forward: the
        forward comes from ``phi(-i)`` rather than an assumed ``exp((r-q)T)``,
        and this would catch the two disagreeing.
        """
        spot, strike, rate = 100.0, 95.0, 0.03
        call = price_heston_european(
            params, spot=spot, strike=strike, maturity=maturity, rate=rate, flag="c"
        ).price
        put = price_heston_european(
            params, spot=spot, strike=strike, maturity=maturity, rate=rate, flag="p"
        ).price
        assert call - put == pytest.approx(spot - strike * np.exp(-rate * maturity), abs=1e-10)

    @pytest.mark.parametrize(("name", "params"), ALL_PARAMS)
    def test_prices_sit_inside_the_no_arbitrage_bounds(
        self, name: str, params: HestonParams
    ) -> None:
        spot, rate = 100.0, 0.03
        for maturity in (0.25, 1.0, 3.0):
            for strike in (60.0, 100.0, 150.0):
                call = price_heston_european(
                    params, spot=spot, strike=strike, maturity=maturity, rate=rate
                ).price
                discount = float(np.exp(-rate * maturity))
                assert call >= max(spot - strike * discount, 0.0) - 1e-8
                assert call <= spot + 1e-8

    @pytest.mark.parametrize(("name", "params"), ALL_PARAMS)
    def test_calls_fall_with_strike_and_rise_with_maturity(
        self, name: str, params: HestonParams
    ) -> None:
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        prices = [
            price_heston_european(params, spot=100.0, strike=k, maturity=1.0, rate=0.03).price
            for k in strikes
        ]
        assert all(b < a for a, b in zip(prices, prices[1:], strict=False))

        # Forward-struck, so the time value is unambiguous.
        short = price_heston_european(
            params, spot=100.0, strike=100.0, maturity=0.5, rate=0.0
        ).price
        long = price_heston_european(params, spot=100.0, strike=100.0, maturity=2.0, rate=0.0).price
        assert long > short

    def test_the_wide_density_that_broke_the_call_route_now_prices(self) -> None:
        """The regression for the ``exp(upper)`` overflow.

        At ``L = 20`` on a three-year high-volatility density the upper bound
        reaches 30.8, and ``exp(30.8)`` is 2.4e13 -- the engine returned
        3.9e13 as a price. Integrating the put and deriving the call by parity
        keeps every coefficient O(1).
        """
        result = price_heston_european(HIGH_VOL, spot=100.0, strike=100.0, maturity=3.0, rate=0.03)
        assert 0.0 < result.price < 100.0
        assert result.upper - result.lower > 40.0, "this density really is that wide"

    def test_a_degenerate_range_is_refused(self) -> None:
        with pytest.raises(EngineError, match="degenerate truncation range"):
            price_european_cos(
                lambda u: np.ones_like(u),
                spot=100.0,
                strike=100.0,
                maturity=1.0,
                rate=0.0,
                lower=1.0,
                upper=1.0,
            )

    @pytest.mark.parametrize(
        ("kwargs", "fragment"),
        [
            ({"spot": 0.0}, "must be positive"),
            ({"strike": -1.0}, "must be positive"),
            ({"maturity": 0.0}, "maturity must be positive"),
            ({"terms": 1}, "terms must be at least 2"),
        ],
    )
    def test_refuses_impossible_inputs(self, kwargs: dict[str, object], fragment: str) -> None:
        base: dict[str, object] = {
            "spot": 100.0,
            "strike": 100.0,
            "maturity": 1.0,
            "rate": 0.03,
            "lower": -1.0,
            "upper": 1.0,
            "terms": 64,
        }
        base.update(kwargs)
        with pytest.raises(ParameterError, match=fragment):
            price_european_cos(lambda u: np.ones_like(u), **base)  # type: ignore[arg-type]

    def test_reports_the_numerics_it_used(self) -> None:
        result = price_heston_european(TEXTBOOK, spot=100.0, strike=100.0, maturity=1.0, rate=0.03)
        payload = result.as_dict()
        assert payload["terms"] == DEFAULT_TERMS
        assert payload["method"] == "cos-fourier"
        assert len(payload["truncation"]) == 2  # type: ignore[arg-type]


class TestAgainstQuantLib:
    """Two independent references: a different method, and a different COS."""

    @staticmethod
    def _quantlib(params: HestonParams, strike: float, days: int, rate: float, engine: str):  # type: ignore[no-untyped-def]
        ql = pytest.importorskip("QuantLib")

        today = ql.Date(2, 1, 2024)
        ql.Settings.instance().evaluationDate = today
        day_count = ql.Actual365Fixed()
        maturity = today + ql.Period(days, ql.Days)

        rates = ql.YieldTermStructureHandle(ql.FlatForward(today, rate, day_count, ql.Continuous))
        dividends = ql.YieldTermStructureHandle(
            ql.FlatForward(today, 0.0, day_count, ql.Continuous)
        )
        process = ql.HestonProcess(
            rates,
            dividends,
            ql.QuoteHandle(ql.SimpleQuote(100.0)),
            params.v0,
            params.kappa,
            params.theta,
            params.sigma,
            params.rho,
        )
        model = ql.HestonModel(process)
        option = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Call, strike), ql.EuropeanExercise(maturity)
        )
        option.setPricingEngine(
            ql.AnalyticHestonEngine(model, 192)
            if engine == "analytic"
            else ql.COSHestonEngine(model)
        )
        return option.NPV()

    @pytest.mark.parametrize(
        ("name", "params"),
        [("textbook", TEXTBOOK), ("510050", CALIBRATED_510050), ("510300", CALIBRATED_510300)],
    )
    @pytest.mark.parametrize("days", [37, 91, 365, 730])
    @pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
    def test_matches_the_analytic_engine(
        self, name: str, params: HestonParams, days: int, strike: float
    ) -> None:
        """A genuinely different method -- numerical integration of the pricing
        integral, not a cosine expansion -- so agreement is real evidence."""
        rate = 0.03
        reference = self._quantlib(params, strike, days, rate, "analytic")
        mine = price_heston_european(
            params, spot=100.0, strike=strike, maturity=days / 365.0, rate=rate
        ).price
        assert mine == pytest.approx(reference, abs=5e-9)

    def test_matches_quantlibs_own_cos_engine_too(self) -> None:
        """Same method, independent implementation.

        Looser than the analytic comparison on purpose: QuantLib's COS engine
        chooses its own truncation and term count, and at these parameters it
        sits 1e-6 or so from the analytic engine while this one sits at 1e-12.
        The agreement that matters is with the analytic engine; this check is
        here to confirm the two cosine implementations are solving the same
        problem.
        """
        for params in (TEXTBOOK, CALIBRATED_510050):
            reference = self._quantlib(params, 100.0, 365, 0.03, "cos")
            mine = price_heston_european(
                params, spot=100.0, strike=100.0, maturity=1.0, rate=0.03
            ).price
            assert mine == pytest.approx(reference, abs=1e-4)
