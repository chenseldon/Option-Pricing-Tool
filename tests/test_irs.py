"""Tests for the interest rate swap, and for the three defects it replaces.

The two load-bearing tests are the floating-leg par identity and the
bump-and-reprice agreement with DV01. Both are chosen because they cannot be
satisfied by accident:

- A floating leg prices to par on *any* curve shape only when it resets off
  that curve at the right compounding. It caught a real convention bug here --
  using the continuously compounded forward instead of the simple one left a
  residue of 0.02% of notional on a 2-year semiannual swap and 1.0% on a
  10-year annual one. Plausible, small, and wrong.
- An analytic sensitivity nobody has differenced against a bump is exactly how
  the previous DV01 survived being four times too small.
"""

from __future__ import annotations

from datetime import date

import pytest

from otcpricer.core.exceptions import ParameterError
from otcpricer.marketdata.rates import RateCurve
from otcpricer.products.irs import ONE_BASIS_POINT, flat_curve, price_swap

AS_OF = date(2024, 12, 25)

FLAT = flat_curve(0.02, AS_OF)
STEEP = RateCurve(as_of=AS_OF, tenors=(1.0, 10.0), zero_rates=(0.005, 0.045))
INVERTED = RateCurve(as_of=AS_OF, tenors=(1.0, 10.0), zero_rates=(0.045, 0.005))

SHAPES = [("flat", FLAT), ("steep", STEEP), ("inverted", INVERTED)]
SCHEDULES = [(2.0, 2), (5.0, 4), (10.0, 1), (1.0, 12)]


class TestFloatingLegPricesToPar:
    """The decisive check: a float leg that is not worth par is not floating."""

    @pytest.mark.parametrize(("name", "curve"), SHAPES)
    @pytest.mark.parametrize(("maturity", "frequency"), SCHEDULES)
    def test_on_every_curve_shape_and_schedule(
        self, name: str, curve: RateCurve, maturity: float, frequency: int
    ) -> None:
        result = price_swap(curve, fixed_rate=0.03, maturity=maturity, frequency=frequency)
        assert result.pv_float == pytest.approx(result.notional, rel=1e-12)
        assert result.float_leg_prices_to_par

    def test_the_continuous_forward_would_fail_this(self) -> None:
        """Pinning the bug the identity caught, so it cannot come back.

        ``sum_i F_i*dt*df(t_i) + df(T) = 1`` telescopes only for the simple
        rate, because ``[df(t_{i-1})/df(t_i) - 1]*df(t_i) = df(t_{i-1}) -
        df(t_i)``. The continuous rate leaves an O((r*dt)^2) residue.
        """
        maturity, frequency = 10.0, 1
        dt = 1.0 / frequency
        notional = 1_000_000.0

        wrong = 0.0
        for i in range(1, int(maturity * frequency) + 1):
            t_start, t_end = (i - 1) * dt, i * dt
            # The continuous forward, which is what RateCurve.forward_rate is.
            forward = INVERTED.forward_rate(max(t_start, 1e-12), t_end)
            wrong += forward * dt * notional * float(INVERTED.discount_factor(t_end))
        wrong += notional * float(INVERTED.discount_factor(maturity))

        # Off by about a third of a percent of notional: small enough to look
        # like rounding, large enough to be a real mispricing.
        assert abs(wrong - notional) > 1e-3 * notional

        right = price_swap(INVERTED, fixed_rate=0.03, maturity=maturity, frequency=frequency)
        assert right.pv_float == pytest.approx(notional, rel=1e-12)


class TestDv01:
    @pytest.mark.parametrize(("name", "curve"), SHAPES)
    @pytest.mark.parametrize(("maturity", "frequency"), SCHEDULES)
    def test_agrees_with_bump_and_reprice(
        self, name: str, curve: RateCurve, maturity: float, frequency: int
    ) -> None:
        """Because an unchecked analytic sensitivity is how the old one survived."""
        base = price_swap(curve, fixed_rate=0.03, maturity=maturity, frequency=frequency)
        bumped = price_swap(
            curve,
            fixed_rate=0.03 + ONE_BASIS_POINT,
            maturity=maturity,
            frequency=frequency,
        )
        assert abs(bumped.npv - base.npv) == pytest.approx(base.dv01, rel=1e-10)

    def test_is_the_whole_annuity_not_the_final_discount_factor(self) -> None:
        """The defect, stated as a number.

        At T=2, freq=2, 2% flat on a million notional: the annuity gives
        195.07 and the old final-discount-factor formula gives 48.04, a factor
        of four. The old formula is reproduced here so the comparison is
        visible rather than asserted.
        """
        import numpy as np

        result = price_swap(FLAT, fixed_rate=0.035, maturity=2.0, frequency=2)
        old_formula = 1_000_000 * 0.5 * float(np.exp(-0.02 * 2.0)) * ONE_BASIS_POINT

        assert result.dv01 == pytest.approx(195.07, abs=0.01)
        assert old_formula == pytest.approx(48.04, abs=0.01)
        assert result.dv01 / old_formula == pytest.approx(4.06, abs=0.05)

    def test_scales_with_notional_and_grows_with_maturity(self) -> None:
        small = price_swap(FLAT, notional=1e6, fixed_rate=0.03, maturity=5.0, frequency=4)
        large = price_swap(FLAT, notional=5e6, fixed_rate=0.03, maturity=5.0, frequency=4)
        assert large.dv01 == pytest.approx(5.0 * small.dv01, rel=1e-12)

        longer = price_swap(FLAT, fixed_rate=0.03, maturity=10.0, frequency=4)
        assert longer.dv01 > small.dv01


class TestParRate:
    @pytest.mark.parametrize(("name", "curve"), SHAPES)
    def test_a_swap_struck_at_the_par_rate_is_worth_nothing(
        self, name: str, curve: RateCurve
    ) -> None:
        quoted = price_swap(curve, fixed_rate=0.03, maturity=5.0, frequency=4)
        at_par = price_swap(curve, fixed_rate=quoted.par_rate, maturity=5.0, frequency=4)
        assert at_par.npv == pytest.approx(0.0, abs=1e-6)

    def test_the_fixed_payer_gains_when_the_fixed_rate_is_below_par(self) -> None:
        """A sign check, which a symmetric formula would pass either way."""
        quoted = price_swap(STEEP, fixed_rate=0.03, maturity=5.0, frequency=4)
        cheap = price_swap(STEEP, fixed_rate=quoted.par_rate - 0.01, maturity=5.0, frequency=4)
        rich = price_swap(STEEP, fixed_rate=quoted.par_rate + 0.01, maturity=5.0, frequency=4)

        assert cheap.npv > 0.0
        assert rich.npv < 0.0

    def test_on_a_flat_curve_the_par_rate_sits_near_the_curve_level(self) -> None:
        """Near, not equal: the curve is continuous and the leg pays simple rates."""
        result = price_swap(FLAT, fixed_rate=0.02, maturity=5.0, frequency=4)
        assert result.par_rate == pytest.approx(0.02, abs=5e-4)
        assert result.par_rate > 0.02


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "fragment"),
        [
            ({"notional": 0.0}, "notional must be positive"),
            ({"notional": -1.0}, "notional must be positive"),
            ({"maturity": 0.0}, "maturity must be positive"),
            ({"maturity": -2.0}, "maturity must be positive"),
            ({"frequency": 0}, "frequency must be at least 1"),
            ({"maturity": 0.2, "frequency": 1}, "no payment dates"),
        ],
    )
    def test_refuses_an_impossible_swap(self, kwargs: dict[str, object], fragment: str) -> None:
        defaults: dict[str, object] = {"fixed_rate": 0.03, "maturity": 5.0, "frequency": 4}
        defaults.update(kwargs)
        with pytest.raises(ParameterError, match=fragment):
            price_swap(FLAT, **defaults)  # type: ignore[arg-type]

    def test_the_schedule_covers_the_maturity_exactly(self) -> None:
        result = price_swap(FLAT, fixed_rate=0.03, maturity=5.0, frequency=4)
        assert len(result.legs) == 20
        assert result.legs[-1].t_end == pytest.approx(5.0)
        assert result.legs[0].t_start == 0.0
        # No gaps or overlaps between accrual periods.
        for earlier, later in zip(result.legs, result.legs[1:], strict=False):
            assert later.t_start == pytest.approx(earlier.t_end)


class TestAgainstQuantLib:
    def test_the_par_rate_matches_a_quantlib_vanilla_swap(self) -> None:
        """An independent implementation, including its own schedule handling."""
        ql = pytest.importorskip("QuantLib")

        as_of = ql.Date(25, 12, 2024)
        ql.Settings.instance().evaluationDate = as_of
        day_count = ql.Actual365Fixed()
        calendar = ql.NullCalendar()

        rate = 0.02
        discount = ql.YieldTermStructureHandle(
            ql.FlatForward(as_of, rate, day_count, ql.Continuous)
        )
        index = ql.IborIndex(
            "test",
            ql.Period(3, ql.Months),
            0,
            ql.CNYCurrency(),
            calendar,
            ql.Unadjusted,
            False,
            day_count,
            discount,
        )

        maturity = as_of + ql.Period(5, ql.Years)
        schedule = ql.Schedule(
            as_of,
            maturity,
            ql.Period(3, ql.Months),
            calendar,
            ql.Unadjusted,
            ql.Unadjusted,
            ql.DateGeneration.Forward,
            False,
        )
        swap = ql.VanillaSwap(
            ql.VanillaSwap.Payer,
            1_000_000.0,
            schedule,
            0.03,
            day_count,
            schedule,
            index,
            0.0,
            day_count,
        )
        swap.setPricingEngine(ql.DiscountingSwapEngine(discount))

        mine = price_swap(FLAT, fixed_rate=0.03, maturity=5.0, frequency=4)

        # Loose on the schedule: QuantLib rolls to real calendar dates while
        # this pricer uses exact year fractions, so a few days of accrual
        # differ. A basis point on the par rate is well inside that.
        assert mine.par_rate == pytest.approx(swap.fairRate(), abs=1e-4)


class TestEndpoint:
    @pytest.fixture
    def client(self):  # type: ignore[no-untyped-def]
        import app as app_module

        app_module.app.config["TESTING"] = True
        return app_module.app.test_client()

    def test_reports_a_dv01_that_is_the_annuity(self, client) -> None:  # type: ignore[no-untyped-def]
        payload = client.post(
            "/api/irs", json={"fixed_rate": 0.035, "T": 2.0, "freq": 2, "r_disc": 0.02}
        ).get_json()

        assert payload["dv01"] == pytest.approx(195.07, abs=0.01)
        assert payload["annuity"] == pytest.approx(1.950742, abs=1e-5)
        assert payload["float_leg_prices_to_par"] is True
        assert payload["npv"] == pytest.approx(payload["pv_float"] - payload["pv_fixed"], abs=0.01)

    def test_carries_the_per_period_forward_rates(self, client) -> None:  # type: ignore[no-untyped-def]
        """So a caller can see the leg resetting rather than taking it on trust."""
        payload = client.post(
            "/api/irs", json={"fixed_rate": 0.03, "T": 2.0, "freq": 4, "r_disc": 0.02}
        ).get_json()

        flows = payload["cashflows"]
        assert len(flows) == 8
        assert all("forward_rate" in flow for flow in flows)
        assert all(flow["forward_rate"] > 0.0 for flow in flows)

    def test_the_bid_ask_straddles_the_npv(self, client) -> None:  # type: ignore[no-untyped-def]
        payload = client.post(
            "/api/irs",
            json={"fixed_rate": 0.03, "T": 2.0, "freq": 4, "r_disc": 0.02, "spread": 500.0},
        ).get_json()
        assert payload["bid"] == pytest.approx(payload["npv"] - 500.0, abs=0.01)
        assert payload["ask"] == pytest.approx(payload["npv"] + 500.0, abs=0.01)

    @pytest.mark.parametrize(
        ("body", "fragment"),
        [
            ({"T": 2.0}, "Invalid parameters"),
            ({"fixed_rate": 0.03}, "Invalid parameters"),
            ({"fixed_rate": 0.03, "T": 0.0}, "maturity must be positive"),
            ({"fixed_rate": 0.03, "T": 2.0, "freq": 0}, "frequency must be at least 1"),
            ({"fixed_rate": 0.03, "T": 2.0, "as_of": "nonsense"}, "YYYY-MM-DD"),
        ],
    )
    def test_refuses_malformed_requests(
        self, client, body: dict[str, object], fragment: str
    ) -> None:  # type: ignore[no-untyped-def]
        response = client.post("/api/irs", json=body)
        assert response.status_code == 400
        assert fragment in response.get_json()["error"]

    def test_reports_the_curve_it_used(self, client) -> None:  # type: ignore[no-untyped-def]
        payload = client.post(
            "/api/irs", json={"fixed_rate": 0.03, "T": 2.0, "r_disc": 0.025}
        ).get_json()
        curve = payload["curve"]
        assert curve["short_rate"] == pytest.approx(0.025)
        assert "not a bootstrap" in str(curve["note"])
