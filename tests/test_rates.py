"""Tests for the two-point government discount curve.

The curve's own arithmetic is checked against closed forms and against
QuantLib. What gets the most attention, though, is the conventions: the file
quotes percentages of an annually compounded bond yield, and everything
downstream wants continuously compounded decimals. That conversion is worth
0.6 to 2.8 bp -- small enough to pass unnoticed and wrong enough to matter, so
it is pinned numerically rather than trusted.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from otcpricer.core.daycount import DayCount
from otcpricer.core.exceptions import MarketDataError
from otcpricer.marketdata.rates import (
    PANEL_TENORS,
    RATE_DAY_COUNT,
    RateCurve,
    rate_curve_from_panel,
)
from tests.test_wind import WIND_DIR, needs_panel

# A curve with round numbers, so the expected values are computable by hand.
CURVE = RateCurve(
    as_of=date(2024, 12, 25),
    tenors=(1.0, 10.0),
    zero_rates=(0.01, 0.019),
    source="test",
)


class TestInterpolation:
    def test_the_knots_come_back_exactly(self) -> None:
        """Not approximately: a curve that cannot reproduce its own inputs is wrong."""
        assert CURVE.zero_rate(1.0) == 0.01
        assert CURVE.zero_rate(10.0) == 0.019

    def test_linear_in_the_zero_rate_between_the_knots(self) -> None:
        # Nine years span 90 bp, so a tenth of a basis point per year... 10 bp.
        assert CURVE.zero_rate(2.0) == pytest.approx(0.011)
        assert CURVE.zero_rate(5.5) == pytest.approx(0.0145)

    def test_flat_below_the_first_knot(self) -> None:
        """Deliberate, and the one choice here that is tempting to get wrong.

        Extrapolating the 1y-10y slope down to zero feels principled. It is
        not: it would move the short rate a few basis points in the wrong
        direction, because the very short CNY end sits below the 1-year rather
        than above it.
        """
        for t in (0.0, 0.01, 0.25, 0.5, 0.999):
            assert CURVE.zero_rate(t) == 0.01

    def test_flat_beyond_the_last_knot(self) -> None:
        for t in (10.0, 15.0, 30.0):
            assert CURVE.zero_rate(t) == 0.019

    def test_vectorises(self) -> None:
        got = CURVE.zero_rate(np.array([0.25, 1.0, 2.0, 10.0, 30.0]))
        assert np.allclose(got, [0.01, 0.01, 0.011, 0.019, 0.019])

    def test_the_whole_option_panel_sits_in_the_flat_region(self) -> None:
        """Which is the honest summary of what a two-point curve can say.

        The longest listed tenor in the panel is 1.008 years, so for surface
        construction this curve is "the 1-year yield, flat, refreshed daily".
        """
        assert PANEL_TENORS[0] == 1.0
        assert CURVE.zero_rate(1.008) == pytest.approx(0.01, abs=2e-5)
        assert CURVE.short_rate == 0.01


class TestDiscounting:
    def test_discount_factor_is_the_closed_form(self) -> None:
        for t in (0.25, 1.0, 3.0, 10.0):
            expected = math.exp(-float(CURVE.zero_rate(t)) * t)
            assert CURVE.discount_factor(t) == pytest.approx(expected, rel=1e-15)

    def test_a_zero_maturity_discounts_to_one(self) -> None:
        assert CURVE.discount_factor(0.0) == pytest.approx(1.0)

    def test_discount_factors_decrease(self) -> None:
        t = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
        factors = np.asarray(CURVE.discount_factor(t))
        assert np.all(np.diff(factors) < 0.0)

    def test_the_forward_rate_reprices_the_discount_factors(self) -> None:
        """The identity a float leg actually depends on."""
        for t_from, t_to in [(0.0, 1.0), (1.0, 2.0), (2.0, 5.0), (0.5, 10.0)]:
            forward = CURVE.forward_rate(t_from, t_to)
            near = float(CURVE.discount_factor(t_from))
            far = float(CURVE.discount_factor(t_to))
            assert near * math.exp(-forward * (t_to - t_from)) == pytest.approx(far, rel=1e-14)

    def test_an_upward_curve_has_forwards_above_the_spot_rate(self) -> None:
        assert CURVE.forward_rate(1.0, 2.0) > float(CURVE.zero_rate(2.0))

    def test_refuses_a_reversed_or_degenerate_period(self) -> None:
        with pytest.raises(MarketDataError, match="t_to > t_from"):
            CURVE.forward_rate(2.0, 1.0)
        with pytest.raises(MarketDataError, match="t_to > t_from"):
            CURVE.forward_rate(1.0, 1.0)

    def test_refuses_a_negative_maturity(self) -> None:
        with pytest.raises(MarketDataError, match="negative"):
            CURVE.zero_rate(-1.0)


class TestConstruction:
    @pytest.mark.parametrize(
        ("tenors", "rates", "fragment"),
        [
            ((1.0, 10.0), (0.01,), "one rate per tenor"),
            ((), (), "at least one tenor"),
            ((10.0, 1.0), (0.019, 0.01), "must ascend"),
            ((0.0, 10.0), (0.01, 0.019), "must be positive"),
            ((-1.0, 10.0), (0.01, 0.019), "must be positive"),
            ((1.0, 10.0), (0.01, float("nan")), "must be finite"),
        ],
    )
    def test_refuses_an_incoherent_curve(
        self, tenors: tuple[float, ...], rates: tuple[float, ...], fragment: str
    ) -> None:
        with pytest.raises(MarketDataError, match=fragment):
            RateCurve(as_of=date(2024, 12, 25), tenors=tenors, zero_rates=rates)

    def test_uses_the_same_day_count_as_the_option_slices(self) -> None:
        """So the two sources' numbers stay comparable.

        The option slices take their year fraction from the feed's own day
        count, which is ``(expiry - as_of).days / 365`` -- ACT/365F exactly.
        """
        assert RATE_DAY_COUNT is DayCount.ACT_365F
        assert RATE_DAY_COUNT.year_fraction(date(2024, 12, 25), date(2025, 12, 25)) == 365 / 365.0

    def test_serialises_with_its_own_caveat(self) -> None:
        payload = CURVE.as_dict()
        assert payload["tenors"] == [1.0, 10.0]
        assert payload["short_rate"] == 0.01
        assert payload["slope"] == pytest.approx(0.009)
        # The claim this module exists to avoid overstating.
        assert "not a bootstrap" in str(payload["note"])


# ---------------------------------------------------------------------------
# Against QuantLib
# ---------------------------------------------------------------------------


class TestAgainstQuantLib:
    def test_discount_factors_match_a_quantlib_zero_curve(self) -> None:
        """An independent implementation of the same interpolation rule."""
        ql = pytest.importorskip("QuantLib")

        as_of = ql.Date(25, 12, 2024)
        ql.Settings.instance().evaluationDate = as_of
        day_count = ql.Actual365Fixed()
        dates = [as_of] + [as_of + ql.Period(int(t * 365), ql.Days) for t in CURVE.tenors]
        # QuantLib wants a rate at the anchor too; flat below the first knot is
        # exactly what this curve does, so the first rate repeats.
        rates = [CURVE.zero_rates[0], *CURVE.zero_rates]
        reference = ql.ZeroCurve(dates, rates, day_count, ql.NullCalendar(), ql.Linear())

        for years in (0.25, 1.0, 2.0, 5.0, 10.0):
            theirs = reference.discount(as_of + ql.Period(int(years * 365), ql.Days))
            assert CURVE.discount_factor(years) == pytest.approx(theirs, rel=5e-5)


# ---------------------------------------------------------------------------
# Against the panel
# ---------------------------------------------------------------------------


@needs_panel
class TestFromThePanel:
    @pytest.fixture(scope="class")
    @classmethod
    def panel(cls):  # type: ignore[no-untyped-def]
        from otcpricer.marketdata.wind import WindPanel

        return WindPanel(WIND_DIR)

    def test_converts_percent_and_annual_compounding(self, panel) -> None:  # type: ignore[no-untyped-def]
        """Two conversions, both free to get wrong and silent when wrong.

        2023-06-30 quotes 1.9656 and 2.6929, meaning percentages of an annually
        compounded bond yield. Continuously compounded that is log(1.019656) =
        1.9465% -- a 1.91 bp correction, below the parity noise but not worth
        skipping when it costs one call.
        """
        curve = rate_curve_from_panel(panel, date(2023, 6, 30))

        assert curve.zero_rates[0] == pytest.approx(math.log1p(0.019656), rel=1e-12)
        assert curve.zero_rates[1] == pytest.approx(math.log1p(0.026929), rel=1e-12)
        # Decimals, not percent -- the other half of the same trap.
        assert 0.0 < curve.zero_rates[0] < 0.1
        # And the correction has the sign and size measured.
        assert 0.0 < 0.019656 - curve.zero_rates[0] < 2.0e-4

    def test_covers_every_trading_day_in_the_panel(self, panel) -> None:  # type: ignore[no-untyped-def]
        """A clean superset, so there is no carry-forward rule to get wrong.

        The file holds all 725 option trading days plus 21 interbank-only ones,
        which is why a missing date is an error here rather than something to
        paper over with a previous-value fill.
        """
        for day in panel.trading_days:
            curve = rate_curve_from_panel(panel, day)
            assert curve.as_of == day
            assert 0.0 < curve.zero_rates[0] < 0.05

    def test_the_curve_was_never_inverted_over_the_panel(self, panel) -> None:  # type: ignore[no-untyped-def]
        """Measured, not assumed -- an inversion would not be an error."""
        for day in panel.trading_days:
            assert rate_curve_from_panel(panel, day).slope > 0.0

    def test_the_hardcoded_two_percent_was_wrong_most_days(self, panel) -> None:  # type: ignore[no-untyped-def]
        """The reason this module exists.

        The 1-year yield ran 0.83% to 2.40%, so the previous flat 2% was off by
        up to 1.17 percentage points and sat more than 50 bp away on 60.6% of
        trading days.
        """
        shorts = [rate_curve_from_panel(panel, d).short_rate for d in panel.trading_days]

        assert min(shorts) == pytest.approx(math.log1p(0.008297), rel=1e-9)
        assert max(shorts) == pytest.approx(math.log1p(0.023999), rel=1e-9)
        far = sum(1 for r in shorts if abs(r - 0.02) > 0.005)
        assert far / len(shorts) == pytest.approx(0.606, abs=0.02)

    def test_a_date_outside_the_file_is_an_error(self, panel) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(MarketDataError, match="no row for"):
            rate_curve_from_panel(panel, date(2001, 1, 1))

    def test_repeated_dates_with_identical_readings_collapse(self, panel) -> None:  # type: ignore[no-untyped-def]
        """The file's tail repeats three dates, with the same values both times.

        An identical repeat is an export artefact and is harmless. A
        *conflicting* one would mean the file disagrees with itself, which is
        not something to pick a winner from -- so that raises instead.
        """
        curve = rate_curve_from_panel(panel, date(2026, 6, 29))
        assert curve.zero_rates[0] == pytest.approx(math.log1p(0.011049), rel=1e-12)

    def test_the_snowball_tail_is_the_only_place_the_ten_year_shows_up(self, panel) -> None:  # type: ignore[no-untyped-def]
        """And the effect there is small, which is worth knowing before relying on it.

        At 2026-06-29 the two-year rate sits about 7 bp above the one-year, so
        a snowball's discounting moves by well under a tenth of a percent
        against using the short rate flat.
        """
        curve = rate_curve_from_panel(panel, date(2026, 6, 29))
        two_year = float(curve.zero_rate(2.0))

        assert two_year - curve.short_rate == pytest.approx(7.05e-4, abs=5e-5)
        flat = math.exp(-curve.short_rate * 2.0)
        assert abs(float(curve.discount_factor(2.0)) / flat - 1.0) < 1.5e-3
