"""Tests for the snowball quote report.

Three things are checked, and the first is the one the module exists for.

**The ordering is the argument.** Carry coverage leads, because 48% of a
one-year note's carry and 84% of a three-year note's is unmeasured, and because
carry more than doubles the quotable coupon at the level measured on the CSI
1000 board. A report that led with the coupon would misrepresent which input is
doing the work. So the section order is pinned, not merely the content.

**An absent measurement is reported, not omitted.** Every other section
disappears when its input is missing -- they cost solves, and a caller should
pay for what it wants. The carry section does not: with nothing measured it
still prints, and says the number is an assumption over the note's whole life.
An omitted heading would read as "nothing to report here", which is the
opposite of true.

**Every number carries its provenance.** The Greeks carry their method, theta
carries the fact that its two estimates are difference quotients over horizons
four times apart, and the carry carries the rate it is quoted against.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

import pytest

from otcpricer.core.market import EquityMarket
from otcpricer.engines.mc import MonteCarloConfig, price_snowball_mc
from otcpricer.engines.pde import PDEConfig
from otcpricer.engines.risk import compare_snowball_risk, snowball_risk_pde
from otcpricer.marketdata.carry import TailCarry, measure_carry
from otcpricer.products.snowball import SnowballTerms
from otcpricer.products.snowball_quote import decompose_par_coupon, solve_par_coupon
from otcpricer.reports.layout import WIDTH
from otcpricer.reports.snowball_report import snowball_report
from tests.test_carry import FakePanel, constant_carry_panel

#: Matches a hyphen that ``textwrap`` broke a word across, e.g. "bit-" at the
#: end of one line and "identical" at the start of the next.
_BROKEN_WORD = re.compile(r"-\n\s*(?=[a-z])")
_WHITESPACE = re.compile(r"\s+")


def unwrapped(text: str) -> str:
    """The report with its reflow undone, so prose can be asserted on.

    ``layout.wrap`` reflows to 78 columns and ``textwrap`` breaks after
    hyphens, so "bit-identical" arrives as "bit-" and "identical" on separate
    lines. Asserting against the wrapped form makes a test sensitive to the
    column width rather than to the content -- the first version of this file
    was, and four tests failed for that reason rather than for a real one.
    """
    return _WHITESPACE.sub(" ", _BROKEN_WORD.sub("-", text).replace("\n", " "))


AS_OF = date(2024, 12, 25)
FAST = PDEConfig(space_nodes=600, steps_per_year=504)
TERMS = SnowballTerms(S0=5000.0, T=1.0, ko_pct=1.03, ki_pct=0.75, coupon_pa=0.15)
MARKET = EquityMarket(S=5000.0, r=0.02, sigma=0.20, q=0.0, basis_pa=0.1015)


@pytest.fixture(scope="module")
def risk() -> object:
    return snowball_risk_pde(TERMS, MARKET, FAST)


@pytest.fixture(scope="module")
def measurement() -> object:
    """A board reaching 0.4849 years, as the real CFFEX one does."""
    return measure_carry(
        constant_carry_panel(0.1015, (23, 58, 86, 177)),
        "IM",
        "000852.SH",
        AS_OF,
        tail=TailCarry.HOLD_LAST,
        rate=0.02,
    )


class TestOrdering:
    def test_carry_coverage_leads(self, risk, measurement) -> None:  # type: ignore[no-untyped-def]
        """Before the quote, before the price, before the Greeks.

        The largest input is also the least observed, and a reader who stops
        after the first screen should learn that rather than the headline.
        """
        text = snowball_report(TERMS, MARKET, risk, carry=measurement)
        assert text.index("CARRY COVERAGE") < text.index("VALUE AND OUTCOMES")
        assert text.index("VALUE AND OUTCOMES") < text.index("RISK\n")
        assert "READ THIS FIRST" in text

    def test_the_quote_comes_before_the_price(self, risk, measurement) -> None:  # type: ignore[no-untyped-def]
        """Because the quote is the deliverable and the price is how it was found."""
        solution = solve_par_coupon(TERMS, MARKET, FAST)
        text = snowball_report(TERMS, MARKET, risk, carry=measurement, solution=solution)
        assert text.index("THE QUOTE") < text.index("VALUE AND OUTCOMES")


class TestCarrySection:
    def test_it_prints_the_measured_fraction_before_the_level(self, risk, measurement) -> None:  # type: ignore[no-untyped-def]
        text = snowball_report(TERMS, MARKET, risk, carry=measurement)
        assert text.index("MEASURED FRACTION") < text.index("near carry")
        assert "48.5% of the note's life" in text
        assert "52% of this note's carry is not measured" in unwrapped(text)

    def test_a_three_year_note_reports_a_far_worse_coverage(self, risk, measurement) -> None:  # type: ignore[no-untyped-def]
        """84% unmeasured, which is the number that should alarm a reader."""
        long_terms = SnowballTerms(S0=5000.0, T=3.0, ko_pct=1.03, ki_pct=0.75, coupon_pa=0.15)
        text = snowball_report(long_terms, MARKET, risk, carry=measurement)
        assert "16.2% of the note's life" in text
        assert "84% of this note's carry is not measured" in unwrapped(text)

    def test_no_measurement_still_prints_the_section(self, risk) -> None:  # type: ignore[no-untyped-def]
        """The one section that does not disappear when its input is missing.

        An omitted heading reads as "nothing to report", and what there is to
        report is that the whole carry is an assumption.
        """
        text = snowball_report(TERMS, MARKET, risk)
        assert "CARRY COVERAGE -- READ THIS FIRST" in text
        assert "supplied by the caller" in text
        assert "assumption over the whole life" in unwrapped(text)

    def test_it_says_the_board_shows_a_drift_not_a_basis(self, risk, measurement) -> None:  # type: ignore[no-untyped-def]
        """One observable against three unknowns, stated where it is read."""
        flat = unwrapped(snowball_report(TERMS, MARKET, risk, carry=measurement))
        assert "observes a drift, not a carry" in flat
        assert "q=0 and basis_pa=carry at r=2.00%" in flat

    def test_a_flat_term_structure_omits_the_shape_caveat(self, risk) -> None:  # type: ignore[no-untyped-def]
        """The caveat is conditional on the measurement, not boilerplate."""
        level = measure_carry(
            constant_carry_panel(0.1015, (23, 177)),
            "IM",
            "000852.SH",
            AS_OF,
            tail=TailCarry.HOLD_LAST,
            rate=0.02,
        )
        assert "term structure is not flat" not in unwrapped(
            snowball_report(TERMS, MARKET, risk, carry=level)
        )

        steep = FakePanel(
            contracts=(
                (AS_OF + timedelta(days=23), 5000.0 * 0.9995),
                (AS_OF + timedelta(days=177), 5000.0 * 0.93),
            )
        )
        bent = measure_carry(steep, "IM", "000852.SH", AS_OF, tail=TailCarry.HOLD_LAST, rate=0.02)
        assert "term structure is not flat" in unwrapped(
            snowball_report(TERMS, MARKET, risk, carry=bent)
        )


class TestQuoteSection:
    def test_the_split_is_reported_with_its_non_linearity(self, risk) -> None:  # type: ignore[no-untyped-def]
        """56% of the quote is the carry, and the slope that produced it is not
        constant -- so the report says the split is a difference of solves."""
        solution = solve_par_coupon(TERMS, MARKET, FAST)
        split = decompose_par_coupon(TERMS, MARKET, FAST)
        text = snowball_report(TERMS, MARKET, risk, solution=solution, decomposition=split)
        assert "what pays for it" in text
        assert "difference of two solves, not a sensitivity times a level" in unwrapped(text)
        assert split.basis_share > 0.5

    def test_the_affinity_residual_is_shown(self, risk) -> None:  # type: ignore[no-untyped-def]
        """Because it is the invariant the quote rests on, measured per call."""
        solution = solve_par_coupon(TERMS, MARKET, FAST)
        text = snowball_report(TERMS, MARKET, risk, solution=solution)
        assert "affinity residual" in text
        assert "check the affinity on every quote rather than assume it" in unwrapped(text)


class TestRiskSection:
    def test_each_greek_carries_its_method(self, risk) -> None:  # type: ignore[no-untyped-def]
        text = snowball_report(TERMS, MARKET, risk)
        assert "method               pde-grid" in text
        for name in ("delta", "gamma", "vega", "theta", "rho", "carry rho"):
            assert name in text

    def test_the_absent_dividend_sensitivity_is_explained(self, risk) -> None:  # type: ignore[no-untyped-def]
        """An absence explains nothing on its own, so the report explains it."""
        flat = unwrapped(snowball_report(TERMS, MARKET, risk))
        assert "no separate dividend one" in flat
        assert "bit-identical" in flat

    def test_rho_is_split_into_its_drift_and_discounting_halves(self, risk) -> None:  # type: ignore[no-untyped-def]
        """Which is what makes carry_rho worth printing beside rho."""
        assert "drift half of the rate exposure" in unwrapped(snowball_report(TERMS, MARKET, risk))


class TestValueSection:
    def test_the_loss_state_is_labelled(self, risk) -> None:  # type: ignore[no-untyped-def]
        """Knocked in and never out is the one that costs money."""
        mc = price_snowball_mc(
            TERMS,
            MARKET,
            MonteCarloConfig(paths=20_000, seed=5, memory_budget_mb=4096.0),
        )
        text = snowball_report(TERMS, MARKET, risk, mc=mc)
        assert "<- the loss state" in text
        assert "knocked in, not out" in text

    def test_no_simulation_says_why_the_probabilities_are_missing(self, risk) -> None:  # type: ignore[no-untyped-def]
        assert "path statistics, not a price" in unwrapped(snowball_report(TERMS, MARKET, risk))


class TestReconciliation:
    def test_theta_is_reported_with_how_weak_its_pass_is(self) -> None:
        """The newest row, and the one most easily over-read.

        Its two numbers are difference quotients over horizons four times
        apart, and three sampling errors come to a large share of the value.
        Both go in the report rather than only in the verdict dict.
        """
        comparison = compare_snowball_risk(
            TERMS,
            MARKET,
            pde_config=FAST,
            mc_config=MonteCarloConfig(
                paths=50_000, steps_per_year=252, seed=42, memory_budget_mb=8192.0
            ),
        )
        risk_result = snowball_risk_pde(TERMS, MARKET, FAST)
        text = snowball_report(TERMS, MARKET, risk_result, comparison=comparison)
        flat = unwrapped(text)

        assert "TWO-ENGINE RECONCILIATION" in text
        assert "horizons four" in flat
        assert "agreement rather than as confirmation" in flat
        assert "advancing the valuation date" in flat
        for name in ("delta", "vega", "rho", "carry_rho", "gamma", "theta"):
            assert name in text


class TestRendering:
    def test_every_line_fits_the_report_width(self, risk, measurement) -> None:  # type: ignore[no-untyped-def]
        """The house format is a terminal, so an overflowing line is a defect."""
        solution = solve_par_coupon(TERMS, MARKET, FAST)
        mc = price_snowball_mc(
            TERMS,
            MARKET,
            MonteCarloConfig(paths=20_000, seed=5, memory_budget_mb=4096.0),
        )
        text = snowball_report(
            TERMS,
            MARKET,
            risk,
            carry=measurement,
            solution=solution,
            decomposition=decompose_par_coupon(TERMS, MARKET, FAST),
            mc=mc,
        )
        too_long = [line for line in text.splitlines() if len(line) > WIDTH]
        assert not too_long, too_long

    def test_it_renders_with_nothing_but_the_risk(self, risk) -> None:  # type: ignore[no-untyped-def]
        """The minimum call, so a caller is never forced to pay for solves."""
        text = snowball_report(TERMS, MARKET, risk)
        assert text.startswith("=")
        assert text.endswith("=" * WIDTH)
        assert "THE QUOTE" not in text
