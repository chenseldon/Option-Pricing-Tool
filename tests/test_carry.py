"""Tests for carry measured off the futures board.

Two kinds of test here, and the first kind is the one that proves correctness.

**The constant-carry degenerate.** Build futures from ``F = S*exp((r-c)*tau)``
for a single ``c`` and every number the module produces must come back as
exactly ``c`` -- each contract's carry, the near and far carries, the average to
any horizon, and the term structure gap at zero. That reduction pins the
anchoring, the integration and the tail rule simultaneously, and it is exact
rather than approximate, so no tolerance is doing the work.

The tenors are built from whole days on purpose. The module recovers a tenor as
``(expiry - as_of).days / 365``, so a panel generated from fractional years
would come back shifted by the rounding and the degenerate would hold only to
2e-05 -- a tolerance loose enough to hide a genuine error. Whole days make it
exact.

**The panel measurements.** These pin facts about the data that the module's
design rests on: that the board reaches only half a year, that the front
contract's annualised drift is noise, and that carry is negative on about one
day in ten. If any stopped being true the design would need revisiting, so they
are asserted rather than only documented.

One correction of mine is recorded in the module docstring and worth repeating
here, because it is what these tests were rewritten to catch: I first measured
``-log(F/S)/tau`` and called it the carry. That is the carry at a zero rate.
This project defines ``carry = q + basis_pa`` with ``drift = r - carry``, so the
board observes the *drift* and the carry is ``r - drift``. The error was a full
2% a year against carries of 2% to 10%.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pytest

from otcpricer.core.exceptions import MarketDataError
from otcpricer.core.market import EquityMarket
from otcpricer.marketdata.carry import (
    MIN_CONTRACTS,
    MIN_FUTURES_TENOR,
    CarryMeasurement,
    TailCarry,
    measure_carry,
)
from otcpricer.marketdata.wind import WindPanel
from tests.test_wind import WIND_DIR, needs_panel

AS_OF = date(2024, 12, 25)
SPOT = 5000.0
RATE = 0.02

#: The reference board's tenors, in whole days: the four CFFEX contracts listed
#: on 2024-12-25, at 0.0630, 0.1589, 0.2356 and 0.4849 years.
REFERENCE_DAYS = (23, 58, 86, 177)


@dataclass
class FakePanel:
    """A futures board from a handful of tuples.

    The real panel is 1.3 million rows of institutional data that is not
    committed, so the module takes a two-method Protocol and the exact-value
    tests run against this instead. That is deliberate: a correctness proof
    should not depend on a file nobody else has.
    """

    contracts: tuple[tuple[date, float], ...]
    spot: float = SPOT

    def futures(self, root: str, as_of: date) -> tuple[tuple[date, float], ...]:
        return self.contracts

    def index_close(self, index_code: str, as_of: date) -> float:
        return self.spot


def constant_carry_panel(carry: float, days: tuple[int, ...], *, rate: float = RATE) -> FakePanel:
    """A board generated from one constant carry, for the degenerate test.

    ``F = S * exp((rate - carry) * tau)`` with ``tau = days / 365`` exactly,
    which is the tenor the module will recover.
    """
    contracts = tuple(
        (
            AS_OF + timedelta(days=d),
            SPOT * float(np.exp((rate - carry) * (d / 365.0))),
        )
        for d in days
    )
    return FakePanel(contracts=contracts)


def measure(panel: FakePanel, **kwargs: object) -> CarryMeasurement:
    params: dict[str, object] = {"tail": TailCarry.HOLD_LAST, "rate": RATE}
    params.update(kwargs)
    return measure_carry(panel, "IF", "000300.SH", AS_OF, **params)  # type: ignore[arg-type]


class TestTheConstantCarryDegenerate:
    """One carry in, the same carry out of every accessor."""

    @pytest.mark.parametrize("carry", [0.0, 0.0370, 0.1035, -0.0234, 0.25])
    def test_every_contract_reports_the_carry_it_was_built_from(self, carry: float) -> None:
        """Including a negative one, which is the case the old validation refused."""
        m = measure(constant_carry_panel(carry, REFERENCE_DAYS))
        for measured in m.contract_carries:
            assert measured == pytest.approx(carry, abs=1e-14)
        assert m.near_carry == pytest.approx(carry, abs=1e-14)
        assert m.far_carry == pytest.approx(carry, abs=1e-14)
        assert m.term_structure_gap == pytest.approx(0.0, abs=1e-14)

    def test_the_drift_is_what_is_observed_and_the_carry_is_derived(self) -> None:
        """The distinction I got wrong first time, asserted both ways.

        ``carry + drift == rate`` for every contract, by construction. If the
        two were ever measured independently this identity would be the thing
        that broke.
        """
        m = measure(constant_carry_panel(0.1035, REFERENCE_DAYS))
        for drift, carry in zip(m.contract_drifts, m.contract_carries, strict=True):
            assert drift + carry == pytest.approx(RATE, abs=1e-14)
        assert m.far_drift == pytest.approx(RATE - 0.1035, abs=1e-14)

    @pytest.mark.parametrize("T", [0.01, 23 / 365, 0.1, 0.3, 177 / 365, 1.0, 3.0])
    def test_the_average_to_any_horizon_is_that_carry_too(self, T: float) -> None:
        """Across the three regimes at once.

        ``T`` below the first contract tests the flat-hold, between contracts
        tests the curve's integration, and beyond the board tests the tail --
        and under a constant carry all three must return the same number, so
        one assertion covers every seam. A sign error or a missing anchor in any
        of them would show here.
        """
        carry = 0.1035
        m = measure(constant_carry_panel(carry, REFERENCE_DAYS))
        assert m.average_carry_to(T) == pytest.approx(carry, abs=1e-13)

    def test_the_forward_is_reproduced_exactly_at_each_listed_contract(self) -> None:
        """Which is what anchoring at the index close buys.

        ``ForwardCurve`` back-extrapolates its own ``F(0)`` from the first
        listed segment's drift, and that is not the index close. Anchoring on
        the close instead means the measured forwards come back exactly.
        """
        panel = constant_carry_panel(0.0370, REFERENCE_DAYS)
        m = measure(panel)
        for tenor, (_, settle) in zip(m.maturities, panel.contracts, strict=True):
            assert m.forward_to(tenor) == pytest.approx(settle, rel=1e-14)

    def test_the_measured_carry_reprices_the_board_through_an_equity_market(self) -> None:
        """The whole point of the exercise, end to end.

        Feed the measured carry in as ``q=0, basis_pa=carry`` and
        :class:`EquityMarket` must reproduce the futures settlement. This is
        the test that would have caught my rate error: at a zero-rate carry the
        forwards come back out by ``exp(r*tau)``, which is 1% at the board's
        far end.
        """
        panel = constant_carry_panel(0.1035, REFERENCE_DAYS)
        m = measure(panel)
        for tenor, (_, settle) in zip(m.maturities, panel.contracts, strict=True):
            market = EquityMarket(
                S=m.index_level,
                r=m.rate,
                sigma=0.20,
                q=0.0,
                basis_pa=m.average_carry_to(tenor),
            )
            assert market.forward(tenor) == pytest.approx(settle, rel=1e-13)

    def test_the_forwards_do_not_depend_on_the_rate_the_carry_is_quoted_against(
        self,
    ) -> None:
        """Because ``r`` cancels: the board is a drift, and a drift has no rate.

        Measured at three rates a hundred basis points apart. Each reports a
        different carry -- that is the label changing -- and an identical
        forward, which is what prices.
        """
        contracts = constant_carry_panel(0.1035, REFERENCE_DAYS).contracts
        panel = FakePanel(contracts=contracts)

        carries, forwards = [], []
        for rate in (0.0, 0.02, 0.04):
            m = measure(panel, rate=rate)
            carries.append(m.far_carry)
            forwards.append(m.forward_to(3.0))

        assert carries[1] == pytest.approx(carries[0] + 0.02, abs=1e-14)
        assert carries[2] == pytest.approx(carries[0] + 0.04, abs=1e-14)
        for f in forwards[1:]:
            assert f == pytest.approx(forwards[0], rel=1e-14)

    def test_the_zero_tail_bends_the_average_down_beyond_the_board(self) -> None:
        """And by exactly the amount the unmeasured fraction implies.

        Under a zero tail the carry past the reach is zero, so the average
        carry to ``T`` is ``far_carry * reach / T`` -- which is ``far_carry``
        times :meth:`measured_fraction`. That identity is the cleanest
        statement of what the tail choice does.
        """
        carry = 0.1035
        m = measure(constant_carry_panel(carry, REFERENCE_DAYS), tail=TailCarry.ZERO)
        assert m.tail_carry == 0.0
        for T in (1.0, 2.0, 3.0):
            expected = carry * m.measured_fraction(T)
            assert m.average_carry_to(T) == pytest.approx(expected, abs=1e-13)

    def test_the_zero_tail_is_the_one_rule_that_moves_with_the_rate(self) -> None:
        """As its docstring says, and for a reason rather than by oversight.

        "No carry" is a claim about carry, which exists only relative to a
        rate, so beyond the board a zero-carry forward grows at ``r``. Inside
        the board it is still the measured drift, so the two rates agree there
        and diverge only past the reach.
        """
        contracts = constant_carry_panel(0.1035, REFERENCE_DAYS).contracts
        panel = FakePanel(contracts=contracts)
        low = measure(panel, rate=0.0, tail=TailCarry.ZERO)
        high = measure(panel, rate=0.04, tail=TailCarry.ZERO)

        assert low.forward_to(0.4) == pytest.approx(high.forward_to(0.4), rel=1e-14)
        assert high.forward_to(3.0) > low.forward_to(3.0) * 1.05

    def test_the_two_tails_agree_inside_the_board_and_diverge_outside(self) -> None:
        held = measure(constant_carry_panel(0.1035, REFERENCE_DAYS))
        zero = measure(constant_carry_panel(0.1035, REFERENCE_DAYS), tail=TailCarry.ZERO)

        assert held.average_carry_to(0.4) == pytest.approx(zero.average_carry_to(0.4), abs=1e-14)

        gaps = [held.average_carry_to(T) - zero.average_carry_to(T) for T in (1.0, 2.0, 3.0)]
        assert all(g > 0.0 for g in gaps)
        assert gaps[1] > gaps[0] and gaps[2] > gaps[1], "the assumption must grow with tenor"


class TestWhatTheTailChoiceCosts:
    def test_an_explicit_tail_needs_its_number(self) -> None:
        with pytest.raises(MarketDataError, match="pass tail_carry"):
            measure(constant_carry_panel(0.02, (37, 110)), tail=TailCarry.EXPLICIT)

    def test_a_computed_tail_refuses_a_second_source_for_the_same_number(self) -> None:
        """Two sources for one number is the failure this phase keeps finding."""
        with pytest.raises(MarketDataError, match="nobody chose"):
            measure(
                constant_carry_panel(0.02, (37, 110)),
                tail=TailCarry.HOLD_LAST,
                tail_carry=0.05,
            )

    def test_an_explicit_tail_is_used_verbatim(self) -> None:
        m = measure(
            constant_carry_panel(0.08, (37, 110)),
            tail=TailCarry.EXPLICIT,
            tail_carry=0.01,
        )
        assert m.tail_carry == 0.01
        # Far beyond the board the average must approach the tail itself.
        assert m.average_carry_to(50.0) == pytest.approx(0.01, abs=2e-3)

    def test_the_tail_is_a_required_argument(self) -> None:
        panel = constant_carry_panel(0.02, (37, 110))
        with pytest.raises(TypeError, match="tail"):
            measure_carry(panel, "IF", "000300.SH", AS_OF)  # type: ignore[call-arg]


class TestRejectingTheFrontContract:
    def test_a_contract_inside_the_minimum_tenor_is_dropped_and_counted(self) -> None:
        m = measure(constant_carry_panel(0.05, (4, 8, 73, 146)))
        assert m.rejected_near == 2
        assert len(m.maturities) == 2
        assert min(m.maturities) >= MIN_FUTURES_TENOR

    def test_dropping_them_does_not_change_the_answer_under_a_constant_carry(self) -> None:
        """So the filter removes noise rather than information.

        On real data the near contract carries a different number; under a
        constant carry it carries the same one, which is how this separates
        "the filter is discarding signal" from "the filter is discarding
        amplified noise".
        """
        carry = 0.05
        with_near = measure(constant_carry_panel(carry, (4, 73, 146)))
        without = measure(constant_carry_panel(carry, (73, 146)))
        assert with_near.average_carry_to(1.0) == pytest.approx(
            without.average_carry_to(1.0), abs=1e-14
        )

    def test_too_few_usable_contracts_is_refused_with_the_count(self) -> None:
        with pytest.raises(MarketDataError, match=f"need {MIN_CONTRACTS}"):
            measure(constant_carry_panel(0.05, (4, 8, 146)))

    def test_an_empty_board_says_so(self) -> None:
        with pytest.raises(MarketDataError, match="no IF futures settled"):
            measure(FakePanel(contracts=()))


class TestReporting:
    def test_the_measured_fraction_is_the_headline(self) -> None:
        m = measure(constant_carry_panel(0.05, (37, 177)))
        assert m.measured_fraction(0.25) == 1.0
        assert m.measured_fraction(1.0) == pytest.approx(177 / 365, abs=1e-12)
        assert m.measured_fraction(3.0) == pytest.approx(177 / 1095, abs=1e-12)

    def test_a_zero_tenor_has_no_measured_fraction(self) -> None:
        m = measure(constant_carry_panel(0.05, (37, 146)))
        with pytest.raises(MarketDataError, match="must be positive"):
            m.measured_fraction(0.0)
        with pytest.raises(MarketDataError, match="must be positive"):
            m.average_carry_to(-1.0)

    def test_the_anchor_gap_is_reported_rather_than_reconciled(self) -> None:
        """It is zero under a constant carry and not otherwise.

        Which is the point: the gap measures how badly the first listed
        *segment's* forward-forward drift stands in for the spot-to-futures
        carry, and under a constant carry there is nothing to stand in for.
        """
        flat = measure(constant_carry_panel(0.05, (37, 73, 146)))
        assert flat.anchor_gap == pytest.approx(0.0, abs=1e-14)

        steep = FakePanel(
            contracts=(
                (AS_OF + timedelta(days=37), SPOT * 0.999),
                (AS_OF + timedelta(days=73), SPOT * 0.990),
                (AS_OF + timedelta(days=177), SPOT * 0.960),
            )
        )
        assert abs(measure(steep).anchor_gap) > 1e-3

    def test_the_dict_says_what_was_measured_and_what_was_assumed(self) -> None:
        m = measure(constant_carry_panel(0.1035, (37, 177)))
        d = m.as_dict()
        assert d["tail"] == "hold_last"
        assert d["measured_fraction_1y"] == pytest.approx(177 / 365, abs=1e-4)
        assert d["rate"] == RATE
        assert "rather than a basis" in str(d["note"])
        assert "q=0, basis_pa=carry" in str(d["note"])

    def test_no_numpy_scalar_leaks_into_the_dict(self) -> None:
        """Because these land in JSON responses, where a numpy float fails."""
        m = measure(constant_carry_panel(0.05, (37, 146)))
        d = m.as_dict()
        for key, value in d.items():
            if isinstance(value, list):
                assert all(type(v) is float for v in value), key
            else:
                assert not isinstance(value, np.generic), key


# ---------------------------------------------------------------------------
# The panel facts the design rests on
# ---------------------------------------------------------------------------


@needs_panel
class TestAgainstTheRealBoard:
    @pytest.fixture(scope="class")
    def panel(self) -> WindPanel:
        return WindPanel(WIND_DIR)

    def test_the_board_reaches_only_about_half_a_year(self, panel: WindPanel) -> None:
        """The single most important fact about this data source.

        Measured over all 725 days: mean 0.4984 years, max 0.6630. So a
        one-year note is half assumption and a three-year note is 84%
        assumption, and no amount of care in the engine changes that.
        """
        reaches = []
        for as_of in panel.trading_days:
            listed = panel.futures("IF", as_of)
            if listed:
                reaches.append((listed[-1][0] - as_of).days / 365.0)
        assert len(reaches) > 700
        assert max(reaches) < 0.70, f"the board now reaches {max(reaches):.3f}y"
        assert 0.45 < float(np.mean(reaches)) < 0.55

    def test_the_front_contract_is_an_order_of_magnitude_noisier(self, panel: WindPanel) -> None:
        """Which is why MIN_FUTURES_TENOR exists.

        Front-contract drift inside 0.03 years has an sd of 0.473 and reaches
        -5.63; the spot-to-farthest drift has an sd of 0.026. The assertion is
        on the ratio rather than on either number, because the ratio is what
        justifies the filter. The rate offset is constant and cancels from
        both, so this measures the drift directly.
        """
        near, far = [], []
        for as_of in panel.trading_days:
            listed = panel.futures("IF", as_of)
            if len(listed) < 2:
                continue
            spot = panel.index_close("000300.SH", as_of)
            t_near = (listed[0][0] - as_of).days / 365.0
            t_far = (listed[-1][0] - as_of).days / 365.0
            if 0.0 < t_near <= 0.03:
                near.append(float(np.log(listed[0][1] / spot)) / t_near)
            if t_far > 0.2:
                far.append(float(np.log(listed[-1][1] / spot)) / t_far)
        assert len(near) > 200 and len(far) > 600
        assert float(np.std(near)) / float(np.std(far)) > 10.0
        assert min(near) < -1.0, "a drift beyond -100% a year is the artefact in question"

    def test_carry_is_negative_on_about_one_day_in_ten(self, panel: WindPanel) -> None:
        """Which is what makes the basis_pa >= 0 validation wrong.

        8.3% of days for IF and 10.3% for IH at the panel's own rate. Clamping
        those to zero would be editing the data to fit the assumption.

        The underlying market fact is larger than the carry statistic: IH
        futures trade *above* the index on 40.9% of panel days. It is only the
        rate that lifts most of those into a positive carry.
        """
        negatives, contango = {}, {}
        for root, code in (("IF", "000300.SH"), ("IH", "000016.SH"), ("IM", "000852.SH")):
            carries, drifts = [], []
            for as_of in panel.trading_days:
                listed = panel.futures(root, as_of)
                if len(listed) < 2:
                    continue
                tenor = (listed[-1][0] - as_of).days / 365.0
                if tenor <= 0.2:
                    continue
                spot = panel.index_close(code, as_of)
                drift = float(np.log(listed[-1][1] / spot)) / tenor
                drifts.append(drift)
                carries.append(RATE - drift)
            negatives[root] = float(np.mean(np.array(carries) < 0.0))
            contango[root] = float(np.mean(np.array(drifts) > 0.0))

        assert 0.03 < negatives["IF"] < 0.15, f"IF carry negative on {negatives['IF']:.1%}"
        assert 0.03 < negatives["IH"] < 0.18, f"IH carry negative on {negatives['IH']:.1%}"
        assert negatives["IM"] == 0.0, "the CSI 1000 was discounted on every panel day"
        assert contango["IH"] > 0.35, f"IH in contango on only {contango['IH']:.1%} of days"

    def test_the_three_roots_rank_as_the_market_view_says(self, panel: WindPanel) -> None:
        """CSI 1000 discounted hardest, SSE 50 least -- which is the whole
        reason the deepest-discount index is the one snowballs are written on.
        """
        measured = {
            root: measure_carry(panel, root, index_code, AS_OF, tail=TailCarry.HOLD_LAST, rate=RATE)
            for root, index_code in (
                ("IF", "000300.SH"),
                ("IH", "000016.SH"),
                ("IM", "000852.SH"),
            )
        }
        carries = {root: m.far_carry for root, m in measured.items()}
        assert carries["IM"] > carries["IF"] > carries["IH"]
        assert carries["IM"] == pytest.approx(0.1015, abs=5e-4)
        assert carries["IF"] == pytest.approx(0.0340, abs=5e-4)
        assert carries["IH"] == pytest.approx(0.0183, abs=5e-4)
        # The SSE 50's futures were above the index on this date; only the rate
        # makes its carry positive.
        assert measured["IH"].far_drift > 0.0

    def test_a_measurement_off_the_real_board_is_self_consistent(self, panel: WindPanel) -> None:
        """The identities that must hold whatever the data says."""
        m = measure_carry(panel, "IM", "000852.SH", AS_OF, tail=TailCarry.HOLD_LAST, rate=RATE)
        assert m.rejected_near == 0
        assert m.average_carry_to(m.observed_reach) == pytest.approx(m.far_carry, abs=1e-13)
        assert m.forward_to(m.observed_reach) == pytest.approx(m.forwards[-1], rel=1e-14)
        # Holding the far carry flat means the average cannot drift away from
        # it -- but the measured near end is not at the far carry, so the
        # approach is gradual rather than exact.
        assert m.average_carry_to(3.0) == pytest.approx(m.far_carry, abs=3e-3)
        assert m.average_carry_to(10.0) == pytest.approx(m.far_carry, abs=1e-3)

    def test_the_real_boards_term_structure_is_not_flat(self, panel: WindPanel) -> None:
        """The reason the shape is kept rather than collapsed to a level."""
        m = measure_carry(panel, "IM", "000852.SH", AS_OF, tail=TailCarry.HOLD_LAST, rate=RATE)
        assert abs(m.term_structure_gap) > 1e-3
        assert abs(m.anchor_gap) > 1e-4, "a real board back-extrapolates away from its index"

    def test_the_last_panel_day_has_no_futures_and_says_so(self, panel: WindPanel) -> None:
        """The option quotes run one day past the futures quotes."""
        with pytest.raises(MarketDataError, match="no IF futures settled"):
            measure_carry(panel, "IF", "000300.SH", date(2026, 6, 29), tail=TailCarry.HOLD_LAST)


# ---------------------------------------------------------------------------
# The endpoint
# ---------------------------------------------------------------------------


class TestTheCarryEndpoint:
    """The measurement, exposed. It had no caller outside this file.

    Two behaviours matter more than the numbers. ``tail`` is required, because
    it governs 52% of a one-year note's carry and 84% of a three-year note's --
    and the test below shows that choice moving the three-year carry by a
    factor of six. And a checkout without the institutional panel gets a 400
    naming the environment variable rather than a 500, because an absent data
    file is a configuration fact, not a solver failure.
    """

    PAYLOAD = {"underlying": "MO", "as_of": "2024-12-25", "tail": "hold_last", "rate": 0.02}

    @pytest.fixture
    def client(self, monkeypatch):  # noqa: ANN001, ANN201
        import app

        app.app.config.update(TESTING=True)
        # The route reads its panel from the module-level cache, which is what
        # that cache is for: 34 MB of parquet parsed once per process. Setting
        # it directly is also how these tests avoid depending on the
        # environment variable a deployment would use.
        if WIND_DIR is not None:
            monkeypatch.setattr(app, "_WIND_PANEL", WindPanel(WIND_DIR))
        return app.app.test_client()

    @pytest.fixture
    def bare_client(self, monkeypatch):  # noqa: ANN001, ANN201
        """A client with no panel, which is the normal checkout."""
        import app

        app.app.config.update(TESTING=True)
        monkeypatch.setattr(app, "_WIND_PANEL", None)
        monkeypatch.delenv("OTCPRICER_WIND_DIR", raising=False)

        from otcpricer.core.config import get_settings

        get_settings.cache_clear()
        monkeypatch.setattr(
            "otcpricer.core.config.get_settings",
            lambda: _settings_without_wind(get_settings),
        )
        return app.app.test_client()

    # -- the required choice ------------------------------------------------

    def test_the_tail_is_required(self, client) -> None:  # noqa: ANN001
        """No default, and the refusal says why rather than just refusing."""
        response = client.post("/api/carry", json={"underlying": "MO", "as_of": "2024-12-25"})
        assert response.status_code == 400
        body = response.get_json()
        assert "tail is required" in body["hint"]
        assert "84%" in body["hint"]

    def test_an_unknown_tail_is_refused(self, client) -> None:  # noqa: ANN001
        response = client.post("/api/carry", json={**self.PAYLOAD, "tail": "guess"})
        assert response.status_code == 400

    def test_two_sources_for_the_tail_are_refused(self, client) -> None:  # noqa: ANN001
        """The defect pattern this phase keeps finding, caught at the edge."""
        response = client.post("/api/carry", json={**self.PAYLOAD, "tail_carry": 0.05})
        assert response.status_code == 400
        assert "nobody chose" in response.get_json()["error"]

    # -- degradation --------------------------------------------------------

    def test_without_the_panel_it_names_the_variable(self, bare_client) -> None:  # noqa: ANN001
        """A missing institutional data file is configuration, not a crash.

        This is the path a fresh checkout takes, so it is the one most likely
        to be hit by someone who has never seen the panel.
        """
        response = bare_client.post("/api/carry", json=self.PAYLOAD)
        assert response.status_code == 400
        assert "OTCPRICER_WIND_DIR" in response.get_json()["error"]

    @needs_panel
    def test_a_date_the_board_does_not_cover_says_so(self, client) -> None:  # noqa: ANN001
        """The option quotes run one day past the futures quotes.

        Needs the board: without it the route reports the missing panel first,
        which is the correct answer to a different question.
        """
        response = client.post("/api/carry", json={**self.PAYLOAD, "as_of": "2026-06-29"})
        assert response.status_code == 400
        assert "no IM futures settled" in response.get_json()["error"]

    def test_a_malformed_date_is_refused_by_shape(self, client) -> None:  # noqa: ANN001
        response = client.post("/api/carry", json={**self.PAYLOAD, "as_of": "25/12/2024"})
        assert response.status_code == 400

    def test_an_underlying_without_a_future_is_refused(self, client) -> None:  # noqa: ANN001
        """The ETF options have no index future sharing their expiry calendar,
        so the interpolation-free measurement is simply unavailable for them."""
        response = client.post("/api/carry", json={**self.PAYLOAD, "underlying": "510050"})
        assert response.status_code == 400
        assert "no index future" in response.get_json()["error"]

    # -- the measurement ----------------------------------------------------

    @needs_panel
    def test_the_three_roots_come_back_as_measured(self, client) -> None:  # noqa: ANN001
        """And the futures root comes from the registry, not the request.

        MO -> IM, IO -> IF, HO -> IH, with the settling index taken from the
        same entry, so the pairing has one source.
        """
        expected = {"MO": ("IM", 0.1015), "IO": ("IF", 0.0340), "HO": ("IH", 0.0183)}
        for code, (root, far_carry) in expected.items():
            body = client.post("/api/carry", json={**self.PAYLOAD, "underlying": code}).get_json()
            assert body["root"] == root
            assert body["far_carry"] == pytest.approx(far_carry, abs=5e-4), code

    @needs_panel
    def test_the_measured_fraction_leads(self, client) -> None:  # noqa: ANN001
        """48.5% for a one-year note, 16.2% for a three-year one."""
        body = client.post("/api/carry", json=self.PAYLOAD).get_json()
        assert body["measured_fraction_1y"] == pytest.approx(0.4849, abs=1e-3)
        assert body["measured_fraction_3y"] == pytest.approx(0.1616, abs=1e-3)
        assert body["observed_reach"] == pytest.approx(0.4849, abs=1e-3)

    @needs_panel
    def test_the_tail_choice_moves_the_three_year_carry_six_fold(self, client) -> None:  # noqa: ANN001
        """Which is the measurement that justifies making it required.

        Measured on the CSI 1000 board: 10.15% a year under hold_last against
        1.64% under zero. Not a refinement -- a different trade.
        """
        held = client.post("/api/carry", json=self.PAYLOAD).get_json()
        zero = client.post("/api/carry", json={**self.PAYLOAD, "tail": "zero"}).get_json()

        assert held["average_carry_3y"] == pytest.approx(0.10154, abs=5e-4)
        assert zero["average_carry_3y"] == pytest.approx(0.01641, abs=5e-4)
        assert held["average_carry_3y"] / zero["average_carry_3y"] > 5.0

        # Inside the measured range they agree, because the tail is beyond it.
        assert held["far_carry"] == pytest.approx(zero["far_carry"], abs=1e-12)

    @needs_panel
    def test_the_sse_50_is_at_a_premium_and_the_payload_shows_it(self, client) -> None:  # noqa: ANN001
        """The state core/market.py used to refuse outright.

        Its carry is positive only because the rate lifts it: the drift the
        board actually shows is positive, meaning the future trades above the
        index.
        """
        body = client.post("/api/carry", json={**self.PAYLOAD, "underlying": "HO"}).get_json()
        assert body["contract_drifts"][-1] > 0.0
        assert body["far_carry"] > 0.0
        assert body["far_carry"] == pytest.approx(0.02 - body["contract_drifts"][-1], abs=1e-8)

    @needs_panel
    def test_it_says_what_to_feed_back_into_a_pricing_route(self, client) -> None:  # noqa: ANN001
        """q=0 and basis_pa=carry, with the same rate, because the engines read
        only the sum and the rate cancels."""
        body = client.post("/api/carry", json=self.PAYLOAD).get_json()
        feed = body["feed_as"]
        assert feed["q"] == 0.0
        assert feed["r"] == 0.02
        assert feed["basis_pa"] == pytest.approx(body["average_carry_1y"], abs=1e-12)

    @needs_panel
    def test_the_payload_says_the_board_shows_a_drift(self, client) -> None:  # noqa: ANN001
        body = client.post("/api/carry", json=self.PAYLOAD).get_json()
        assert "rather than a basis" in body["note"]
        assert len(body["contract_drifts"]) == len(body["contract_carries"])
        for drift, carry in zip(body["contract_drifts"], body["contract_carries"], strict=True):
            assert drift + carry == pytest.approx(0.02, abs=1e-8)


def _settings_without_wind(loader):  # noqa: ANN001, ANN202
    """A settings object whose wind_dir is empty, for the degradation test."""
    from dataclasses import replace as _replace

    settings = loader()
    try:
        return _replace(settings, data=_replace(settings.data, wind_dir=""))
    except TypeError:  # pydantic model rather than a dataclass
        return settings.model_copy(
            update={"data": settings.data.model_copy(update={"wind_dir": ""})}
        )
