"""Tests for the futures-versus-parity forward check.

The headline assertion here is a negative one: that the forward gap cannot be
explained by the discount rate. That is the opposite of what this check was
designed to show, so it is pinned deliberately -- with the leverage measurement
that forces the conclusion -- rather than left as a remark in a docstring.
"""

from __future__ import annotations

from datetime import date

import pytest

from otcpricer.marketdata.contracts import resolve_underlying
from otcpricer.marketdata.rates import rate_curve_from_panel
from otcpricer.validation.futures_check import RATE_BUMP, ForwardCheck, check_forwards
from tests.test_wind import WIND_DIR, needs_panel

A_DATE = date(2024, 12, 25)


def _check(**overrides: object) -> ForwardCheck:
    """A row with plausible defaults, for the pure-arithmetic tests."""
    fields: dict[str, object] = {
        "expiry": date(2025, 3, 21),
        "expiry_tag": "2503",
        "year_fraction": 0.236,
        "n_paired": 29,
        "forward_parity": 3996.07,
        "forward_futures": 3984.20,
        "rate_used": 0.0089,
        "parity_residual_relative": 0.000926,
        "forward_sensitivity": 0.2735,
    }
    fields.update(overrides)
    return ForwardCheck(**fields)  # type: ignore[arg-type]


class TestArithmetic:
    def test_the_gap_is_parity_minus_futures(self) -> None:
        row = _check()
        assert row.gap_points == pytest.approx(11.87)
        assert row.gap_relative == pytest.approx(11.87 / 3984.20)

    def test_a_gap_below_the_parity_residual_is_not_signal(self) -> None:
        """When the quotes behind one forward are less precise than the gap."""
        assert _check(
            forward_parity=3984.5, parity_residual_relative=0.01
        ).explained_by_parity_noise
        assert not _check().explained_by_parity_noise

    def test_the_rate_needed_to_close_the_gap_is_reported_however_absurd(self) -> None:
        """Kept because the answer being absurd *is* the finding.

        11.87 points of gap against 0.2735 points per 100 bp needs 4,340 bp.
        """
        needed = _check().rate_change_to_close
        assert needed == pytest.approx(11.87 / 0.2735 * RATE_BUMP)
        assert needed * 1e4 == pytest.approx(4340, abs=10)
        assert not _check().rate_can_explain_the_gap

    def test_an_opposite_signed_sensitivity_points_the_wrong_way(self) -> None:
        """Which is the sharpest form of the non-identification.

        On the two short expiries the parity forward *falls* as the rate rises,
        while the gap is positive -- so no rate closes it, in either direction.
        """
        row = _check(forward_sensitivity=-0.0132)
        assert row.gap_points > 0.0
        assert row.rate_change_to_close < 0.0
        assert not row.rate_can_explain_the_gap

    def test_a_gap_a_plausible_rate_could_explain_is_admitted(self) -> None:
        """The predicate is not rigged to always answer no.

        A tenth of a point of gap against the same sensitivity needs 37 bp,
        which is well inside the 200 bp the predicate allows.
        """
        row = _check(forward_parity=3984.30)
        assert row.rate_can_explain_the_gap
        assert abs(row.rate_change_to_close) < 0.02

    def test_an_unresponsive_forward_gives_no_answer_rather_than_infinity(self) -> None:
        row = _check(forward_sensitivity=0.0)
        assert row.rate_change_to_close != row.rate_change_to_close  # nan
        assert not row.rate_can_explain_the_gap

    def test_serialises_with_the_caveat_attached(self) -> None:
        payload = _check().as_dict()
        assert payload["rate_can_explain_the_gap"] is False
        assert payload["rate_change_to_close_bp"] == pytest.approx(4340, abs=10)


@needs_panel
class TestAgainstThePanel:
    @pytest.fixture(scope="class")
    def panel(self):  # type: ignore[no-untyped-def]
        from otcpricer.marketdata.wind import WindPanel

        return WindPanel(WIND_DIR)

    @pytest.fixture(scope="class")
    def reports(self, panel):  # type: ignore[no-untyped-def]
        from otcpricer.marketdata.wind import WindPanelSource

        out = {}
        for day in (A_DATE, date(2025, 6, 10)):
            curve = rate_curve_from_panel(panel, day)
            for root in ("IO", "HO", "MO"):
                chain = WindPanelSource(panel=panel, as_of=day).fetch(root)
                futures = panel.futures(resolve_underlying(root).futures_root, day)
                out[(day, root)] = check_forwards(chain, futures, rate_curve=curve)
        return out

    def test_the_rate_explains_no_gap_anywhere(self, reports) -> None:  # type: ignore[no-untyped-def]
        """The central finding, and the reason the government curve stays an input.

        Closing the measured gaps with the rate would take 757 to 35,699 basis
        points. Phase 2a found that options alone cannot identify the rate;
        this extends that to options plus futures, because the rate's leverage
        on the parity forward is an order of magnitude below the gap.
        """
        for key, report in reports.items():
            assert not report.rate_explains_any_gap, key
            for row in report.checks:
                assert abs(row.rate_change_to_close) > 0.02, (key, row.expiry_tag)

    def test_the_rate_barely_moves_the_forward_at_all(self, reports) -> None:  # type: ignore[no-untyped-def]
        """The measurement the conclusion rests on.

        100 bp of rate moves the parity forward by at most 0.042% of forward
        across both dates and all three roots -- the worst case being MO's
        longest checked expiry. That is an order of magnitude below the
        0.08-0.46% gaps, and consistent with forward.py's independently
        measured 0.003% on the akshare captures.
        """
        worst = 0.0
        for report in reports.values():
            for row in report.checks:
                relative = abs(row.forward_sensitivity) / row.forward_futures
                worst = max(worst, relative)
        assert worst < 5.0e-4

    def test_the_sensitivity_sign_is_not_even_stable(self, reports) -> None:  # type: ignore[no-untyped-def]
        """Which rules the rate out structurally, not just by magnitude.

        ``dF/dr = T * mean(C - P)/D``, so the sign follows whether the traded
        strikes sit above or below the forward. Both signs occur in this data,
        so there is no rate adjustment that could move every expiry the same
        way.
        """
        signs = {
            row.forward_sensitivity > 0.0 for report in reports.values() for row in report.checks
        }
        assert signs == {True, False}

    def test_the_gaps_are_small_but_real(self, reports) -> None:  # type: ignore[no-untyped-def]
        """0.08% to 0.46% of forward, which lands on the moneyness coordinate."""
        worst = max(r.worst_gap_relative for r in reports.values())
        assert 0.001 < worst < 0.006

        # And on the better-quoted date they all clear their own parity noise,
        # so the check has resolving power rather than measuring its own error.
        for root in ("IO", "HO", "MO"):
            assert reports[(A_DATE, root)].all_gaps_exceed_parity_noise

    def test_the_gap_grows_with_tenor_on_most_dates(self, reports) -> None:  # type: ignore[no-untyped-def]
        """The pattern that made a rate story look plausible.

        True on 5 of 6 root-date combinations. Recorded as an observation --
        the rate is ruled out above, and separating a liquidity explanation
        from a call-versus-put settlement bias needs data this panel lacks.
        """
        growing = sum(1 for r in reports.values() if r.gap_grows_with_tenor)
        assert growing == 5

    def test_two_of_six_expiries_are_never_checked(self, reports) -> None:  # type: ignore[no-untyped-def]
        """Only four futures list at a time, and the report has to say so.

        The unchecked ones are the longest -- the tenors that matter most for a
        one-to-three-year snowball.
        """
        report = reports[(A_DATE, "IO")]
        assert report.n_option_expiries == 6
        assert len(report.checks) == 4
        assert report.coverage == pytest.approx(4 / 6)
        assert report.unchecked_expiries == ("2509", "2512")

        checked = max(c.year_fraction for c in report.checks)
        assert checked < 0.5  # the half-year point is the furthest confirmed

    def test_futures_are_matched_on_the_exact_expiry_only(self, panel) -> None:  # type: ignore[no-untyped-def]
        """Never a nearest neighbour, which would reintroduce interpolation.

        The whole value of this check is that the two instruments share an
        expiry date exactly, so a near-miss match would quietly compare
        forwards to different maturities.
        """
        from otcpricer.marketdata.wind import WindPanelSource

        chain = WindPanelSource(panel=panel, as_of=A_DATE).fetch("IO")
        report = check_forwards(chain, panel.futures("IF", A_DATE))

        option_expiries = {s.expiry for s in chain.slices}
        futures_expiries = dict(panel.futures("IF", A_DATE))
        for row in report.checks:
            assert row.expiry in option_expiries
            assert row.forward_futures == pytest.approx(futures_expiries[row.expiry])

    def test_no_futures_at_all_leaves_everything_unchecked(self, panel) -> None:  # type: ignore[no-untyped-def]
        from otcpricer.marketdata.wind import WindPanelSource

        chain = WindPanelSource(panel=panel, as_of=A_DATE).fetch("IO")
        report = check_forwards(chain, ())

        assert report.checks == ()
        assert report.coverage == 0.0
        assert len(report.unchecked_expiries) == 6
        assert report.worst_gap_relative != report.worst_gap_relative  # nan
        assert not report.rate_explains_any_gap
        assert not report.all_gaps_exceed_parity_noise
