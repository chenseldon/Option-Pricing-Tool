"""Tests for the local volatility validation report.

Two things are being checked. The report must say what it is required to say
-- and in the right order, because the ordering is the argument: the
extrapolation share leads, since it is the one exposure no amount of grid
refinement touches. And the round trip must actually be run against the real
surfaces, because a report that renders only a description is not a
validation.
"""

from __future__ import annotations

import pytest

from otcpricer.engines.pde import PDEConfig
from otcpricer.models.local_vol import local_vol_from_fitted
from otcpricer.models.surface import fit_surface
from otcpricer.reports.local_vol_report import local_vol_report, round_trip
from tests.test_marketdata import SNAPSHOT_DIR

FAST = PDEConfig(space_nodes=600, steps_per_year=800)


@pytest.fixture
def pieces() -> tuple[object, object]:
    if not SNAPSHOT_DIR.exists():
        pytest.skip("no snapshots committed")
    fitted = fit_surface("510300", source_kind="snapshot")
    return fitted, local_vol_from_fitted(fitted, tail_drift=0.02)


class TestReportContent:
    def test_extrapolation_leads(self, pieces: tuple) -> None:
        """The ordering is the argument, so it is pinned.

        86% of a three-year horizon sits past the last quoted expiry, and no
        grid refinement touches that. Burying it below the numerics would
        misrepresent which assumption is doing the work.
        """
        fitted, local_vol = pieces
        text = local_vol_report(fitted, local_vol, horizon=3.0)
        headings = [line for line in text.splitlines() if line.isupper() and line.strip()]

        assert headings[1].startswith("EXTRAPOLATION")
        assert text.index("EXTRAPOLATION") < text.index("NUMERICAL DIAGNOSTICS")
        assert "86" in text

    def test_carries_every_section(self, pieces: tuple) -> None:
        fitted, local_vol = pieces
        text = local_vol_report(fitted, local_vol, horizon=3.0)
        for heading in (
            "EXTRAPOLATION -- READ THIS FIRST",
            "FORWARD CURVE",
            "DUPIRE ROUND TRIP -- THE EVIDENCE",
            "LOCAL VOLATILITY AGAINST THE MONEY",
            "NUMERICAL DIAGNOSTICS",
        ):
            assert heading in text, heading

    def test_names_the_forward_kink(self, pieces: tuple) -> None:
        """The acknowledged cost of the tail convention, quantified not hidden."""
        fitted, local_vol = pieces
        text = local_vol_report(fitted, local_vol, horizon=3.0)
        assert "kink at the handover" in text
        assert f"{local_vol.forwards.kink:+.5f}" in text

    def test_says_when_the_round_trip_was_not_run(self, pieces: tuple) -> None:
        """A description must not be mistaken for a validation."""
        fitted, local_vol = pieces
        assert "Not run." in local_vol_report(fitted, local_vol, horizon=3.0)

    def test_a_horizon_inside_the_quotes_says_so(self, pieces: tuple) -> None:
        fitted, local_vol = pieces
        text = local_vol_report(fitted, local_vol, horizon=0.3)
        assert "beyond the quotes    0.0%" in text
        assert "inside the quoted range" in text

    def test_lines_stay_terminal_width(self, pieces: tuple) -> None:
        fitted, local_vol = pieces
        text = local_vol_report(fitted, local_vol, horizon=3.0)
        overlong = [line for line in text.splitlines() if len(line) > 110]
        assert not overlong, overlong


@pytest.mark.slow
class TestRoundTrip:
    @pytest.mark.parametrize("code", ["510050", "510300"])
    def test_runs_against_the_real_surfaces(self, code: str) -> None:
        """The evidence section, actually produced."""
        if not SNAPSHOT_DIR.exists():
            pytest.skip("no snapshots committed")
        fitted = fit_surface(code, source_kind="snapshot")
        local_vol = local_vol_from_fitted(fitted, tail_drift=0.02)

        rows = round_trip(fitted, local_vol, config=FAST)

        assert len(rows) == 5 * len(fitted.surface_input.usable_slices)
        for row in rows:
            assert 0.01 < row.recovered_vol < 1.5
            assert row.price > 0.0
        # Loose here on purpose: the budget lives in test_european, measured
        # at a production grid. This only asserts the section is not nonsense.
        assert max(abs(row.error_vol_points) for row in rows) < 2.0

    def test_the_report_states_the_verdict_against_the_fit_error(self) -> None:
        if not SNAPSHOT_DIR.exists():
            pytest.skip("no snapshots committed")
        fitted = fit_surface("510300", source_kind="snapshot")
        local_vol = local_vol_from_fitted(fitted, tail_drift=0.02)

        text = local_vol_report(
            fitted,
            local_vol,
            horizon=3.0,
            round_trip_rows=round_trip(fitted, local_vol, config=FAST),
        )
        assert "worst error" in text
        assert "calibration rms" in text
        assert "PASS" in text

    def test_rows_serialise(self) -> None:
        if not SNAPSHOT_DIR.exists():
            pytest.skip("no snapshots committed")
        fitted = fit_surface("510300", source_kind="snapshot")
        local_vol = local_vol_from_fitted(fitted, tail_drift=0.02)
        payload = round_trip(fitted, local_vol, config=FAST, moneyness=(1.0,))[0].as_dict()
        for key in ("maturity", "moneyness", "target_vol", "recovered_vol", "error_vol_points"):
            assert key in payload
