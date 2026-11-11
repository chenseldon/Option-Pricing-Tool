"""Tests for the published surface mark.

The thing being tested is a rule about process, not a number: one surface is
blessed, it has to pass stated checks to get blessed, and everything prices
off that one. So most of these tests are about the refusals -- a gate that
cannot stop a publication is decoration, and this project has already found
one comment claiming validation on a formula that was wrong for every down
barrier.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from otcpricer.core.exceptions import MarketDataError
from otcpricer.marketdata.marks import (
    MAX_ATM_VOL_DRIFT,
    MAX_RMS_VOL_POINTS,
    GateResult,
    SurfaceMark,
    evaluate_gates,
    fingerprint,
    load_mark,
    publish_mark,
)
from otcpricer.models.surface import fit_surface


@pytest.fixture(scope="module")
def fitted():  # noqa: ANN201
    """The committed 510300 snapshot, fitted once for the whole module."""
    return fit_surface("510300", source_kind="snapshot", rate=0.02)


class TestTheGates:
    def test_the_committed_snapshot_passes_all_of_them(self, fitted) -> None:  # noqa: ANN001
        """It has to, or nothing in this project could be published.

        Measured: fit 0.654 vol points against a 2.0 limit, butterfly margin
        0.263, calendar 0.034, zero violations in 9,660 checks.
        """
        gates = evaluate_gates(fitted)
        failed = [g.name for g in gates if not g.passed]
        assert failed == [], f"gates failed on the reference chain: {failed}"

    def test_every_gate_carries_the_number_it_decided_on(self, fitted) -> None:  # noqa: ANN001
        """A refusal that does not say how far off it was leaves the caller
        guessing whether the data is slightly stale or completely wrong."""
        for gate in evaluate_gates(fitted):
            assert gate.note, f"{gate.name} has no explanation"
            if gate.name != "drift":  # drift has no value on a first mark
                assert gate.value is not None, f"{gate.name} reports no value"
                assert gate.threshold is not None, f"{gate.name} has no threshold"

    def test_all_gates_run_rather_than_stopping_at_the_first_failure(self, fitted) -> None:  # noqa: ANN001
        """A reader deciding whether to override needs to see whether one
        thing is marginal or everything is wrong."""
        names = [g.name for g in evaluate_gates(fitted)]
        assert names == ["fit_quality", "butterfly", "calendar", "arbitrage_free", "drift"]

    def test_the_fit_gate_is_set_from_this_project_s_own_data(self) -> None:
        """Not from the ~5bp a listed-vanilla desk would use.

        The committed chains fit to 65bp and 59bp. A 5bp gate would reject
        every surface this project can build, and a gate nothing passes is a
        gate that gets switched off rather than a check.
        """
        assert MAX_RMS_VOL_POINTS > 0.654, "the reference chain could not be published"
        assert MAX_RMS_VOL_POINTS < 5.0, "loose enough to admit a degenerate chain"


class TestTheDriftGate:
    def test_a_first_mark_has_nothing_to_compare_against(self, fitted) -> None:  # noqa: ANN001
        drift = next(g for g in evaluate_gates(fitted, None) if g.name == "drift")
        assert drift.passed
        assert drift.value is None
        assert "No previous mark" in drift.note

    def test_an_identical_surface_drifts_by_exactly_zero(self, fitted, tmp_path: Path) -> None:  # noqa: ANN001
        """Not 3e-09.

        The mark stores at-the-money volatilities rounded to eight places, so
        comparing a stored value against a freshly computed one carried the
        rounding. It never mattered against a 0.10 threshold, but a drift of
        "zero" printing as 3.3e-09 is the kind of thing that later costs
        somebody an hour.
        """
        publish_mark(fitted, source_kind="snapshot", snapshot=None, rate=0.02, directory=tmp_path)
        previous = load_mark("510300", directory=tmp_path, fitted=fitted)
        drift = next(g for g in evaluate_gates(fitted, previous) if g.name == "drift")
        assert drift.value == 0.0

    def test_an_implausible_jump_is_refused(self, fitted, tmp_path: Path) -> None:  # noqa: ANN001
        """Ten points of at-the-money volatility in a day is a data error."""
        publish_mark(fitted, source_kind="snapshot", snapshot=None, rate=0.02, directory=tmp_path)
        previous = load_mark("510300", directory=tmp_path, fitted=fitted)
        assert previous is not None
        moved = replace(
            previous,
            atm_vols=[v + MAX_ATM_VOL_DRIFT * 2 for v in previous.atm_vols],
        )
        drift = next(g for g in evaluate_gates(fitted, moved) if g.name == "drift")
        assert not drift.passed
        assert drift.value == pytest.approx(MAX_ATM_VOL_DRIFT * 2, rel=1e-9)

    def test_it_compares_only_the_shared_expiries(self, fitted, tmp_path: Path) -> None:  # noqa: ANN001
        """A chain that gained or lost an expiry overnight is normal -- a
        monthly contract rolls off. Comparing by position past the shorter of
        the two would read a roll as a drift."""
        publish_mark(fitted, source_kind="snapshot", snapshot=None, rate=0.02, directory=tmp_path)
        previous = load_mark("510300", directory=tmp_path, fitted=fitted)
        assert previous is not None
        shorter = replace(previous, atm_vols=previous.atm_vols[:1])
        drift = next(g for g in evaluate_gates(fitted, shorter) if g.name == "drift")
        assert drift.passed
        assert "over 1 shared expiries" in drift.note


class TestPublishing:
    def test_it_writes_a_file_named_for_the_underlying(self, fitted, tmp_path: Path) -> None:  # noqa: ANN001
        publish_mark(fitted, source_kind="snapshot", snapshot=None, rate=0.02, directory=tmp_path)
        assert (tmp_path / "510300.json").exists()

    def test_the_file_is_readable_json_a_person_can_check(self, fitted, tmp_path: Path) -> None:  # noqa: ANN001
        """Indented and not minified, because the point of publishing to disk
        rather than to a process variable is that someone can look."""
        publish_mark(fitted, source_kind="snapshot", snapshot=None, rate=0.02, directory=tmp_path)
        raw = json.loads((tmp_path / "510300.json").read_text(encoding="utf-8"))
        assert raw["underlying"] == "510300"
        assert raw["as_of"] == fitted.as_of
        assert len(raw["gates"]) == 5
        assert raw["forced"] is False

    def test_a_failing_gate_refuses_the_publication(self, fitted, tmp_path: Path) -> None:  # noqa: ANN001
        """The check that makes the rest of this module worth anything.

        Simulated by publishing against a previous mark whose at-the-money
        volatilities are implausibly far away, which is the one gate that can
        be failed without corrupting the fit itself.
        """
        prior = SurfaceMark(
            underlying="510300",
            as_of="2026-10-26",
            source="snapshot:fake",
            source_kind="snapshot",
            snapshot=None,
            csv_token=None,
            rate=0.02,
            published_at="2026-10-26T00:00:00+00:00",
            fingerprint="0" * 16,
            atm_vols=[v + 0.5 for v in _atm(fitted)],
            maturities=[float(t) for t in fitted.calibration.params.maturities],
            rms_vol_points=0.6,
            gates=[],
        )
        (tmp_path / "510300.json").write_text(json.dumps(prior.as_dict()), encoding="utf-8")
        with pytest.raises(MarketDataError, match="refusing to publish"):
            publish_mark(
                fitted,
                source_kind="snapshot",
                snapshot=None,
                rate=0.02,
                directory=tmp_path,
            )

    def test_the_refusal_names_the_gate_and_the_distance(self, fitted, tmp_path: Path) -> None:  # noqa: ANN001
        prior_path = tmp_path / "510300.json"
        prior_path.write_text(
            json.dumps(
                {
                    "underlying": "510300",
                    "as_of": "2026-10-26",
                    "source": "snapshot:fake",
                    "rate": 0.02,
                    "published_at": "2026-10-26T00:00:00+00:00",
                    "fingerprint": "0" * 16,
                    "atm_vols": [v + 0.5 for v in _atm(fitted)],
                    "maturities": [],
                    "rms_vol_points": 0.6,
                    "gates": [],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(MarketDataError) as excinfo:
            publish_mark(
                fitted,
                source_kind="snapshot",
                snapshot=None,
                rate=0.02,
                directory=tmp_path,
            )
        message = str(excinfo.value)
        assert "drift" in message
        assert "force=True" in message, "the refusal does not say how to override"

    def test_force_publishes_anyway_and_records_that_it_did(self, fitted, tmp_path: Path) -> None:  # noqa: ANN001
        """A desk sometimes has to mark a surface it knows is ugly -- a
        half-session, a single expiry gone wide. The alternative to an
        override is editing the thresholds, which removes the check for every
        later day. So the override exists and is recorded, and the next reader
        can see that a gate failed and someone decided anyway.
        """
        (tmp_path / "510300.json").write_text(
            json.dumps(
                {
                    "underlying": "510300",
                    "as_of": "2026-10-26",
                    "source": "snapshot:fake",
                    "rate": 0.02,
                    "published_at": "2026-10-26T00:00:00+00:00",
                    "fingerprint": "0" * 16,
                    "atm_vols": [v + 0.5 for v in _atm(fitted)],
                    "maturities": [],
                    "rms_vol_points": 0.6,
                    "gates": [],
                }
            ),
            encoding="utf-8",
        )
        mark = publish_mark(
            fitted,
            source_kind="snapshot",
            snapshot=None,
            rate=0.02,
            directory=tmp_path,
            force=True,
        )
        assert any(not g.passed for g in mark.gates), "nothing was overridden"
        raw = json.loads((tmp_path / "510300.json").read_text(encoding="utf-8"))
        assert raw["forced"] is True


class TestReadingBack:
    def test_no_mark_reads_as_none_rather_than_raising(self, tmp_path: Path) -> None:
        """An unmarked underlying is an ordinary state, not an error: it is
        every underlying before the first publish of the day."""
        assert load_mark("510050", directory=tmp_path) is None

    def test_a_round_trip_preserves_the_fingerprint(self, fitted, tmp_path: Path) -> None:  # noqa: ANN001
        published = publish_mark(
            fitted, source_kind="snapshot", snapshot=None, rate=0.02, directory=tmp_path
        )
        loaded = load_mark("510300", directory=tmp_path, fitted=fitted)
        assert loaded is not None
        assert loaded.fingerprint == published.fingerprint == fingerprint(fitted)

    def test_a_fingerprint_mismatch_is_an_error_not_a_warning(self, fitted, tmp_path: Path) -> None:  # noqa: ANN001
        """The whole point of a mark is that everything prices off the same
        surface. One that no longer matches its own fingerprint is a different
        surface wearing yesterday's blessing, and serving it quietly would
        defeat the exercise.
        """
        publish_mark(fitted, source_kind="snapshot", snapshot=None, rate=0.02, directory=tmp_path)
        path = tmp_path / "510300.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["fingerprint"] = "deadbeefdeadbeef"
        path.write_text(json.dumps(raw), encoding="utf-8")

        with pytest.raises(MarketDataError, match="no longer reproduces"):
            load_mark("510300", directory=tmp_path, fitted=fitted)

    def test_verification_can_be_skipped_when_there_is_nothing_to_check(
        self, fitted, tmp_path: Path
    ) -> None:  # noqa: ANN001
        """publish_mark reads the previous mark to compute drift, and at that
        moment it has no fitted surface for the *previous* inputs to check
        against. So the skip is used internally and has to work."""
        publish_mark(fitted, source_kind="snapshot", snapshot=None, rate=0.02, directory=tmp_path)
        path = tmp_path / "510300.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["fingerprint"] = "deadbeefdeadbeef"
        path.write_text(json.dumps(raw), encoding="utf-8")
        assert load_mark("510300", directory=tmp_path, verify=False) is not None

    def test_a_corrupt_file_says_so(self, tmp_path: Path) -> None:
        (tmp_path / "510300.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(MarketDataError, match="could not be read"):
            load_mark("510300", directory=tmp_path)

    def test_a_file_missing_a_required_field_says_so(self, tmp_path: Path) -> None:
        (tmp_path / "510300.json").write_text('{"underlying":"510300"}', encoding="utf-8")
        with pytest.raises(MarketDataError, match="not readable"):
            load_mark("510300", directory=tmp_path)


class TestTheFingerprint:
    def test_it_is_over_the_parameters_not_the_response(self, fitted) -> None:  # noqa: ANN001
        """The response carries rounded diagnostics and would make the
        fingerprint move for reasons that are not the surface."""
        assert fingerprint(fitted) == fingerprint(fitted)
        assert len(fingerprint(fitted)) == 16

    def test_a_different_surface_fingerprints_differently(self) -> None:
        a = fit_surface("510300", source_kind="snapshot", rate=0.02)
        b = fit_surface("510050", source_kind="snapshot", rate=0.02)
        assert fingerprint(a) != fingerprint(b)

    def test_the_same_inputs_fingerprint_the_same(self) -> None:
        """Which is what makes it worth checking on read. If the calibration
        were not deterministic this whole mechanism would be noise."""
        a = fit_surface("510300", source_kind="snapshot", rate=0.02)
        b = fit_surface("510300", source_kind="snapshot", rate=0.02)
        assert fingerprint(a) == fingerprint(b)


def _atm(fitted) -> list[float]:  # noqa: ANN001
    params = fitted.calibration.params
    return [
        float((theta / t) ** 0.5) for t, theta in zip(params.maturities, params.thetas, strict=True)
    ]


def test_gate_result_round_trips_through_a_dict() -> None:
    gate = GateResult(name="x", value=1.5, threshold=2.0, passed=True, note="n")
    assert gate.as_dict() == {
        "name": "x",
        "value": 1.5,
        "threshold": 2.0,
        "passed": True,
        "note": "n",
    }
