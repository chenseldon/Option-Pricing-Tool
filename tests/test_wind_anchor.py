"""Tests against the committed 725-day panel anchor.

Why the anchor is an artefact
-----------------------------
A full sweep is 2,175 fits and takes about eleven minutes. That is fine offline
and impossible in a test suite, so it is computed once by
``scripts/build_wind_anchor.py``, committed, and asserted here.

An artefact can rot, so three things guard it:

1. **The manifest pins every screening threshold** the sweep ran under, and a
   test compares them against the live modules. This is the most valuable test
   in the file, because the realistic failure is not code rot -- it is someone
   tuning ``MAX_MONEYNESS_SD`` and the distribution never being re-measured.
2. **Spot checks recompute fits from committed snapshots** and reconcile them
   against their artefact rows. These run *without* the Wind drive, which no
   other machine has.
3. **Coverage facts are pinned exactly**, so a different panel vintage fails
   loudly instead of quietly shifting every statistic.

What the panel shows that one snapshot could not
------------------------------------------------
Two findings here are structurally invisible on a single date:

- SSVI's ``gamma`` sits at its **upper bound on 44% of fits**. Since the
  short-maturity wing diverges like ``t^(-gamma/2)``, that rate is being set by
  the bound rather than by the data on nearly half the panel.
- The three indices have **genuinely different smile orientations**: HO quotes
  a positive ``rho`` on 90% of days, MO on only 32%. A single date cannot
  distinguish a persistent structure from that day's weather.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
ANCHOR = REPO / "data" / "wind" / "anchor.csv"
MANIFEST = REPO / "data" / "wind" / "manifest.json"
WIND_SNAPSHOTS = REPO / "data" / "snapshots" / "wind"

needs_anchor = pytest.mark.skipif(
    not ANCHOR.exists() or not MANIFEST.exists(),
    reason="panel anchor not generated (scripts/build_wind_anchor.py)",
)


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    with ANCHOR.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def manifest() -> dict[str, object]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fitted(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows if r["fitted"] == "True"]


def _floats(rows: list[dict[str, str]], column: str, root: str | None = None) -> np.ndarray:
    return np.array(
        [
            float(r[column])
            for r in rows
            if r[column] not in ("", None) and (root is None or r["underlying"] == root)
        ]
    )


# ---------------------------------------------------------------------------
# The anti-rot mechanism
# ---------------------------------------------------------------------------


@needs_anchor
class TestManifest:
    def test_the_recorded_thresholds_still_match_the_code(self, manifest) -> None:  # type: ignore[no-untyped-def]
        """The highest-value test here.

        Every number in the anchor was produced under these screens. Tuning one
        without re-sweeping would leave a committed distribution that no longer
        describes the code, and nothing else would notice.
        """
        import sys

        sys.path.insert(0, str(REPO / "scripts"))
        from build_wind_anchor import thresholds

        recorded = dict(manifest["thresholds"])  # type: ignore[arg-type]
        live = thresholds()

        assert set(recorded) == set(live), "a threshold was added or removed"
        for name, value in live.items():
            assert recorded[name] == value, (
                f"{name} is now {value!r} but the anchor was built with "
                f"{recorded[name]!r}; regenerate with scripts/build_wind_anchor.py"
            )

    def test_records_what_it_was_built_from(self, manifest) -> None:  # type: ignore[no-untyped-def]
        """So a changed input is detectable even though the inputs are not committed."""
        inputs = manifest["inputs"]
        assert len(inputs) == 6  # type: ignore[arg-type]
        for digest in inputs.values():  # type: ignore[union-attr]
            assert len(digest["sha256"]) == 64
            assert digest["bytes"] > 0

        assert len(str(manifest["git_revision"])) == 40
        assert manifest["n_rows"] == 2175
        assert manifest["first_date"] == "2023-06-30"
        assert manifest["last_date"] == "2026-06-29"

    def test_the_anchor_matches_its_recorded_digest(self, manifest) -> None:  # type: ignore[no-untyped-def]
        """An edited CSV is a different claim and must not pass silently."""
        import hashlib

        sha = hashlib.sha256()
        with ANCHOR.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                sha.update(block)
        assert sha.hexdigest() == manifest["anchor_sha256"]


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


@needs_anchor
class TestCoverage:
    def test_every_date_and_root_has_a_row(self, rows: list[dict[str, str]]) -> None:
        assert len(rows) == 2175
        assert len({r["as_of"] for r in rows}) == 725
        assert {r["underlying"] for r in rows} == {"IO", "HO", "MO"}

    def test_the_fittable_population_is_690_of_725_per_root(
        self, rows: list[dict[str, str]], fitted: list[dict[str, str]]
    ) -> None:
        """Which is the denominator every distributional claim below uses.

        The static contract file stops before the quote panel does, so 35 dates
        per root fall below the two-slice minimum. Saying "725 days" of a
        statistic computed on 690 would be the kind of number this project
        refuses to publish.
        """
        assert len(fitted) == 2070
        assert len(rows) - len(fitted) == 105
        for root in ("IO", "HO", "MO"):
            assert sum(1 for r in fitted if r["underlying"] == root) == 690

    def test_complete_six_expiry_chains_are_516_days(self, rows: list[dict[str, str]]) -> None:
        counts = [int(r["n_expiries_quoted"]) for r in rows if r["underlying"] == "IO"]
        assert sum(1 for n in counts if n == 6) == 516
        assert sum(1 for n in counts if n >= 4) == 656
        assert sum(1 for n in counts if n >= 2) == 694
        assert sum(1 for n in counts if n == 0) == 6

    def test_every_failure_says_why(self, rows: list[dict[str, str]]) -> None:
        for row in rows:
            if row["fitted"] != "True":
                assert row["failure"], f"{row['as_of']} {row['underlying']} failed silently"


# ---------------------------------------------------------------------------
# Fit quality over 725 days
# ---------------------------------------------------------------------------


@needs_anchor
class TestFitQuality:
    @pytest.mark.parametrize(
        ("root", "median", "p90", "worst"),
        [
            ("IO", 0.476, 0.716, 3.155),
            ("HO", 0.565, 0.873, 2.806),
            ("MO", 0.588, 1.025, 2.178),
        ],
    )
    def test_the_rms_distribution(
        self, fitted: list[dict[str, str]], root: str, median: float, p90: float, worst: float
    ) -> None:
        """A distribution, where Phase 2a could only report one afternoon.

        For scale, the akshare captures fitted to 0.587 (510050) and 0.654
        (510300) volatility points. These medians are comparable or better on
        roughly three times the points and 2.4 times the tenor reach -- which
        is the substantive argument that the migration improved the inputs and
        not just their quantity.
        """
        rms = _floats(fitted, "rms_vp", root)
        assert np.median(rms) == pytest.approx(median, abs=0.01)
        assert np.percentile(rms, 90) == pytest.approx(p90, abs=0.02)
        assert rms.max() == pytest.approx(worst, abs=0.02)

    def test_every_fit_converged_and_is_arbitrage_free(self, fitted: list[dict[str, str]]) -> None:
        """2,070 of 2,070, on both counts.

        The constrained SSVI fit imposes the no-arbitrage conditions
        structurally rather than checking them afterwards, and this is that
        claim tested across a panel instead of a date.
        """
        assert all(r["converged"] == "True" for r in fitted)
        assert all(r["arbitrage_free"] == "True" for r in fitted)

    def test_the_arbitrage_margins_are_strictly_positive(
        self, fitted: list[dict[str, str]]
    ) -> None:
        assert _floats(fitted, "butterfly_margin").min() > 0.0
        assert _floats(fitted, "calendar_margin").min() > 0.0

    def test_the_tenor_reach_is_what_the_migration_bought(
        self, fitted: list[dict[str, str]]
    ) -> None:
        """Up to 1.008 years, against the akshare captures' 0.41.

        It does not solve the extrapolation problem -- a three-year snowball is
        still mostly extrapolated -- but it moves the extrapolated share from
        86.5% to about 66%.
        """
        max_tau = _floats(fitted, "max_tau")
        assert max_tau.max() == pytest.approx(1.008, abs=0.002)
        assert np.median(max_tau) > 0.8
        assert 1.0 - max_tau.max() / 3.0 == pytest.approx(0.664, abs=0.01)


# ---------------------------------------------------------------------------
# What only a panel can show
# ---------------------------------------------------------------------------


@needs_anchor
class TestPanelOnlyFindings:
    def test_gamma_sits_at_its_upper_bound_on_44_percent_of_fits(
        self, fitted: list[dict[str, str]]
    ) -> None:
        """A real limitation, and invisible on any single date.

        SSVI's power-law ``phi(theta) = eta / (theta^gamma * (1+theta)^(1-gamma))``
        makes the short-maturity wing diverge like ``t^(-gamma/2)``. With gamma
        pinned at its 0.5 bound on nearly half the panel, that divergence rate
        is being set by the bound rather than by the quotes -- which matters
        directly for local volatility, where gamma governs the short-dated
        wing.

        Pinned so the fraction cannot drift unnoticed, and so that relaxing the
        bound shows up here as a deliberate change.
        """
        gamma = _floats(fitted, "gamma")
        at_upper = int((gamma >= 0.4999).sum())

        assert at_upper == 956
        assert at_upper / len(gamma) == pytest.approx(0.462, abs=0.01)
        # The lower bound, by contrast, is almost never reached.
        assert int((gamma <= 0.0011).sum()) == 6

    def test_the_three_indices_have_different_smile_orientations(
        self, fitted: list[dict[str, str]]
    ) -> None:
        """Persistent structure, which one date cannot distinguish from weather.

        HO (SSE 50) quotes a positive rho -- a richer call wing -- on 90% of
        days, while MO (CSI 1000) does so on only 32%. This is why a positive
        rho on a single IO fit is data rather than a sign error.
        """
        shares = {
            root: float((_floats(fitted, "rho", root) > 0.0).mean()) for root in ("IO", "HO", "MO")
        }
        assert shares["HO"] == pytest.approx(0.903, abs=0.02)
        assert shares["IO"] == pytest.approx(0.691, abs=0.02)
        assert shares["MO"] == pytest.approx(0.320, abs=0.02)
        assert shares["HO"] > shares["IO"] > shares["MO"]

    def test_the_rate_is_excluded_on_most_but_not_all_rows(
        self, fitted: list[dict[str, str]]
    ) -> None:
        """The honest version of the futures-check conclusion.

        On two hand-picked dates no expiry's gap could be explained by a
        plausible rate error, which is what the module's own tests assert. Over
        the panel it is 87.6% of rows, not 100% -- and the exceptions are the
        rows with the smallest gaps, where there is least to explain. Stated as
        a proportion because the absolute claim is false.
        """
        explains = [r["futures_rate_explains"] == "True" for r in fitted]
        assert sum(explains) == 257
        assert 1.0 - sum(explains) / len(explains) == pytest.approx(0.876, abs=0.01)

    def test_the_futures_gap_distribution(self, fitted: list[dict[str, str]]) -> None:
        """Median a quarter to a third of a percent, worst case 2.5%.

        This is the error bar on the moneyness coordinate the surface is fitted
        in, which nothing could measure before the futures were available.
        """
        for root, median, worst in [
            ("IO", 0.358, 1.277),
            ("HO", 0.261, 1.227),
            ("MO", 0.363, 2.540),
        ]:
            gaps = _floats(fitted, "futures_gap_worst", root) * 100.0
            assert np.median(gaps) == pytest.approx(median, abs=0.02)
            assert gaps.max() == pytest.approx(worst, abs=0.02)


# ---------------------------------------------------------------------------
# Spot checks, which run without the Wind drive
# ---------------------------------------------------------------------------


@needs_anchor
class TestSpotChecks:
    """Recompute a fit from a committed chain and reconcile it with its row.

    The point of using snapshots rather than the panel is that these run
    anywhere. "Spot-check a few dates live" is not available on a machine
    without the institutional data, which is every machine but one.
    """

    @pytest.fixture(scope="class")
    def picks(self, manifest) -> dict[str, tuple[str, str]]:  # type: ignore[no-untyped-def]
        return {
            label: (pair[0], pair[1])
            for label, pair in manifest["spot_check_dates"].items()  # type: ignore[union-attr]
        }

    def test_the_snapshots_cover_the_interesting_dates(
        self, picks: dict[str, tuple[str, str]]
    ) -> None:
        """Chosen from the distribution, not at random, so a regression shows up."""
        assert set(picks) >= {"best_fit", "worst_fit", "median_fit", "widest_futures_gap"}
        for stamp, root in picks.values():
            assert (WIND_SNAPSHOTS / f"{root}_{stamp}.json").exists()

    def test_each_snapshot_refits_to_its_anchor_row(
        self, picks: dict[str, tuple[str, str]], rows: list[dict[str, str]]
    ) -> None:
        """The reconciliation that makes the artefact trustworthy.

        Tight tolerances on purpose: the fit is deterministic, so anything
        beyond floating-point noise means the pipeline moved under the
        committed numbers.
        """
        from otcpricer.marketdata.rates import RateCurve
        from otcpricer.marketdata.snapshots import load_snapshot
        from otcpricer.models.surface import fit_surface_from_chain

        by_key = {(r["as_of"], r["underlying"]): r for r in rows}

        for stamp, root in picks.values():
            expected = by_key[(stamp, root)]
            chain = load_snapshot(WIND_SNAPSHOTS, root, name=stamp)

            # The sweep used the panel's curve; rebuild it from the row's own
            # recorded rates so the snapshot needs no access to the rate file.
            curve = RateCurve(
                as_of=chain.as_of,
                tenors=(1.0, 10.0),
                zero_rates=(float(expected["rate_1y"]), float(expected["rate_10y"])),
                source="anchor-replay",
            )
            refitted = fit_surface_from_chain(chain, rate_curve=curve)
            calibration = refitted.calibration

            assert calibration.n_points == int(expected["n_points"]), (stamp, root)
            assert calibration.rms_vol_error * 100.0 == pytest.approx(
                float(expected["rms_vp"]), abs=1e-4
            ), (stamp, root)
            assert calibration.params.rho == pytest.approx(float(expected["rho"]), abs=1e-5)
            assert calibration.params.eta == pytest.approx(float(expected["eta"]), abs=1e-5)
            assert calibration.params.gamma == pytest.approx(float(expected["gamma"]), abs=1e-5)

    def test_the_committed_chains_carry_their_provenance(
        self, picks: dict[str, tuple[str, str]]
    ) -> None:
        """Including the price convention, which is the choice most worth recording.

        Writing these snapshots exposed a provenance loss: ``load_snapshot``
        set the source to ``snapshot:<name>`` and discarded the origin the file
        had stored all along. A Wind chain replayed that way no longer said it
        was built from settlement prices rather than last trades, and those are
        not interchangeable -- the two conventions differ by 5.7x in parity
        residual. The loader now carries both layers.
        """
        from otcpricer.marketdata.snapshots import load_snapshot

        for stamp, root in picks.values():
            chain = load_snapshot(WIND_SNAPSHOTS, root, name=stamp)

            # That it is a replay...
            assert chain.source.startswith(f"snapshot:{stamp}")
            # ...and what it is a replay of, price convention included.
            assert chain.source.endswith(f"wind:{stamp}:settle")

            assert chain.as_of.isoformat() == stamp
            assert any("rejections" in w for w in chain.warnings)
