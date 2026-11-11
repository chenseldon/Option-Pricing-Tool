"""Sweep the Wind panel and write the committed anchor artefacts.

Run by hand, not in CI:

    python scripts/build_wind_anchor.py

One ``fit_surface`` takes about 0.3 s on a six-expiry CFFEX chain, so 725 dates
across three roots is roughly ten minutes. That is fine offline and impossible
in a test suite, which is why the distribution is computed once here and
committed rather than recomputed per run.

Three artefacts come out:

``data/wind/anchor.csv``
    One row per (date, underlying): fit quality, SSVI parameters, arbitrage
    margins, rejection counts, the rate used, and the futures forward check.
    CSV rather than parquet, for the reason ``snapshots.py`` already gives --
    readable in a diff, and a changed row is visible in review. Parquet would
    also make the content digest depend on the writer version, which defeats
    the manifest.

``data/wind/manifest.json``
    The digests of the six input files, and **every screening threshold in
    effect when the sweep ran**. A test asserts the live constants still match.
    This is the highest-value part: the realistic failure is not code rot, it
    is someone tuning ``MAX_MONEYNESS_SD`` and the distribution never being
    re-measured.

``data/snapshots/wind/``
    A handful of full chains, written in the existing snapshot format, chosen
    from the artefact's own distribution -- worst fit, best fit, widest futures
    gap, first degraded date, a median day. These let the spot-check tests
    recompute a fit and reconcile it against its artefact row **without access
    to the Wind drive**, which no other machine has.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from otcpricer.core.exceptions import OtcPricerError  # noqa: E402
from otcpricer.marketdata import surface_input as si  # noqa: E402
from otcpricer.marketdata.contracts import resolve_underlying  # noqa: E402
from otcpricer.marketdata.forward import (  # noqa: E402
    DEFAULT_RATE,
    MAX_PARITY_RESIDUAL_REL,
    MIN_PAIRED_STRIKES,  # noqa: E402
)
from otcpricer.marketdata.rates import rate_curve_from_panel  # noqa: E402
from otcpricer.marketdata.snapshots import save_snapshot  # noqa: E402
from otcpricer.marketdata.wind import WIND_FILES, WindPanel, WindPanelSource  # noqa: E402
from otcpricer.models.surface import MIN_USABLE_SLICES, fit_surface_from_chain  # noqa: E402
from otcpricer.validation.futures_check import check_forwards  # noqa: E402

ROOTS = ("IO", "HO", "MO")
OUT_DIR = REPO / "data" / "wind"
SNAPSHOT_DIR = REPO / "data" / "snapshots" / "wind"

#: Column order is fixed so a regenerated file diffs cleanly.
COLUMNS = (
    "as_of",
    "underlying",
    "n_expiries_quoted",
    "n_slices_usable",
    "n_points",
    "n_rejected_quotes",
    "n_rejected_adjusted",
    "n_rejected_no_static",
    "n_rejected_non_positive",
    "n_rejected_expired",
    "raw_rows",
    "fitted",
    "converged",
    "rms_vp",
    "max_vp",
    "rho",
    "eta",
    "gamma",
    "butterfly_margin",
    "calendar_margin",
    "arbitrage_free",
    "min_tau",
    "max_tau",
    "parity_residual_med",
    "parity_residual_max",
    "rate_1y",
    "rate_10y",
    "futures_checked",
    "futures_gap_med",
    "futures_gap_worst",
    "futures_rate_explains",
    "failure",
)


@dataclass
class Row:
    values: dict[str, object]

    def as_csv(self) -> list[object]:
        return [self.values.get(name, "") for name in COLUMNS]


def _round(value: object, places: int) -> object:
    try:
        return round(float(value), places)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""


def build_row(panel: WindPanel, root: str, day: date) -> Row:
    """One (date, underlying) row, with failures recorded rather than raised."""
    values: dict[str, object] = {"as_of": day.isoformat(), "underlying": root}

    try:
        curve = rate_curve_from_panel(panel, day)
        values["rate_1y"] = _round(curve.zero_rates[0], 8)
        values["rate_10y"] = _round(curve.zero_rates[1], 8)
    except OtcPricerError as error:
        values["failure"] = f"rate: {error}"
        return Row(values)

    try:
        chain = WindPanelSource(panel=panel, as_of=day).fetch(root)
    except OtcPricerError as error:
        values["failure"] = f"chain: {str(error)[:80]}"
        values["n_expiries_quoted"] = 0
        values["fitted"] = False
        return Row(values)

    ledger = panel.ledger(root, day)
    values.update(
        {
            "n_expiries_quoted": len(chain.slices),
            "raw_rows": ledger.raw_rows,
            "n_rejected_adjusted": ledger.adjusted_contract,
            "n_rejected_no_static": ledger.no_static_record,
            "n_rejected_non_positive": ledger.non_positive_price,
            "n_rejected_expired": ledger.expired,
            "min_tau": _round(min(s.year_fraction for s in chain.slices), 6),
            "max_tau": _round(max(s.year_fraction for s in chain.slices), 6),
        }
    )

    try:
        fitted = fit_surface_from_chain(chain, rate_curve=curve)
    except OtcPricerError as error:
        values["fitted"] = False
        values["failure"] = f"fit: {str(error)[:80]}"
        return Row(values)

    calibration = fitted.calibration
    params = calibration.params
    usable = fitted.surface_input.usable_slices
    residuals = sorted(s.forward.residual_relative for s in usable)

    values.update(
        {
            "fitted": True,
            "converged": calibration.converged,
            "n_slices_usable": len(usable),
            "n_points": calibration.n_points,
            "n_rejected_quotes": fitted.surface_input.n_rejected,
            "rms_vp": _round(calibration.rms_vol_error * 100.0, 5),
            "max_vp": _round(calibration.max_vol_error * 100.0, 5),
            "rho": _round(params.rho, 6),
            "eta": _round(params.eta, 6),
            "gamma": _round(params.gamma, 6),
            "butterfly_margin": _round(calibration.butterfly_margin, 6),
            "calendar_margin": _round(calibration.calendar_margin, 6),
            "arbitrage_free": fitted.arbitrage.arbitrage_free,
            "parity_residual_med": _round(residuals[len(residuals) // 2], 8) if residuals else "",
            "parity_residual_max": _round(residuals[-1], 8) if residuals else "",
        }
    )

    try:
        futures = panel.futures(resolve_underlying(root).futures_root or "", day)
        report = check_forwards(chain, futures, rate_curve=curve)
        values.update(
            {
                "futures_checked": len(report.checks),
                "futures_gap_med": _round(report.median_gap_relative, 8),
                "futures_gap_worst": _round(report.worst_gap_relative, 8),
                "futures_rate_explains": report.rate_explains_any_gap,
            }
        )
    except OtcPricerError:
        values["futures_checked"] = 0

    return Row(values)


def _digest(path: Path) -> dict[str, object]:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return {"sha256": sha.hexdigest(), "bytes": path.stat().st_size}


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return "unknown"


def thresholds() -> dict[str, object]:
    """Every screening constant the sweep ran under.

    Read off the live modules rather than retyped, so the manifest cannot
    disagree with the code at the moment it is written. A test then asserts the
    live values still match what is recorded here.
    """
    return {
        "DEFAULT_RATE": DEFAULT_RATE,
        "MIN_PAIRED_STRIKES": MIN_PAIRED_STRIKES,
        "MAX_PARITY_RESIDUAL_REL": MAX_PARITY_RESIDUAL_REL,
        "MAX_VOL_UNCERTAINTY": si.MAX_VOL_UNCERTAINTY,
        "MAX_TICK_VOL_UNCERTAINTY": si.MAX_TICK_VOL_UNCERTAINTY,
        "MAX_MONEYNESS_SD": si.MAX_MONEYNESS_SD,
        "MIN_MATURITY_YEARS": si.MIN_MATURITY_YEARS,
        "MIN_POINTS_PER_SLICE": si.MIN_POINTS_PER_SLICE,
        "MIN_USABLE_SLICES": MIN_USABLE_SLICES,
        "tick_size": {root: resolve_underlying(root).tick_size for root in ROOTS},
        "price_convention": "settle",
        "traded_from": "volume",
        "drop_adjusted": True,
        "rate_source": "wind-cgb-2tenor-linear-flat-outside",
        "day_count": "ACT/365F",
    }


def pick_spot_check_dates(rows: list[Row]) -> dict[str, tuple[str, str]]:
    """Choose snapshot dates from the sweep's own distribution, not at random.

    Each one is picked because it is extreme or structurally interesting, so a
    regression has somewhere obvious to show up.
    """
    fitted = [r for r in rows if r.values.get("fitted") is True and r.values.get("rms_vp") != ""]
    if not fitted:  # pragma: no cover
        return {}

    by_rms = sorted(fitted, key=lambda r: float(r.values["rms_vp"]))  # type: ignore[arg-type]
    with_gap = [r for r in fitted if r.values.get("futures_gap_worst") not in ("", None)]
    by_gap = sorted(
        with_gap,
        key=lambda r: float(r.values["futures_gap_worst"]),  # type: ignore[arg-type]
    )

    picks: dict[str, tuple[str, str]] = {
        "best_fit": (str(by_rms[0].values["as_of"]), str(by_rms[0].values["underlying"])),
        "worst_fit": (str(by_rms[-1].values["as_of"]), str(by_rms[-1].values["underlying"])),
        "median_fit": (
            str(by_rms[len(by_rms) // 2].values["as_of"]),
            str(by_rms[len(by_rms) // 2].values["underlying"]),
        ),
    }
    if by_gap:
        picks["widest_futures_gap"] = (
            str(by_gap[-1].values["as_of"]),
            str(by_gap[-1].values["underlying"]),
        )

    # The first date the stale static file starts costing expiries.
    degraded = [
        r
        for r in fitted
        if r.values.get("n_expiries_quoted") not in ("", None)
        and int(r.values["n_expiries_quoted"]) < 6  # type: ignore[arg-type]
    ]
    if degraded:
        first = min(degraded, key=lambda r: str(r.values["as_of"]))
        picks["first_degraded"] = (str(first.values["as_of"]), str(first.values["underlying"]))

    return picks


def main() -> int:
    warnings.filterwarnings("ignore")

    from otcpricer.core.config import get_settings

    directory = get_settings().data.wind_dir
    if not directory:
        print("OTCPRICER_WIND_DIR is not set; nothing to sweep.", file=sys.stderr)
        return 1

    panel = WindPanel(directory)
    days = panel.trading_days
    print(f"sweeping {len(days)} dates x {len(ROOTS)} roots = {len(days) * len(ROOTS)} fits")

    rows: list[Row] = []
    started = datetime.now()
    for index, day in enumerate(days, start=1):
        for root in ROOTS:
            rows.append(build_row(panel, root, day))
        if index % 50 == 0:
            elapsed = (datetime.now() - started).total_seconds()
            print(f"  {index}/{len(days)} dates, {elapsed:.0f}s elapsed", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    anchor_path = OUT_DIR / "anchor.csv"
    with anchor_path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows(row.as_csv() for row in rows)
    print(f"wrote {anchor_path} ({len(rows)} rows)")

    picks = pick_spot_check_dates(rows)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    for label, (stamp, root) in picks.items():
        year, month, day_of = (int(part) for part in stamp.split("-"))
        chain = WindPanelSource(panel=panel, as_of=date(year, month, day_of)).fetch(root)
        path = save_snapshot(chain, SNAPSHOT_DIR, name=stamp)
        print(f"  {label}: {root} {stamp} -> {path.name}")

    manifest = {
        "format_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "git_revision": _git_revision(),
        "roots": list(ROOTS),
        "n_dates": len(days),
        "n_rows": len(rows),
        "first_date": days[0].isoformat(),
        "last_date": days[-1].isoformat(),
        "inputs": {role: _digest(Path(directory) / name) for role, name in WIND_FILES.items()},
        "anchor_sha256": _digest(anchor_path)["sha256"],
        "thresholds": thresholds(),
        "spot_check_dates": {label: list(pair) for label, pair in picks.items()},
        "note": (
            "Regenerate with scripts/build_wind_anchor.py. The thresholds block "
            "is asserted against the live modules by tests/test_wind_anchor.py, "
            "so tuning a screen without re-sweeping fails loudly."
        ),
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {manifest_path}")

    elapsed = (datetime.now() - started).total_seconds()
    print(f"done in {elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
