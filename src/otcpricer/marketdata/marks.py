"""The published surface: the one every pricer reads.

Why this exists
---------------
Every pricing route fitted its own surface. There is a process cache, so in
practice one fit was shared, but nothing made it *the* surface -- no record of
which fit was blessed, when, or on what evidence. A desk does not work that
way. The surface is marked once a day, checked, published, and then everything
in the book is valued off that one mark, so a call, its put, its spreads and
its digital cannot contradict each other. The usual phrasing is that no
consumer fits its own surface.

What a mark stores
------------------
The *inputs* plus a fingerprint of the fitted parameters -- not the surface
object. Reading a mark re-fits from the same inputs, which costs 0.3s and is
cached after the first, and then verifies the fingerprint.

That buys two things a pickled surface would not. It survives a restart
without a deserialiser to maintain. And it makes reproducibility a checked
property rather than an assumption: if the same inputs stop producing the same
surface -- a library upgrade changing an optimiser's path, say -- reading the
mark fails and says which number moved, instead of quietly serving a different
surface under yesterday's blessing.

The gates
---------
A mark cannot be published unless the fit passes. The thresholds here are
measured on this project's own data, not borrowed: published guidance for a
listed-vanilla desk quotes a fit tolerance around 5 basis points, and the
committed chains fit to 65bp (510300) and 59bp (510050). A 5bp gate would
reject every surface this project can build, which would make the gate a
formality to be switched off rather than a check. So the tolerance is set from
what the data actually supports, and the measurement is recorded beside it.

One gate is deliberately weaker than the rest, and says so: the day-over-day
parameter drift. The right form is a band from the historical distribution of
daily moves, and that needs a history this installation does not have in hand
-- the 725-day panel is institutional data and is not committed. So it is a
sanity bound on an implausible jump rather than a statistical band, and the
docstring on :data:`MAX_ATM_VOL_DRIFT` says how to replace it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from otcpricer.core.exceptions import MarketDataError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from otcpricer.models.surface import FittedSurface

__all__ = [
    "MAX_ATM_VOL_DRIFT",
    "MAX_RMS_VOL_POINTS",
    "GateResult",
    "SurfaceMark",
    "evaluate_gates",
    "fingerprint",
    "load_mark",
    "mark_directory",
    "publish_mark",
]

#: Worst fit a surface may carry and still be publishable, in volatility
#: points.
#:
#: Measured on the committed chains: 0.654 (510300) and 0.587 (510050). Set at
#: three times the worse of those, so a normal day passes comfortably and a
#: chain that has degenerated -- a stale feed, a holiday half-session, a
#: screening bug letting junk through -- does not.
#:
#: Deliberately *not* the ~5bp a listed-vanilla desk would use. These are
#: exchange settlement prices on an ETF option book, screened down from 126
#: quotes to 42; 5bp is unreachable on them, and a gate nothing can pass is a
#: gate that gets turned off.
MAX_RMS_VOL_POINTS = 2.0

#: Largest at-the-money volatility move, in points, between consecutive marks.
#:
#: **This is a sanity bound, not a statistical band.** The right test is
#: whether today's move sits inside the historical distribution of daily
#: moves, which needs that distribution; the 725-day panel that would give it
#: is institutional data and is not committed here. Ten points of at-the-money
#: volatility in one day on an index is essentially always a data error rather
#: than a market move, so that is the bound until the band can be measured.
#:
#: To replace it: sweep the panel the way ``scripts/build_wind_anchor.py``
#: does, fit each day, and take a high quantile of ``|Δ atm_vol|``.
MAX_ATM_VOL_DRIFT = 0.10


@dataclass(frozen=True, slots=True)
class GateResult:
    """One publication check, with the number it was decided on.

    Carrying ``value`` and ``threshold`` rather than only ``passed`` is the
    point: a refusal that does not say how far off it was leaves the caller
    guessing whether the data is slightly stale or completely wrong.
    """

    name: str
    value: float | None
    threshold: float | None
    passed: bool
    note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": None if self.value is None else round(float(self.value), 8),
            "threshold": self.threshold,
            "passed": self.passed,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class SurfaceMark:
    """A published surface, as the inputs that produce it plus a fingerprint."""

    underlying: str
    as_of: str
    source: str
    source_kind: str | None
    snapshot: str | None
    csv_token: str | None
    rate: float
    published_at: str
    fingerprint: str
    atm_vols: list[float]
    maturities: list[float]
    rms_vol_points: float
    gates: list[GateResult]

    def as_dict(self) -> dict[str, object]:
        return {
            "underlying": self.underlying,
            "as_of": self.as_of,
            "source": self.source,
            "source_kind": self.source_kind,
            "snapshot": self.snapshot,
            "csv_token": self.csv_token,
            "rate": self.rate,
            "published_at": self.published_at,
            "fingerprint": self.fingerprint,
            "atm_vols": self.atm_vols,
            "maturities": self.maturities,
            "rms_vol_points": self.rms_vol_points,
            "gates": [g.as_dict() for g in self.gates],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SurfaceMark:
        try:
            return cls(
                underlying=str(raw["underlying"]),
                as_of=str(raw["as_of"]),
                source=str(raw["source"]),
                source_kind=raw.get("source_kind"),
                snapshot=raw.get("snapshot"),
                csv_token=raw.get("csv_token"),
                rate=float(raw["rate"]),
                published_at=str(raw["published_at"]),
                fingerprint=str(raw["fingerprint"]),
                atm_vols=[float(v) for v in raw.get("atm_vols", [])],
                maturities=[float(v) for v in raw.get("maturities", [])],
                rms_vol_points=float(raw["rms_vol_points"]),
                gates=[
                    GateResult(
                        name=str(g["name"]),
                        value=None if g.get("value") is None else float(g["value"]),
                        threshold=g.get("threshold"),
                        passed=bool(g["passed"]),
                        note=str(g.get("note", "")),
                    )
                    for g in raw.get("gates", [])
                ],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MarketDataError(f"mark file is not readable: {exc}") from exc


def fingerprint(fitted: FittedSurface) -> str:
    """A short digest of the fitted parameters.

    Over the parameters rather than over the response: the response carries
    rounded diagnostics and a timestamp, so digesting it would make the
    fingerprint change for reasons that are not the surface.
    """
    params = fitted.calibration.params
    payload = json.dumps(
        {
            "rho": round(float(params.rho), 10),
            "eta": round(float(params.eta), 10),
            "gamma": round(float(params.gamma), 10),
            "maturities": [round(float(t), 10) for t in params.maturities],
            "thetas": [round(float(t), 12) for t in params.thetas],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def evaluate_gates(fitted: FittedSurface, previous: SurfaceMark | None = None) -> list[GateResult]:
    """Run every publication check and return all of them.

    All of them, not the first failure: a reader deciding whether to override
    needs to see whether one thing is marginal or everything is wrong.
    """
    calibration = fitted.calibration
    arbitrage = fitted.arbitrage
    rms = float(calibration.rms_vol_error) * 100.0

    gates = [
        GateResult(
            name="fit_quality",
            value=rms,
            threshold=MAX_RMS_VOL_POINTS,
            passed=rms <= MAX_RMS_VOL_POINTS,
            note=(
                f"vega-weighted RMS across {calibration.n_points} screened quotes, "
                f"in volatility points. Measured on the committed chains: 0.654 "
                f"and 0.587."
            ),
        ),
        GateResult(
            name="butterfly",
            value=float(arbitrage.min_butterfly),
            threshold=0.0,
            passed=bool(arbitrage.butterfly_free),
            note=(
                "Minimum of Gatheral's g over the checked grid. At or below zero "
                "the surface implies a negative density, and a barrier payoff is "
                "concentrated exactly where that would be."
            ),
        ),
        GateResult(
            name="calendar",
            value=float(arbitrage.min_calendar),
            threshold=0.0,
            passed=bool(arbitrage.calendar_free),
            note=(
                "Minimum increase in total variance between adjacent maturities. "
                "Negative means a longer expiry carries less variance than a "
                "shorter one."
            ),
        ),
        GateResult(
            name="arbitrage_free",
            value=float(arbitrage.n_violations),
            threshold=0.0,
            passed=bool(arbitrage.arbitrage_free),
            note=(
                f"{arbitrage.n_checked} checks on a dense grid, independent of the "
                f"constraints the calibration imposed. A fit can satisfy its own "
                f"constraints and still violate these off the nodes it was fitted on."
            ),
        ),
    ]

    # Rounded to the precision the mark file stores, so re-publishing an
    # identical surface reports a drift of exactly zero rather than the 3e-09
    # of comparing a rounded stored value against an unrounded fresh one. The
    # difference never mattered against a 0.10 threshold; a drift of "0" that
    # prints as 3.3e-09 is the kind of thing that later costs someone an hour.
    atm = [round(float(v), 8) for v in _atm_vols(fitted)]
    if previous is None or not previous.atm_vols:
        gates.append(
            GateResult(
                name="drift",
                value=None,
                threshold=MAX_ATM_VOL_DRIFT,
                passed=True,
                note=(
                    "No previous mark to compare against, so nothing to check. "
                    "The first mark for an underlying always passes this."
                ),
            )
        )
    else:
        n = min(len(atm), len(previous.atm_vols))
        worst = max((abs(atm[i] - previous.atm_vols[i]) for i in range(n)), default=0.0)
        gates.append(
            GateResult(
                name="drift",
                value=worst,
                threshold=MAX_ATM_VOL_DRIFT,
                passed=worst <= MAX_ATM_VOL_DRIFT,
                note=(
                    f"Largest at-the-money move against the mark published at "
                    f"{previous.published_at}, over {n} shared expiries. A sanity "
                    f"bound on a data error, not a statistical band -- see "
                    f"MAX_ATM_VOL_DRIFT."
                ),
            )
        )
    return gates


def _atm_vols(fitted: FittedSurface) -> list[float]:
    params = fitted.calibration.params
    return [
        float((theta / t) ** 0.5) for t, theta in zip(params.maturities, params.thetas, strict=True)
    ]


def mark_directory(root: Path | str = "data/marks") -> Path:
    return Path(root)


def _mark_path(underlying: str, directory: Path) -> Path:
    return directory / f"{underlying}.json"


def publish_mark(
    fitted: FittedSurface,
    *,
    source_kind: str | None,
    snapshot: str | None,
    rate: float,
    csv_token: str | None = None,
    directory: Path | str = "data/marks",
    force: bool = False,
) -> SurfaceMark:
    """Publish ``fitted`` as the mark for its underlying.

    Refuses if any gate fails, unless ``force``. ``force`` exists because a
    desk sometimes has to mark a surface it knows is ugly -- a half-session, a
    holiday, a single expiry gone wide -- and the alternative to an override
    is editing the thresholds, which would remove the check for every
    subsequent day. The override is recorded in the mark, so the next reader
    can see the gate failed and that someone decided anyway.
    """
    directory = Path(directory)
    previous = load_mark(fitted.underlying_code, directory=directory, verify=False)
    gates = evaluate_gates(fitted, previous)
    failed = [g for g in gates if not g.passed]
    if failed and not force:
        detail = "; ".join(
            f"{g.name}={g.value:.4f} against {g.threshold}" if g.value is not None else g.name
            for g in failed
        )
        raise MarketDataError(
            f"refusing to publish {fitted.underlying_code}: {len(failed)} of "
            f"{len(gates)} gates failed ({detail}). Pass force=True to publish "
            f"anyway; the override is recorded in the mark."
        )

    mark = SurfaceMark(
        underlying=fitted.underlying_code,
        as_of=fitted.as_of,
        source=fitted.source,
        source_kind=source_kind,
        snapshot=snapshot,
        csv_token=csv_token,
        rate=float(rate),
        published_at=datetime.now(UTC).isoformat(timespec="seconds"),
        fingerprint=fingerprint(fitted),
        atm_vols=[round(v, 8) for v in _atm_vols(fitted)],
        maturities=[round(float(t), 8) for t in fitted.calibration.params.maturities],
        rms_vol_points=round(float(fitted.calibration.rms_vol_error) * 100.0, 6),
        gates=gates,
    )

    directory.mkdir(parents=True, exist_ok=True)
    payload = mark.as_dict()
    payload["forced"] = bool(failed and force)
    _mark_path(mark.underlying, directory).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return mark


def load_mark(
    underlying: str,
    *,
    directory: Path | str = "data/marks",
    verify: bool = True,
    fitted: FittedSurface | None = None,
) -> SurfaceMark | None:
    """Read the published mark for ``underlying``, or ``None`` if there is none.

    With ``verify`` and a ``fitted`` surface, checks that re-fitting the mark's
    inputs reproduced it. A mismatch is an error rather than a warning: the
    whole point of a mark is that everything prices off the same surface, and
    a surface that no longer matches its own fingerprint is a different
    surface wearing yesterday's blessing.
    """
    path = _mark_path(underlying, Path(directory))
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MarketDataError(f"mark file {path} could not be read: {exc}") from exc

    mark = SurfaceMark.from_dict(raw)
    if verify and fitted is not None:
        actual = fingerprint(fitted)
        if actual != mark.fingerprint:
            raise MarketDataError(
                f"the mark for {underlying} was published with fingerprint "
                f"{mark.fingerprint} and the same inputs now fit to {actual}. "
                f"Re-publish, or find out what changed -- a mark that no longer "
                f"reproduces is not a mark."
            )
    return mark
