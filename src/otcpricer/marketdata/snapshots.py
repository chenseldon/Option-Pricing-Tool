"""Capture and replay option chains as on-disk snapshots.

A snapshot is a real chain, pulled once and committed. It serves three
purposes, and the first two matter more than the convenience:

* **Tests become deterministic and offline.** A suite that depends on a third
  party's uptime cannot tell you whether your code works. The live feed dropped
  connections twice during development, so this is not hypothetical.
* **A regression baseline exists.** Calibration results can be pinned to a
  fixed input, which is impossible against a moving feed.
* It avoids hammering an undocumented public endpoint during development.

Stored as JSON rather than a pickle: a committed fixture should be readable in
a diff, and should not execute anything on load.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from otcpricer.core.exceptions import MarketDataError
from otcpricer.core.logging import get_logger
from otcpricer.marketdata.contracts import resolve_underlying
from otcpricer.marketdata.quotes import OptionChain, build_slice

__all__ = ["capture_snapshot", "list_snapshots", "load_snapshot", "save_snapshot"]

_log = get_logger(__name__)

#: Bumped if the on-disk shape changes, so an old file fails loudly rather than
#: being misread.
SNAPSHOT_FORMAT_VERSION = 1


def _snapshot_path(directory: Path, underlying_code: str, name: str) -> Path:
    return Path(directory) / f"{underlying_code}_{name}.json"


def save_snapshot(chain: OptionChain, directory: Path | str, *, name: str | None = None) -> Path:
    """Write a chain to disk and return the path.

    Args:
        chain: The chain to store.
        directory: Destination directory, created if absent.
        name: Snapshot name; defaults to the capture date.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    snapshot_name = name or chain.as_of.isoformat()

    payload = {
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "underlying": chain.underlying.code,
        "underlying_name": chain.underlying.name,
        "as_of": chain.as_of.isoformat(),
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "source": chain.source,
        "warnings": list(chain.warnings),
        "slices": [
            {
                "expiry": option_slice.expiry.isoformat(),
                "days_to_expiry": option_slice.days_to_expiry,
                "quotes": [
                    {
                        "code": quote.contract.raw,
                        "last": quote.last,
                        "previous_settlement": quote.previous_settlement,
                        "strike": quote.strike,
                    }
                    for quote in option_slice.quotes
                ],
            }
            for option_slice in chain.slices
        ],
    }

    path = _snapshot_path(directory, chain.underlying.code, snapshot_name)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _log.info(
        "snapshot saved",
        extra={"path": str(path), "expiries": len(chain.slices), "source": chain.source},
    )
    return path


def load_snapshot(
    directory: Path | str, underlying_code: str, *, name: str | None = None
) -> OptionChain:
    """Read a chain back from disk.

    Args:
        directory: Where snapshots live.
        underlying_code: Exchange ticker.
        name: Snapshot name; defaults to the most recent available.

    The returned ``source`` is ``snapshot:<name>:<origin>``, carrying both the
    replay layer and the feed the capture came from. The origin matters because
    it encodes the price convention: a Wind chain is built from settlement
    prices and an akshare one from last trades, and those are not
    interchangeable.

    Raises:
        MarketDataError: if nothing is stored for this underlying, the named
            snapshot is missing, or the file was written by a different format
            version.
    """
    directory = Path(directory)
    underlying = resolve_underlying(underlying_code)

    if name is None:
        available = list_snapshots(directory, underlying.code)
        if not available:
            raise MarketDataError(
                f"no snapshots for {underlying.code} in {directory}. Capture one with "
                f"capture_snapshot(), or set the data source to 'akshare' to go live."
            )
        name = available[0]

    path = _snapshot_path(directory, underlying.code, name)
    if not path.exists():
        raise MarketDataError(f"snapshot not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("format_version")
    if version != SNAPSHOT_FORMAT_VERSION:
        raise MarketDataError(
            f"snapshot {path.name} is format version {version}, this build reads "
            f"{SNAPSHOT_FORMAT_VERSION}; re-capture it rather than risk a misread"
        )

    as_of = date.fromisoformat(payload["as_of"])
    slices = [
        build_slice(
            underlying,
            raw_slice["quotes"],
            expiry=date.fromisoformat(raw_slice["expiry"]),
            as_of=as_of,
            days_to_expiry=int(raw_slice["days_to_expiry"]),
        )
        for raw_slice in payload["slices"]
    ]

    # Two facts, both worth keeping: that this is a replay, and what it is a
    # replay *of*. The origin was always written to the file and always thrown
    # away on load, which quietly lost the price convention -- a Wind chain
    # replayed as "snapshot:2024-08-27" no longer says it was built from
    # settlement prices. Appended rather than substituted so the existing
    # "snapshot:" prefix still identifies the replay layer.
    origin = str(payload.get("source", "") or "").strip()
    source = f"snapshot:{name}" if not origin else f"snapshot:{name}:{origin}"

    return OptionChain(
        underlying=underlying,
        as_of=as_of,
        slices=tuple(sorted(slices, key=lambda s: s.expiry)),
        source=source,
        warnings=tuple(payload.get("warnings", ())),
    )


def list_snapshots(directory: Path | str, underlying_code: str) -> list[str]:
    """Snapshot names available for an underlying, newest first."""
    directory = Path(directory)
    if not directory.exists():
        return []
    prefix = f"{underlying_code}_"
    names = [
        path.stem[len(prefix) :]
        for path in directory.glob(f"{prefix}*.json")
        if path.stem.startswith(prefix)
    ]
    return sorted(names, reverse=True)


def capture_snapshot(
    underlying_code: str, directory: Path | str, *, name: str | None = None
) -> Path:
    """Pull a chain from the live feed and store it.

    The one function here that deliberately touches the network. Intended to be
    run by hand when a fresh fixture is wanted, never from a test.
    """
    from otcpricer.marketdata.sources import AkshareSource

    chain = AkshareSource().fetch(underlying_code)
    return save_snapshot(chain, directory, name=name)
