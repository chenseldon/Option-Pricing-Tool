"""Option chain sources: live feed, offline snapshot, and CSV.

Offline is the default
----------------------
:class:`~otcpricer.core.config.DataSettings` ships with ``source="snapshot"``,
so nothing reaches for the network unless asked. That is a correctness
requirement, not a preference: a test suite whose result depends on a third
party's uptime does not tell you whether your code works.

The live feed earned that caution during development. Two distinct failures
showed up within minutes of each other -- a ``RemoteDisconnected`` mid-pull and
an ``HTTPError`` on a different endpoint -- so :class:`AkshareSource` retries
with backoff and, when it finally gives up, says what it was asking for rather
than surfacing a bare transport error.

The feed is also an undocumented scrape of a public site. Column *names* arrive
in Chinese and have changed shape across versions, so rows are read
positionally and the layout is validated once per pull.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

from otcpricer.core.exceptions import MarketDataError
from otcpricer.core.logging import get_logger
from otcpricer.marketdata.contracts import Underlying, resolve_underlying
from otcpricer.marketdata.quotes import OptionChain, OptionSlice, build_slice

__all__ = [
    "AkshareSource",
    "CachingSource",
    "CsvSource",
    "OptionChainSource",
    "SnapshotSource",
    "build_source",
]

_log = get_logger(__name__)

#: Column order of the board payload. Read positionally because the header is
#: localised and has moved between upstream versions.
_BOARD_COLUMNS = (
    "timestamp",
    "code",
    "last",
    "change_pct",
    "previous_settlement",
    "strike",
    "count",
)


class OptionChainSource(Protocol):
    """Anything that can supply an option chain."""

    @property
    def name(self) -> str:
        """Short identifier recorded on the chain, for provenance."""
        ...

    def fetch(self, underlying_code: str) -> OptionChain:
        """Return every available expiry for one underlying."""
        ...


# ---------------------------------------------------------------------------
# Live feed
# ---------------------------------------------------------------------------


@dataclass
class AkshareSource:
    """Pull the live board from akshare.

    One request per expiry: the board endpoint returns a whole slice, calls and
    puts together, so a four-expiry surface costs four calls rather than one per
    contract.

    Attributes:
        max_attempts: Tries per request before giving up.
        backoff_seconds: Initial pause between tries; doubles each time.
        max_expiries: Cap on expiries pulled, nearest first. The far months are
            thinly traded and contribute little to a surface fit.
    """

    max_attempts: int = 4
    backoff_seconds: float = 2.0
    max_expiries: int = 6

    @property
    def name(self) -> str:
        return "akshare"

    def _retry(
        self,
        what: str,
        call: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call with backoff, reporting what was being fetched if it fails."""
        delay = self.backoff_seconds
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return call(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - transport errors are varied
                last_error = exc
                _log.warning(
                    "market data request failed",
                    extra={"what": what, "attempt": attempt, "error": type(exc).__name__},
                )
                if attempt < self.max_attempts:
                    time.sleep(delay)
                    delay *= 2.0
        raise MarketDataError(
            f"could not fetch {what} after {self.max_attempts} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    def fetch(self, underlying_code: str) -> OptionChain:
        """Pull every available expiry for one underlying.

        The symbol is resolved against the registry *before* the feed library
        is imported. That ordering is the whole point of the registry: an
        unsupported symbol is refused without a network library, let alone a
        request. The import used to come first, so on a machine without
        akshare a bad symbol produced ``ModuleNotFoundError`` instead of a
        message naming the symbols that are supported -- and the test for it
        is called "refused before any request".
        """
        underlying = resolve_underlying(underlying_code)
        sina_symbol = _sina_symbol(underlying)

        import akshare as ak

        expiry_tags = self._retry(
            f"expiry list for {underlying.code}",
            ak.option_sse_list_sina,
            symbol=sina_symbol,
            exchange="null",
        )
        tags = list(expiry_tags)[: self.max_expiries]
        if not tags:
            raise MarketDataError(f"feed reported no expiries for {underlying.code}")

        as_of = date.today()
        slices: list[OptionSlice] = []
        warnings: list[str] = []

        for tag in tags:
            try:
                expiry_date, days = self._retry(
                    f"expiry date for {underlying.code} {tag}",
                    ak.option_sse_expire_day_sina,
                    trade_date=str(tag),
                    symbol=sina_symbol,
                    exchange="null",
                )
                board = self._retry(
                    f"board for {underlying.code} {tag}",
                    ak.option_finance_board,
                    symbol=underlying.feed_symbol,
                    end_month=str(tag)[-4:],
                )
            except MarketDataError as exc:
                # One bad expiry should not lose the whole surface.
                warnings.append(f"expiry {tag} skipped: {exc}")
                continue

            rows = _board_rows(board, underlying)
            if not rows:
                warnings.append(f"expiry {tag} skipped: board returned no usable rows")
                continue

            slices.append(
                build_slice(
                    underlying,
                    rows,
                    expiry=_parse_date(expiry_date),
                    as_of=as_of,
                    days_to_expiry=int(days),
                )
            )

        if not slices:
            raise MarketDataError(
                f"no usable expiries for {underlying.code}; " + "; ".join(warnings)
            )

        return OptionChain(
            underlying=underlying,
            as_of=as_of,
            slices=tuple(sorted(slices, key=lambda s: s.expiry)),
            source=self.name,
            warnings=tuple(warnings),
        )


def _sina_symbol(underlying: Underlying) -> str:
    """Map an underlying to the symbol the Sina expiry endpoints expect.

    That endpoint uses a short product nickname rather than the ticker or the
    long board name -- a third naming scheme for the same instrument.
    """
    nicknames = {"510050": "50ETF", "510300": "300ETF"}
    try:
        return nicknames[underlying.code]
    except KeyError:
        raise MarketDataError(
            f"no expiry-endpoint nickname registered for {underlying.code}"
        ) from None


def _board_rows(board: Any, underlying: Underlying) -> list[dict[str, object]]:
    """Convert a board DataFrame into plain row dicts, read positionally."""
    if board is None or len(board) == 0:
        return []
    if board.shape[1] < len(_BOARD_COLUMNS):
        raise MarketDataError(
            f"board for {underlying.code} has {board.shape[1]} columns, expected at "
            f"least {len(_BOARD_COLUMNS)}; the upstream format has changed"
        )

    rows: list[dict[str, object]] = []
    for values in board.itertuples(index=False, name=None):
        rows.append(
            {
                "code": str(values[1]).strip(),
                "last": float(values[2]),
                "previous_settlement": float(values[4]),
                "strike": float(values[5]),
            }
        )
    return rows


def _parse_date(value: object) -> date:
    """Parse a feed date, which arrives as ``'2026-11-25'`` or a date already."""
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise MarketDataError(f"cannot parse expiry date {value!r}")


# ---------------------------------------------------------------------------
# Offline sources
# ---------------------------------------------------------------------------


@dataclass
class SnapshotSource:
    """Replay a chain captured earlier and stored on disk.

    The default source, so tests and CI never touch the network.
    """

    directory: Path
    snapshot: str | None = None

    @property
    def name(self) -> str:
        return f"snapshot:{self.snapshot or 'latest'}"

    def fetch(self, underlying_code: str) -> OptionChain:
        from otcpricer.marketdata.snapshots import load_snapshot

        underlying = resolve_underlying(underlying_code)
        return load_snapshot(self.directory, underlying.code, name=self.snapshot)

    def available(self, underlying_code: str) -> list[str]:
        """Snapshot names on disk for this underlying, newest first."""
        from otcpricer.marketdata.snapshots import list_snapshots

        return list_snapshots(self.directory, resolve_underlying(underlying_code).code)


@dataclass
class CsvSource:
    """Read a chain from a user-supplied CSV.

    Expected columns: ``code``, ``last``, ``previous_settlement``, ``strike``,
    ``expiry``, ``days_to_expiry``. One row per contract.
    """

    path: Path

    @property
    def name(self) -> str:
        return f"csv:{self.path.name}"

    def fetch(self, underlying_code: str) -> OptionChain:
        import csv
        from collections import defaultdict

        underlying = resolve_underlying(underlying_code)
        if not self.path.exists():
            raise MarketDataError(f"CSV not found: {self.path}")

        grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
        with self.path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (str(row["expiry"]), int(row["days_to_expiry"]))
                grouped[key].append(
                    {
                        "code": row["code"],
                        "last": float(row["last"]),
                        "previous_settlement": float(row["previous_settlement"]),
                        "strike": float(row["strike"]),
                    }
                )

        if not grouped:
            raise MarketDataError(f"CSV {self.path} contained no rows")

        as_of = date.today()
        slices = [
            build_slice(
                underlying,
                rows,
                expiry=_parse_date(expiry_text),
                as_of=as_of,
                days_to_expiry=days,
            )
            for (expiry_text, days), rows in grouped.items()
        ]
        return OptionChain(
            underlying=underlying,
            as_of=as_of,
            slices=tuple(sorted(slices, key=lambda s: s.expiry)),
            source=self.name,
        )


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


@dataclass
class CachingSource:
    """Wrap a source with a time-to-live cache.

    A surface fit touches the same chain repeatedly -- once per expiry for the
    forward, again for the volatility extraction, again if the caller refits.
    Without this, each of those is a fresh round trip to a feed already shown to
    drop connections.
    """

    inner: OptionChainSource
    ttl_seconds: float = 300.0
    _cache: dict[str, tuple[float, OptionChain]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.inner.name

    def fetch(self, underlying_code: str) -> OptionChain:
        now = time.monotonic()
        cached = self._cache.get(underlying_code)
        if cached is not None and now - cached[0] < self.ttl_seconds:
            _log.debug("market data cache hit", extra={"underlying": underlying_code})
            return cached[1]

        chain = self.inner.fetch(underlying_code)
        self._cache[underlying_code] = (now, chain)
        return chain

    def invalidate(self) -> None:
        """Drop everything cached."""
        self._cache.clear()


def build_source(
    kind: str | None = None,
    *,
    snapshot_dir: Path | str | None = None,
    snapshot: str | None = None,
    csv_path: Path | str | None = None,
    ttl_seconds: float | None = None,
    wind_dir: Path | str | None = None,
    as_of: date | None = None,
) -> OptionChainSource:
    """Construct the configured source, wrapped in a cache.

    Defaults come from :func:`~otcpricer.core.config.get_settings`, whose
    ``source`` is ``"snapshot"`` -- so an unconfigured caller stays offline.

    Raises:
        MarketDataError: on an unrecognised source kind.
    """
    from otcpricer.core.config import get_settings

    settings = get_settings()
    resolved_kind = (kind or settings.data.source).lower()
    directory = Path(snapshot_dir or settings.data.snapshot_dir)
    ttl = settings.data.cache_ttl_seconds if ttl_seconds is None else ttl_seconds

    inner: OptionChainSource
    if resolved_kind == "akshare":
        inner = AkshareSource()
    elif resolved_kind == "snapshot":
        inner = SnapshotSource(directory=directory, snapshot=snapshot)
    elif resolved_kind == "csv":
        if csv_path is None:
            raise MarketDataError("source 'csv' requires csv_path")
        inner = CsvSource(path=Path(csv_path))
    elif resolved_kind == "wind":
        from otcpricer.marketdata.wind import WindPanel, WindPanelSource

        directory = Path(wind_dir or settings.data.wind_dir or "")
        if not str(directory) or str(directory) == ".":
            raise MarketDataError(
                "source 'wind' requires wind_dir, or OTCPRICER_WIND_DIR to be set"
            )
        if as_of is None:
            raise MarketDataError(
                "source 'wind' requires as_of: the panel holds 725 dates and "
                "there is no sensible default among them"
            )
        # Deliberately not wrapped in CachingSource. Its key is the underlying
        # code and its invalidation is a wall clock, both of which are wrong for
        # a static panel -- and WindPanel already caches, by date as well as by
        # code.
        return WindPanelSource(panel=WindPanel(directory), as_of=as_of)
    else:
        raise MarketDataError(
            f"unknown market data source {resolved_kind!r}; expected one of "
            f"'akshare', 'snapshot', 'csv', 'wind'"
        )

    return CachingSource(inner=inner, ttl_seconds=ttl)
