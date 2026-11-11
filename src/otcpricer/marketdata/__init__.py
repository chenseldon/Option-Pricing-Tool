"""Market data adapters: live feed, offline snapshots, CSV import."""

from __future__ import annotations

from otcpricer.marketdata.contracts import (
    SUPPORTED_UNDERLYINGS,
    ContractCode,
    Underlying,
    parse_contract_code,
    resolve_underlying,
)
from otcpricer.marketdata.quotes import OptionChain, OptionQuote, OptionSlice
from otcpricer.marketdata.snapshots import (
    capture_snapshot,
    list_snapshots,
    load_snapshot,
    save_snapshot,
)
from otcpricer.marketdata.sources import (
    AkshareSource,
    CachingSource,
    CsvSource,
    OptionChainSource,
    SnapshotSource,
    build_source,
)

__all__ = [
    "SUPPORTED_UNDERLYINGS",
    "AkshareSource",
    "CachingSource",
    "ContractCode",
    "CsvSource",
    "OptionChain",
    "OptionChainSource",
    "OptionQuote",
    "OptionSlice",
    "SnapshotSource",
    "Underlying",
    "build_source",
    "capture_snapshot",
    "list_snapshots",
    "load_snapshot",
    "parse_contract_code",
    "resolve_underlying",
    "save_snapshot",
]
