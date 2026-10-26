"""Cross-cutting foundations: types, validation, errors, dates, config, logging."""

from __future__ import annotations

from otcpricer.core.daycount import (
    CALENDAR_DAYS_PER_YEAR,
    TRADING_DAYS_PER_YEAR,
    DayCount,
    parse_tenor,
    year_fraction,
)
from otcpricer.core.exceptions import (
    ArbitrageError,
    CalibrationError,
    ConvergenceError,
    EngineError,
    MarketDataError,
    OtcPricerError,
    ParameterError,
)
from otcpricer.core.types import (
    BarrierKind,
    ExerciseStyle,
    Greeks,
    KIStrikeMode,
    OptionType,
    ProductType,
)

__all__ = [
    "CALENDAR_DAYS_PER_YEAR",
    "TRADING_DAYS_PER_YEAR",
    "ArbitrageError",
    "BarrierKind",
    "CalibrationError",
    "ConvergenceError",
    "DayCount",
    "EngineError",
    "ExerciseStyle",
    "Greeks",
    "KIStrikeMode",
    "MarketDataError",
    "OptionType",
    "OtcPricerError",
    "ParameterError",
    "ProductType",
    "parse_tenor",
    "year_fraction",
]
