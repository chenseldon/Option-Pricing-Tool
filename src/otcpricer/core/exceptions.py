"""Exception hierarchy for the pricing library.

Every error raised by otcpricer derives from :class:`OtcPricerError`, so a
caller can trap the whole library with one ``except`` and an API layer can map
error classes onto HTTP status codes without string matching.
"""

from __future__ import annotations

__all__ = [
    "OtcPricerError",
    "ParameterError",
    "ArbitrageError",
    "ConvergenceError",
    "CalibrationError",
    "MarketDataError",
    "EngineError",
]


class OtcPricerError(Exception):
    """Base class for every error this library raises."""


class ParameterError(OtcPricerError, ValueError):
    """A trade or model input is outside its admissible domain.

    Also derives from ``ValueError`` so existing callers that catch
    ``ValueError`` around the legacy pricing functions keep working.
    """


class ArbitrageError(OtcPricerError):
    """Inputs imply a static arbitrage.

    Raised when a quoted price sits outside no-arbitrage bounds, or when a
    fitted volatility surface violates a butterfly or calendar-spread
    condition.
    """


class ConvergenceError(OtcPricerError):
    """An iterative solver failed to reach its tolerance."""


class CalibrationError(OtcPricerError):
    """A model could not be fitted to the supplied market quotes."""


class MarketDataError(OtcPricerError):
    """Market data could not be retrieved, parsed, or validated."""


class EngineError(OtcPricerError):
    """A pricing engine was asked for something it cannot do.

    For example: an analytic engine asked for an American exercise, or a
    product/engine pair that was never wired together.
    """
