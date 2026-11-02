"""End-to-end volatility surface construction, from a ticker to a fitted surface.

One call runs the whole chain: fetch quotes, imply a forward per expiry, solve
and screen implied volatilities, calibrate SSVI under the no-arbitrage
constraints, and verify the result independently on a dense grid.

The returned object carries the evidence as well as the answer. A surface is an
*estimate* built on non-synchronous last-trade prices with no bid/ask, so
"here is the surface" is not a useful deliverable on its own -- how well it
fits, which quotes were discarded and why, how much arbitrage headroom it has,
and whether a local volatility can be derived from it all have to travel with
it. That is what makes the difference between a number and a number you can
act on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from otcpricer.core.exceptions import CalibrationError
from otcpricer.core.logging import get_logger
from otcpricer.marketdata.forward import DEFAULT_RATE
from otcpricer.marketdata.quotes import OptionChain
from otcpricer.marketdata.rates import RateCurve
from otcpricer.marketdata.sources import OptionChainSource, build_source
from otcpricer.marketdata.surface_input import SurfaceInput, extract_surface_input
from otcpricer.models.arbitrage import (
    ArbitrageReport,
    check_surface,
    local_variance_is_well_defined,
)
from otcpricer.models.svi import SSVISurface
from otcpricer.models.svi_calibration import CalibrationResult, calibrate_ssvi

__all__ = ["FittedSurface", "fit_surface", "fit_surface_from_chain"]

_log = get_logger(__name__)

#: Fewest usable expiries to attempt a surface. Two is the minimum at which a
#: variance term structure exists at all; the calendar condition is vacuous
#: with one.
MIN_USABLE_SLICES = 2


@dataclass(frozen=True, slots=True)
class FittedSurface:
    """A calibrated surface, with everything needed to judge it.

    Attributes:
        underlying_code: Exchange ticker.
        as_of: Capture date of the underlying data.
        source: Where the quotes came from.
        surface_input: The screened inputs, including every rejection and its
            reason.
        calibration: Fit quality, per slice and pooled, in volatility points.
        arbitrage: Independent pointwise check on a dense grid.
        forwards: Implied forward per expiry tag.
    """

    underlying_code: str
    as_of: str
    source: str
    surface_input: SurfaceInput
    calibration: CalibrationResult
    arbitrage: ArbitrageReport
    forwards: dict[str, float]

    @property
    def surface(self) -> SSVISurface:
        """The fitted surface itself."""
        return self.calibration.surface

    @property
    def usable_for_local_volatility(self) -> tuple[bool, str]:
        """Whether Dupire local variance can be derived, and why or why not.

        The precondition Phase 2b depends on, surfaced here so a caller does
        not have to reason from the arbitrage report themselves.
        """
        return local_variance_is_well_defined(self.arbitrage)

    def implied_vol(self, strike: float, expiry_tag: str) -> float:
        """Implied volatility for a strike at one of the quoted expiries.

        Uses that expiry's own implied forward, so the moneyness is consistent
        with how the surface was fitted.
        """
        for slice_input in self.surface_input.slices:
            if slice_input.expiry_tag == expiry_tag:
                k = float(np.log(strike / slice_input.forward.forward))
                return float(self.surface.implied_vol(k, slice_input.year_fraction))
        available = ", ".join(s.expiry_tag for s in self.surface_input.slices)
        raise CalibrationError(f"no expiry {expiry_tag!r}; available: {available}")

    def as_dict(self) -> dict[str, object]:
        """Flat dict for API responses and reports."""
        ok, reason = self.usable_for_local_volatility
        return {
            "underlying": self.underlying_code,
            "as_of": self.as_of,
            "source": self.source,
            "forwards": {tag: round(f, 6) for tag, f in self.forwards.items()},
            "input": self.surface_input.as_dict(),
            "calibration": self.calibration.as_dict(),
            "arbitrage": self.arbitrage.as_dict(),
            "local_volatility_ready": ok,
            "local_volatility_note": reason,
        }


def fit_surface_from_chain(
    chain: OptionChain,
    *,
    rate: float = DEFAULT_RATE,
    rate_curve: RateCurve | None = None,
    min_usable_slices: int = MIN_USABLE_SLICES,
) -> FittedSurface:
    """Fit a surface to an already-fetched chain.

    Separate from :func:`fit_surface` so a caller holding a chain -- a replayed
    snapshot, a CSV, a synthetic chain in a test -- can fit it without going
    back to a source.

    Raises:
        CalibrationError: if too few expiries survive screening, with the
            per-slice reasons attached so the cause is visible rather than
            needing to be rediscovered.
    """
    surface_input = extract_surface_input(chain, rate=rate, rate_curve=rate_curve)
    usable = surface_input.usable_slices

    if len(usable) < min_usable_slices:
        detail = "; ".join(f"{s.expiry_tag}: {s.reason or 'usable'}" for s in surface_input.slices)
        raise CalibrationError(
            f"only {len(usable)} of {len(surface_input.slices)} expiries survived "
            f"screening for {chain.underlying.code}, need {min_usable_slices} for a "
            f"variance term structure. Per expiry: {detail}"
        )

    calibration = calibrate_ssvi(
        [s.year_fraction for s in usable],
        [np.array([p.log_moneyness for p in s.points]) for s in usable],
        [np.array([p.implied_vol for p in s.points]) for s in usable],
        [np.array([p.vega for p in s.points]) for s in usable],
    )
    arbitrage = check_surface(calibration.surface)

    result = FittedSurface(
        underlying_code=chain.underlying.code,
        as_of=chain.as_of.isoformat(),
        source=chain.source,
        surface_input=surface_input,
        calibration=calibration,
        arbitrage=arbitrage,
        forwards={s.expiry_tag: s.forward.forward for s in surface_input.slices},
    )
    _log.info(
        "surface fitted",
        extra={
            "underlying": chain.underlying.code,
            "rms_vol_error": calibration.rms_vol_error,
            "slices": len(usable),
            "points": calibration.n_points,
            "arbitrage": arbitrage.summary(),
        },
    )
    return result


def fit_surface(
    underlying_code: str,
    *,
    source: OptionChainSource | None = None,
    source_kind: str | None = None,
    snapshot: str | None = None,
    rate: float = DEFAULT_RATE,
    rate_curve: RateCurve | None = None,
) -> FittedSurface:
    """Fetch a chain and fit a surface to it.

    Args:
        underlying_code: Exchange ticker, e.g. ``"510050"``.
        source: An explicit source. Built from configuration when omitted,
            which defaults to offline snapshots.
        source_kind: ``"akshare"``, ``"snapshot"`` or ``"csv"``, overriding
            configuration.
        snapshot: Snapshot name, when replaying.
        rate: Continuously compounded rate for the forward and discounting.
            Ignored when ``rate_curve`` is given.
        rate_curve: A term structure to read each slice's rate off, at that
            slice's own tenor.

    Returns:
        A :class:`FittedSurface`.
    """
    resolved = source or build_source(source_kind, snapshot=snapshot)
    return fit_surface_from_chain(resolved.fetch(underlying_code), rate=rate, rate_curve=rate_curve)
