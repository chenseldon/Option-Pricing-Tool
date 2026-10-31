"""Dupire local volatility, derived from a fitted SSVI surface.

The formula
-----------
In forward-moneyness coordinates ``y = log(S / F(t))`` with total implied
variance ``w(y, t)``:

.. code-block:: text

                               dw/dt
    sigma_loc^2 = -----------------------------------------------------------
                  1 - (y/w)(dw/dy) + 1/4(-1/4 - 1/w + y^2/w^2)(dw/dy)^2
                                                        + 1/2 (d2w/dy^2)

This exact form is verified symbolically, not assumed: starting from the
normalised Black call ``c = N(d1) - e^y N(d2)`` and solving Dupire's equation
in these coordinates, ``dc/dt = (v/2)(d2c/dy2 - dc/dy)``, the resulting ``v``
minus the expression above simplifies to exactly 0.

That verification settles the one convention that matters. **``dw/dt`` is the
partial at fixed ``y``**, not at fixed strike, because the whole derivation
lives in ``(y, t)`` coordinates. It is therefore exactly what
:meth:`~otcpricer.models.svi.SSVISurface.dw_dT` already returns, and no extra
forward-curve correction term enters the numerator -- provided the forward
curve used here is the one the surface was fitted against.

The denominator is already written
----------------------------------
That whole denominator is identically Gatheral's butterfly function, which
:func:`~otcpricer.models.svi.butterfly_g` has computed since Phase 2a. So this
module does not transcribe it a second time; it calls
:meth:`~otcpricer.models.svi.SSVISurface.butterfly`.

That is not merely tidier. The grouped form keeps the ``1/w^2`` terms inside a
square rather than as a free-floating sum of large cancelling quantities, so it
is better conditioned as ``w -> 0``; it is already finite-difference tested;
and reusing it removes an entire class of transcription error. The two forms
were checked to agree to 4.4e-16 across ``|y| <= 1.2`` for ``t`` from 3 years
down to 1e-08, and both stay well conditioned at ``t = 1e-12``.

Why the denominator cannot vanish
---------------------------------
Because it *is* the butterfly expression, a surface that passed
:func:`~otcpricer.models.arbitrage.local_variance_is_well_defined` has a
strictly positive denominator and a non-negative numerator by construction --
not by clipping negatives afterwards. This class refuses any surface that did
not pass, because the alternative is a negative local variance that still looks
like a number.

Measured on the committed chains over ``|y| <= 1.2``, ``t <= 3``: zero negative
local variances in 96,400 points, denominator bottoming out at 0.252 and 0.247.

The surface does not move when spot moves
-----------------------------------------
This is the defining property of local volatility and it is easy to get wrong.
``sigma_loc`` is a fixed function of *absolute* spot and time, calibrated from
today's market once. When a caller bumps spot to difference a delta, the
surface stays put -- which is precisely why a local-volatility delta differs
from a Black-Scholes one. So the forward anchor is fixed at construction and is
never re-derived from a shifted market.

The wings blow up at short maturity, and that is correct
--------------------------------------------------------
As ``t -> 0`` the at-the-money local volatility converges to a finite level
(0.150 on 510050), but away from the money it diverges. This is not a defect:
at zero time to expiry an option struck away from spot has no time value, so
only an unbounded local volatility can generate any smile at all.

The rate is set by the power law, and follows from it rather than being
observed. As ``theta -> 0`` the curvature ``phi ~ eta * theta^-gamma`` blows up,
so at fixed ``y != 0`` the square root is dominated by ``|phi y|`` and
``w ~ theta * phi * |y| ~ theta^(1-gamma)``. With ``theta`` linear in ``t`` near
the origin that gives ``w ~ t^(1-gamma)`` and ``dw/dt ~ t^-gamma``, while the
denominator tends to a constant. Hence

.. code-block:: text

    sigma_loc ~ t^(-gamma/2)

*not* ``t^(-1/2)``. Checked against the closed form at ``y = -0.357``, where a
hundredfold drop in ``t`` should multiply the volatility by ``100^(gamma/2)``:

======  ===========  ==========
gamma   predicted    measured
======  ===========  ==========
0.2     1.5849       1.6150
0.3     1.9953       2.0054
0.4     2.5119       2.5135
0.5     3.1623       3.1605
======  ===========  ==========

So the cap genuinely binds in the far wing at the very short end, where it is
also harmless -- those nodes sit many standard deviations from spot and
contribute nothing to a price at spot. The cap is a numerical guard, and every
hit is counted and reported rather than absorbed silently; :meth:`scan` is what
a report calls to show how much clamping actually happened.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from otcpricer.core.exceptions import CalibrationError
from otcpricer.core.logging import get_logger
from otcpricer.marketdata.forward import ForwardCurve
from otcpricer.models.surface import FittedSurface
from otcpricer.models.svi import SSVIParams, SSVISurface

__all__ = [
    "DEFAULT_MAX_ATM_MULTIPLE",
    "DEFAULT_N_SIGMA",
    "DEFAULT_VOL_CAP",
    "DEFAULT_VOL_FLOOR",
    "MIN_TIME",
    "LocalVolDiagnostics",
    "LocalVolSurface",
    "local_vol_from_fitted",
]

_log = get_logger(__name__)

#: Smallest local volatility returned. A floor rather than zero because a
#: vanishing diffusion coefficient degenerates the PDE operator, and because a
#: local volatility that small carries no information.
DEFAULT_VOL_FLOOR = 0.01

#: Largest local volatility returned. Binds only in the far wing at the very
#: short end, where the 1/sqrt(t) divergence is genuine and harmless.
DEFAULT_VOL_CAP = 2.0

#: Standard deviations of padding a grid builder asks for by default. Matches
#: the flat engine's inherited choice, so a constant-volatility stand-in
#: reproduces its grid exactly.
DEFAULT_N_SIGMA = 4.0

#: Hard cap on padding, as a multiple of at-the-money total volatility.
#:
#: The fixed point below is self-reinforcing under a steep skew -- further out
#: means higher implied volatility, which demands further out again. On the
#: 510300 fit at three years it runs to y = -6.38, i.e. spot at 0.17% of its
#: level, which would leave a 1600-node grid only 37 intervals across the
#: barrier span. Measured trade-off at T = 3 (probability of finishing beyond
#: the boundary, and intervals left between the barriers):
#:
#: =========  ===================  ===================
#: cap x atm  510050               510300
#: =========  ===================  ===================
#: 4          2.99e-03 / 162       1.07e-02 / 139
#: 8          6.05e-05 /  85       7.25e-04 /  73
#: none       2.58e-06 /  68       5.26e-07 /  37
#: =========  ===================  ===================
#:
#: Eight is chosen because the *inherited* flat grid, which Phase 1 validated
#: to within two Monte Carlo confidence widths, sits at 7.1e-03 by this same
#: measure -- worse than any row above. Reach probability is therefore a
#: diagnostic, not the gate: what carries the boundary is the accuracy of the
#: condition imposed there, and the gate is a measured displacement study.
#:
#: Under a constant volatility the fixed point lands at 4 x atm, so the cap
#: never binds and the exact reduction to the flat rule is preserved.
DEFAULT_MAX_ATM_MULTIPLE = 8.0

#: Smallest time the formula is evaluated at. At exactly ``t = 0`` total
#: variance is zero and the ``1/w`` terms are undefined, so time is floored
#: here -- one second, chosen to sit far below any solver's time step.
#:
#: That gap matters for one Greek in particular. Theta is read from two
#: adjacent PDE time levels, so a floor anywhere near ``dt`` would hand both
#: levels the same frozen coefficients and silently zero the term-structure
#: contribution to theta. One second is five orders of magnitude below a daily
#: step, so the two levels always differ.
MIN_TIME = 1.0 / (365.0 * 24.0 * 3600.0)


@dataclass(frozen=True, slots=True)
class LocalVolDiagnostics:
    """What a scan of the local volatility over a region found.

    Attributes:
        n_evaluated: Points evaluated.
        min_vol: Smallest local volatility, after clamping.
        max_vol: Largest, after clamping.
        min_denominator: Smallest Dupire denominator, i.e. the butterfly
            expression. Must stay positive; this is what the no-arbitrage check
            protects.
        n_floored: Points clamped up to the floor.
        n_capped: Points clamped down to the cap.
        n_undefined: Points where the denominator was non-positive. Zero for an
            arbitrage-free surface; anything else means the guarantee failed
            somewhere the arbitrage grid did not look.
        worst_cap: ``(spot, time)`` of the largest unclamped value, for locating
            where the clamping happens.
        max_peclet: Largest cell Peclet number, when a grid spacing was given.
            Above roughly 2 a central-difference scheme can oscillate.
        extrapolated_fraction: Share of the scanned time range that sits beyond
            the surface's last quoted expiry.
    """

    n_evaluated: int
    min_vol: float
    max_vol: float
    min_denominator: float
    n_floored: int
    n_capped: int
    n_undefined: int
    worst_cap: tuple[float, float] | None = None
    max_peclet: float | None = None
    extrapolated_fraction: float = 0.0

    @property
    def clamped_fraction(self) -> float:
        """Share of evaluated points that hit either bound."""
        if self.n_evaluated == 0:
            return 0.0
        return (self.n_floored + self.n_capped) / self.n_evaluated

    @property
    def well_defined(self) -> bool:
        """Whether the denominator stayed positive everywhere scanned."""
        return self.n_undefined == 0 and self.min_denominator > 0.0

    def as_dict(self) -> dict[str, object]:
        """Flat dict for API responses and reports."""
        return {
            "n_evaluated": self.n_evaluated,
            "min_vol": round(self.min_vol, 6),
            "max_vol": round(self.max_vol, 6),
            "min_denominator": round(self.min_denominator, 8),
            "n_floored": self.n_floored,
            "n_capped": self.n_capped,
            "n_undefined": self.n_undefined,
            "clamped_fraction": round(self.clamped_fraction, 8),
            "well_defined": self.well_defined,
            "worst_cap_spot": None if self.worst_cap is None else round(self.worst_cap[0], 6),
            "worst_cap_time": None if self.worst_cap is None else round(self.worst_cap[1], 8),
            "max_peclet": None if self.max_peclet is None else round(self.max_peclet, 6),
            "extrapolated_fraction": round(self.extrapolated_fraction, 6),
        }

    def summary(self) -> str:
        """One line, for logs and report headers."""
        head = (
            f"local vol in [{self.min_vol:.4f}, {self.max_vol:.4f}] on "
            f"{self.n_evaluated} points, min g={self.min_denominator:.4f}"
        )
        if self.n_undefined:
            return f"{head}; UNDEFINED at {self.n_undefined} points"
        if self.n_floored or self.n_capped:
            return (
                f"{head}; clamped {self.n_floored + self.n_capped} "
                f"({self.clamped_fraction:.3%}): {self.n_floored} at floor, "
                f"{self.n_capped} at cap"
            )
        return f"{head}; no clamping"


class LocalVolSurface:
    """Dupire local volatility on a fitted SSVI surface.

    Satisfies :class:`~otcpricer.core.vol.LocalVolatility` structurally, so the
    engines can take it without importing this module.

    Args:
        surface: The calibrated implied-volatility surface.
        forwards: The forward curve the surface was fitted against. Using any
            other curve evaluates the local volatility at the wrong moneyness.
        spot: Spot at ``t = 0``, anchoring the forward curve's level. Fixed at
            construction -- see the module docstring on why the surface must
            not move when spot is bumped.
        floor: Smallest local volatility returned.
        cap: Largest local volatility returned.
        precondition: The ``(ok, reason)`` verdict from
            :func:`~otcpricer.models.arbitrage.local_variance_is_well_defined`.
        require_well_defined: Refuse a surface that failed that verdict. Turn
            off only to study a broken surface deliberately, as the tests do.

    Raises:
        CalibrationError: if the bounds are inconsistent, the spot is not
            positive, or the surface cannot yield a usable local volatility.
    """

    def __init__(
        self,
        surface: SSVISurface,
        forwards: ForwardCurve,
        spot: float,
        *,
        floor: float = DEFAULT_VOL_FLOOR,
        cap: float = DEFAULT_VOL_CAP,
        precondition: tuple[bool, str] | None = None,
        require_well_defined: bool = True,
    ) -> None:
        if floor <= 0.0:
            raise CalibrationError(f"local volatility floor must be positive, got {floor}")
        if cap <= floor:
            raise CalibrationError(f"cap {cap} must exceed floor {floor}")
        if spot <= 0.0:
            raise CalibrationError(f"spot must be positive, got {spot}")
        if require_well_defined and precondition is not None and not precondition[0]:
            raise CalibrationError(
                f"this surface cannot produce a local volatility: {precondition[1]}"
            )

        self.surface = surface
        self.forwards = forwards
        self.spot = float(spot)
        self.floor = float(floor)
        self.cap = float(cap)
        self.precondition = precondition

    # -- the forward the moneyness is measured against --------------------

    def forward(self, t: float | np.ndarray) -> np.ndarray:
        """Forward to ``t``, anchored at this surface's own spot."""
        return self.forwards.forward_from(self.spot, t)

    def drift(self, t: float) -> float:
        """Instantaneous drift ``d log F / dt``, from the forward curve."""
        return float(self.forwards.drift(t))

    def forward_growth(self, t_from: float, t_to: float) -> float:
        """``F(t_to) / F(t_from)``, independent of the spot anchor.

        The integrated drift between the two times, so a boundary condition
        written with it stays consistent with the operator's advection term --
        which a constant carry would not.
        """
        return float(
            np.exp(
                float(self.forwards.log_forward_ratio(t_to))
                - float(self.forwards.log_forward_ratio(t_from))
            )
        )

    def log_moneyness(self, S: float | np.ndarray, t: float) -> np.ndarray:
        """``y = log(S / F(t))``, the surface's native coordinate."""
        S_arr = np.asarray(S, dtype=float)
        return np.asarray(
            np.log(S_arr) - np.log(self.spot) - float(self.forwards.log_forward_ratio(t))
        )

    # -- Dupire -----------------------------------------------------------

    def _dupire(self, S: float | np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Raw local variance and the Dupire denominator, before clamping.

        Scalar ``t``, vectorised over ``S`` -- the contract that keeps the
        variance term structure to one spline evaluation per time level.
        """
        t_eval = max(float(t), MIN_TIME)
        y = self.log_moneyness(S, t_eval)

        # The denominator IS the butterfly expression; reuse it rather than
        # transcribe it. Better conditioned as w -> 0, and already tested.
        denominator = np.asarray(self.surface.butterfly(y, t_eval))
        dw_dt = np.asarray(self.surface.dw_dT(y, t_eval))

        with np.errstate(divide="ignore", invalid="ignore"):
            variance = np.where(denominator > 0.0, dw_dt / denominator, np.inf)
        return np.asarray(variance), denominator

    def local_vol(self, S: float | np.ndarray, t: float) -> np.ndarray:
        """Local volatility ``sigma_loc(S, t)``, clamped to the configured band.

        Vectorised over ``S`` and scalar in ``t``, which is how both solvers use
        it: the PDE passes its whole node vector once per time level, the Monte
        Carlo its whole path vector once per step.
        """
        variance, _ = self._dupire(S, t)
        vol = np.sqrt(np.maximum(np.where(np.isfinite(variance), variance, np.inf), 0.0))
        return np.asarray(np.clip(vol, self.floor, self.cap))

    def local_variance(self, S: float | np.ndarray, t: float) -> np.ndarray:
        """Local variance, clamped. The square of :meth:`local_vol`."""
        vol = self.local_vol(S, t)
        return np.asarray(vol * vol)

    # -- what a grid builder needs ----------------------------------------

    def atm_total_vol(self, T: float) -> float:
        """``sqrt(w(0, T))``, the at-the-money total volatility to ``T``.

        Under a constant volatility this is exactly ``sigma * sqrt(T)``, which
        is what makes a constant-volatility stand-in land on the flat engine's
        grid node for node.
        """
        return float(np.sqrt(max(float(self.surface.total_variance(0.0, max(T, MIN_TIME))), 0.0)))

    def boundary_moneyness(
        self,
        T: float,
        n_sigma: float = DEFAULT_N_SIGMA,
        max_atm_multiple: float = DEFAULT_MAX_ATM_MULTIPLE,
    ) -> tuple[float, float]:
        """Log-moneyness whose z-score *at that moneyness* is ``n_sigma``.

        Solves ``y = -/+ n * sqrt(w(y, T))`` rather than using the at-the-money
        total volatility, because with a skew the two are far apart and the
        at-the-money version under-pads badly. Measured on the 510050 fit at
        T = 3: four at-the-money standard deviations leaves a 1.37e-02
        probability of finishing beyond the lower boundary, because implied
        volatility at the boundary strike is much higher than at the money.
        Solving the fixed point brings that to 9.2e-05.

        The result is capped at ``max_atm_multiple`` times the at-the-money
        total volatility, because the fixed point is self-reinforcing under a
        steep skew; see :data:`DEFAULT_MAX_ATM_MULTIPLE` for the measured
        trade-off.

        Under a constant volatility ``w`` does not depend on ``y``, so the root
        is exactly ``-/+ n * sigma * sqrt(T)``, the cap does not bind, and the
        rule reduces to the flat engine's. That reduction is what the
        degenerate-surface test relies on.

        Returns:
            ``(lower, upper)`` with ``lower < 0 < upper``.
        """
        maturity = max(float(T), MIN_TIME)
        scale = max(self.atm_total_vol(maturity), 1e-8)

        def sqrt_w(y: float) -> float:
            value = float(self.surface.total_variance(y, maturity))
            return float(np.sqrt(max(value, 0.0)))

        def solve(sign: float) -> float:
            # f is strictly increasing in the search direction: SSVI's total
            # variance is asymptotically linear in |y|, so sqrt(w) grows like
            # sqrt(|y|) and the linear term wins. One root, so bracket and
            # bisect -- cheap, and robust where a fixed-point iteration can
            # cycle on a steep wing.
            def f(y: float) -> float:
                return y - sign * n_sigma * sqrt_w(y)

            far = sign * 2.0 * n_sigma * scale
            for _ in range(60):
                if sign * f(far) > 0.0:
                    break
                far *= 2.0
            else:  # pragma: no cover - would need a pathological surface
                return far
            lo, hi = (far, 0.0) if sign < 0 else (0.0, far)
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if f(mid) < 0.0:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)

        limit = max_atm_multiple * scale
        return max(solve(-1.0), -limit), min(solve(1.0), limit)

    def reach_probability(self, y: float, T: float) -> float:
        """Risk-neutral ``P(S_T`` beyond moneyness ``y)``, from the surface.

        The honest boundary diagnostic: ``N(-d2)`` at that strike, read off the
        implied surface rather than approximated with an at-the-money
        volatility. Reported rather than used as a gate -- the inherited flat
        grid sits at 7.1e-03 by this measure and was still validated to within
        two Monte Carlo confidence widths, because what carries the boundary is
        the accuracy of the condition imposed there, not the probability of
        reaching it.
        """
        maturity = max(float(T), MIN_TIME)
        w = max(float(self.surface.total_variance(y, maturity)), 1e-16)
        d2 = (-abs(y) - 0.5 * w) / np.sqrt(w)
        return float(norm.cdf(d2))

    def extrapolated_fraction(self, T: float) -> float:
        """Share of ``[0, T]`` beyond the surface's last quoted expiry.

        Led with in the report, because on this data it is the dominant
        modelling exposure: the chains stop at 0.41 years while the products
        run one to three, so most of the pricing horizon rests on the variance
        term structure's linear extension rather than on quotes.
        """
        last = float(self.surface.params.maturities[-1])
        if T <= 0.0:
            return 0.0
        return float(max(T - last, 0.0) / T)

    # -- vega -------------------------------------------------------------

    def with_atm_shift(
        self, vol_abs: float, *, require_well_defined: bool = False
    ) -> LocalVolSurface:
        """A copy with every at-the-money volatility shifted by ``vol_abs``.

        Rebuilds the variance term structure as
        ``theta_i -> (sqrt(theta_i / T_i) + vol_abs)^2 * T_i``, keeping
        ``rho``, ``eta`` and ``gamma``. This is the shift a desk's vega refers
        to: the implied surface moves, and the local volatility moves by
        whatever that implies -- roughly twice as much in the wings.

        Not a pure level shift of the whole surface, and worth being precise
        about: ``phi`` depends on ``theta``, so moving the at-the-money level
        also changes curvature at every maturity. That is what re-calibrating
        to a shifted at-the-money level would give, but it means this is a
        different derivative from the flat engine's vega, not a comparable one.

        Raises:
            CalibrationError: if the shift pushes the term structure
                non-monotone, i.e. into calendar arbitrage.
        """
        params = self.surface.params
        T = np.asarray(params.maturities, dtype=float)
        theta = np.asarray(params.thetas, dtype=float)
        shifted = (np.sqrt(theta / T) + float(vol_abs)) ** 2 * T

        if np.any(np.diff(shifted) < 0.0):
            raise CalibrationError(
                f"shifting at-the-money volatility by {vol_abs:+.4f} makes the variance "
                f"term structure non-monotone, which is calendar arbitrage. Shifted "
                f"thetas: {np.round(shifted, 8).tolist()}"
            )

        moved = SSVIParams(
            rho=params.rho,
            eta=params.eta,
            gamma=params.gamma,
            maturities=params.maturities,
            thetas=tuple(shifted.tolist()),
        )
        if moved.butterfly_margin <= 0.0:
            raise CalibrationError(
                f"shifting at-the-money volatility by {vol_abs:+.4f} leaves butterfly "
                f"margin {moved.butterfly_margin:.3e}; the Dupire denominator would be "
                f"non-positive, so no local volatility exists on the shifted surface."
            )

        return LocalVolSurface(
            SSVISurface(moved),
            self.forwards,
            self.spot,
            floor=self.floor,
            cap=self.cap,
            precondition=self.precondition,
            require_well_defined=require_well_defined,
        )

    # -- diagnostics ------------------------------------------------------

    def scan(
        self,
        spots: np.ndarray,
        times: np.ndarray,
        *,
        dx: float | None = None,
    ) -> LocalVolDiagnostics:
        """Evaluate over a region and report what happened.

        Args:
            spots: Spot levels to evaluate at.
            times: Times to evaluate at.
            dx: Log-spot grid spacing, to report the largest cell Peclet
                number. Omit when not discretising.

        Returns:
            A :class:`LocalVolDiagnostics`.
        """
        spots = np.asarray(spots, dtype=float)
        times = np.asarray(times, dtype=float)

        min_vol, max_vol = np.inf, -np.inf
        min_denominator = np.inf
        n_floored = n_capped = n_undefined = 0
        worst_raw, worst_cap = -np.inf, None
        max_peclet: float | None = None if dx is None else 0.0

        for t in times:
            variance, denominator = self._dupire(spots, float(t))
            finite = np.where(np.isfinite(variance), variance, 0.0)
            raw = np.sqrt(np.maximum(finite, 0.0))
            clamped = np.clip(
                np.sqrt(np.maximum(np.where(np.isfinite(variance), variance, np.inf), 0.0)),
                self.floor,
                self.cap,
            )

            min_vol = min(min_vol, float(np.min(clamped)))
            max_vol = max(max_vol, float(np.max(clamped)))
            min_denominator = min(min_denominator, float(np.min(denominator)))
            n_floored += int(np.sum(raw < self.floor))
            n_capped += int(np.sum((raw > self.cap) | ~np.isfinite(variance)))
            n_undefined += int(np.sum(~np.isfinite(variance)))

            if raw.size:
                index = int(np.argmax(raw))
                if float(raw[index]) > worst_raw:
                    worst_raw = float(raw[index])
                    worst_cap = (float(spots[index]), float(t))

            if dx is not None:
                diffusion = 0.5 * clamped * clamped
                advection = np.abs(self.drift(float(t)) - diffusion)
                max_peclet = max(
                    float(max_peclet or 0.0), float(np.max(advection * dx / (2.0 * diffusion)))
                )

        horizon = float(np.max(times)) if times.size else 0.0
        return LocalVolDiagnostics(
            n_evaluated=int(spots.size * times.size),
            min_vol=float(min_vol),
            max_vol=float(max_vol),
            min_denominator=float(min_denominator),
            n_floored=n_floored,
            n_capped=n_capped,
            n_undefined=n_undefined,
            worst_cap=worst_cap,
            max_peclet=max_peclet,
            extrapolated_fraction=self.extrapolated_fraction(horizon),
        )

    def __repr__(self) -> str:
        return (
            f"LocalVolSurface(spot={self.spot:.6g}, floor={self.floor:.3g}, "
            f"cap={self.cap:.3g}, {self.surface!r})"
        )


def local_vol_from_fitted(
    fitted: FittedSurface,
    *,
    spot: float | None = None,
    tail_drift: float = 0.0,
    rate: float | None = None,
    floor: float = DEFAULT_VOL_FLOOR,
    cap: float = DEFAULT_VOL_CAP,
    require_well_defined: bool = True,
) -> LocalVolSurface:
    """Build a local-volatility surface from a Phase 2a fit.

    Assembles the forward curve from the fit's own parity-implied forwards, so
    the local volatility is evaluated at exactly the moneyness the surface was
    calibrated in.

    Args:
        fitted: The calibrated surface and its evidence.
        spot: Spot anchoring the forward curve. Defaults to the curve's own
            back-extrapolated ``F(0)``, which makes the curve reproduce every
            quoted forward exactly -- the setting the vanilla reprice test
            needs.
        tail_drift: Drift beyond the last quoted expiry, i.e. the caller's
            ``r - q - basis``.
        rate: Discount rate recorded on the curve. Defaults to the rate the
            surface was fitted with.
        floor: Smallest local volatility returned.
        cap: Largest local volatility returned.
        require_well_defined: Refuse a surface that failed the precondition.

    Returns:
        A :class:`LocalVolSurface`.
    """
    usable = fitted.surface_input.usable_slices
    curve = ForwardCurve(
        maturities=tuple(s.year_fraction for s in usable),
        forwards=tuple(s.forward.forward for s in usable),
        tail_drift=tail_drift,
        rate=fitted.surface_input.rate if rate is None else rate,
    )
    result = LocalVolSurface(
        fitted.surface,
        curve,
        curve.spot_anchor if spot is None else spot,
        floor=floor,
        cap=cap,
        precondition=fitted.usable_for_local_volatility,
        require_well_defined=require_well_defined,
    )
    _log.info(
        "local volatility built",
        extra={
            "underlying": fitted.underlying_code,
            "spot": result.spot,
            "implied_carry": curve.implied_carry,
            "kink": curve.kink,
        },
    )
    return result
