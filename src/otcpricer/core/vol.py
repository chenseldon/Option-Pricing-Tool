"""The volatility interface the engines price against.

Why this is in ``core`` and not next to the implementation
----------------------------------------------------------
``engines`` imports from ``core`` and ``products`` only -- never from ``models``
or ``marketdata``. That is deliberate: a solver should depend on the shape of a
volatility, not on how one was fitted or where its quotes came from.

The concrete Dupire surface lives in :mod:`otcpricer.models.local_vol` and
depends on ``marketdata`` for its forward curve. Typing the engines against a
structural :class:`Protocol` declared here keeps that dependency out of
``engines`` entirely: the implementation satisfies the protocol without
inheriting from it, so no import crosses the layer.

What this is *not*
------------------
This is not an abstraction over "any volatility model", and specifically not
the seam a Heston implementation would plug into. Heston's variance is a second
*state variable*, not a function of ``(S, t)``: it needs a two-dimensional PDE
with a cross-derivative term and an ADI solver, and two correlated factors in
the Monte Carlo. A protocol shaped like "give me a volatility at this spot and
time" cannot express that, and pretending otherwise would buy a false sense of
extensibility.

So flat volatility is *not* routed through here either. The engines keep their
original scalar code paths untouched and take a local volatility as an optional
keyword argument. That makes the flat results identical by construction --
the same bytes execute -- rather than identical by assumption and verified only
by running the tests one is trying not to break.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["LocalVolatility"]


@runtime_checkable
class LocalVolatility(Protocol):
    """A volatility that varies with spot and time, plus the drift it implies.

    The drift belongs here rather than on the market object because a local
    volatility derived from a surface carries that surface's own forward curve,
    and using any other drift would evaluate the volatility at the wrong
    moneyness.
    """

    def local_vol(self, S: float | np.ndarray, t: float) -> np.ndarray:
        """Local volatility at spot levels ``S`` and a single time ``t``.

        **Vectorised over ``S``, scalar in ``t``.** This is a contract, not an
        accident, and it is what keeps the solvers fast: the variance term
        structure depends only on ``t``, so one call per time level costs one
        spline evaluation for the whole grid. Calling with a meshgrid instead
        would turn thousands of spline evaluations into millions.
        """
        ...

    def drift(self, t: float) -> float:
        """Instantaneous drift ``d log F / dt`` at time ``t``."""
        ...

    def forward_growth(self, t_from: float, t_to: float) -> float:
        """``F(t_to) / F(t_from)``, the forward growth factor between two times.

        Boundary conditions are discounted forward intrinsics, so they need
        this. Expressed as a ratio rather than as a level because it is the
        ratio that is scale-free, and because a boundary should not have to
        know what spot the surface was anchored at.
        """
        ...

    def atm_total_vol(self, T: float) -> float:
        """At-the-money total volatility to ``T``, i.e. ``sqrt(w(0, T))``.

        The scalar a grid builder uses where a flat model would use
        ``sigma * sqrt(T)``. Under a constant volatility the two coincide
        exactly, which is what lets a constant-volatility stand-in reproduce
        the flat engine's grid node for node.
        """
        ...

    def boundary_moneyness(self, T: float, n_sigma: float) -> tuple[float, float]:
        """Log-moneyness bounds whose z-score at the boundary is ``n_sigma``.

        Returns ``(lower, upper)`` with ``lower < 0 < upper``.
        """
        ...

    def with_atm_shift(self, vol_abs: float) -> LocalVolatility:
        """A copy with every at-the-money *implied* volatility moved by ``vol_abs``.

        Vega lives here rather than on the market object, and that is the whole
        point. A desk hedges the implied surface, so the sensitivity that
        matters is "move the quoted volatilities and reprice" -- which moves
        the local volatility by whatever that implies, roughly twice as much in
        the wings.

        Bumping ``market.sigma`` instead would do nothing to the diffusion on
        this path, since the surface supplies it. Worse than nothing: ``sigma``
        also sets the flat grid's padding, so the bump would change the grid
        and return a small number produced by discretisation. A plausible,
        meaningless vega is harder to catch than a zero.
        """
        ...

    def with_carry_shift(self, carry_abs: float) -> LocalVolatility:
        """A copy whose drift reflects a shifted carry, for ``carry_rho``.

        Distinct from :meth:`with_rate_shift` in exactly one respect, and it is
        the respect that matters: a carry shift moves the *drift* without
        moving the rate. ``ForwardCurve`` stores both, and its
        :meth:`~otcpricer.marketdata.forward.ForwardCurve.carry` is
        ``rate - drift`` -- so shifting both together, as a rate shift does,
        leaves the carry unchanged and would measure nothing at all.

        Which is the trap this method exists to avoid. On this path the drift
        comes from the surface rather than from ``market``, so bumping
        ``market.basis_pa`` alone does not reach the diffusion, and the Greek
        would come back near zero: a plausible wrong answer rather than an
        obvious one, which is the same failure mode
        :meth:`with_atm_shift` documents for vega.

        Only the extrapolated tail moves, for the same reason as there: the
        forwards inside the quoted range were implied from the option chain and
        are observations, not functions of a hypothetical carry.
        """
        ...

    def with_rate_shift(self, rate_abs: float) -> LocalVolatility:
        """A copy whose drift reflects a shifted rate, for rho.

        Needed for the same reason as the shift above: on this path the drift
        comes from the surface's forward curve, not from ``market``, so bumping
        the market rate alone would move only the discounting and return a
        rho that is missing its drift half. Measured on a one-year snowball
        with a constant surface, that omission turned +0.0036 into -0.0001 --
        not a small distortion but a sign change.

        What moves and what does not is the whole content of the method.
        Forwards *observed* from the option chain do not move when a
        hypothetical rate does; they are data. Only the extrapolated tail,
        which was built from the caller's own ``r - q - basis``, moves with it.
        """
        ...
