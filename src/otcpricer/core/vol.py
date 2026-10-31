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
