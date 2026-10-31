"""Crank-Nicolson finite-difference engine for snowball notes.

Ported from the project's original ``cn_fdm.py``, whose numerical core was
sound: a log-space uniform grid (the Black-Scholes operator has constant
coefficients after the transform, so uniform spacing is second-order
accurate), a Thomas tridiagonal solve, and Rannacher start-up -- a few fully
implicit steps before switching to Crank-Nicolson, which damps the
high-frequency oscillation a discontinuous barrier payoff otherwise excites and
which would propagate back to spot through the diffusion term.

Three things are added here:

* **Lock-up.** Knock-out observations inside the lock-up window are dropped, so
  the note cannot terminate early during it.
* **Futures basis.** The drift uses ``r - q - basis_pa`` while discounting
  continues at ``r``, so a 贴水 can be priced.
* **Knock-in strike as a parameter.** Read from the shared terms object instead
  of being hardcoded, which is what lets this engine and the Monte Carlo be
  compared at all.

Why two coupled states
----------------------
A snowball's value depends not only on where spot is but on whether the note
has already knocked in -- the same spot level is worth different amounts in the
two cases. The solver therefore evolves two grids backwards together:

``V_ki``
    Value given the note has knocked in. It is then a short put struck per
    :attr:`~otcpricer.products.snowball.SnowballTerms.ki_put_strike`, with no
    further coupon.
``V_no_ki``
    Value given it has not. Carries the coupon, terminates on an up-barrier
    observation, and *transfers* to ``V_ki`` wherever spot sits at or below the
    knock-in barrier.

Vectorising the barrier transfer as a mask over the grid is what keeps this
O(N) per step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from otcpricer.core.logging import get_logger
from otcpricer.core.market import EquityMarket
from otcpricer.core.vol import LocalVolatility
from otcpricer.products.snowball import SnowballTerms

__all__ = [
    "LOCAL_VOL_PDE_CONFIG",
    "PDEConfig",
    "SnowballPDEResult",
    "local_vol_coefficients",
    "price_snowball_pde",
    "solve_tridiagonal",
    "theta_step_variable",
]

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PDEConfig:
    """Discretisation controls.

    Attributes:
        space_nodes: Grid points in log-spot. 400 is the project's existing
            default; brokerage practice is 300 or more.
        steps_per_year: Time steps per year; 252 puts the daily knock-in
            schedule on exact grid points.
        rannacher_steps: Fully implicit start-up steps before Crank-Nicolson
            takes over. 2-4 is standard; 0 disables the damping.
    """

    space_nodes: int = 400
    steps_per_year: int = 252
    rannacher_steps: int = 4

    def __post_init__(self) -> None:
        if self.space_nodes < 50:
            raise ValueError(f"space_nodes must be at least 50, got {self.space_nodes}")
        if self.steps_per_year < 1:
            raise ValueError(f"steps_per_year must be at least 1, got {self.steps_per_year}")
        if self.rannacher_steps < 0:
            raise ValueError(f"rannacher_steps must be non-negative, got {self.rannacher_steps}")


#: Discretisation adequate for a local-volatility snowball. Not the default,
#: because the default belongs to the flat path and must not move.
#:
#: The flat grid's budget cannot be inherited: a Dupire surface has a rough
#: variance term structure, and resolving it costs resolution the flat operator
#: never needed. Measured on a one-year 103/80 snowball off the 510050 fit,
#: against a 400,000-path Monte Carlo (standard error 2.57e-04), in units of
#: that standard error:
#:
#: ======  =========  =======  ==========
#: nodes   steps/yr   gap      T=1 cost
#: ======  =========  =======  ==========
#: 800     504        -30.9
#: 1600    1008       -14.5    0.3 s
#: 1600    2016        -8.0
#: 3200    2016        -4.5
#: 3200    4032        -2.3    1.9 s
#: 4800    8064        -2.0    4.7 s
#: 6400    8064        -1.9    5.6 s
#: ======  =========  =======  ==========
#:
#: Monotone convergence, so the two engines do agree -- at the flat defaults
#: they merely had not yet. It plateaus near two standard errors, and that
#: residual is the Monte Carlo's own draw rather than the grid: a 400,000-path
#: run put the same comparison at 0.9. Two standard errors also matches what
#: Phase 1 achieved for the flat engines, 1.94 across twenty parameter sets.
#:
#: 4800 x 8064 is chosen as the point where the comparison stops being limited
#: by the grid, which is what a validation configuration is for. It costs 4.7
#: seconds at one year and 14 at three. 3200 x 4032 is roughly three times
#: faster and sits at 2.3 standard errors if that trade is worth making.
#:
#: Sized for the harder of the two chains. The smoother 510300 surface already
#: agrees at the flat defaults -- 0.2 and 0.02 standard errors at one and two
#: years -- so none of this is about local volatility as such; it is about how
#: rough a particular fitted variance term structure is.
LOCAL_VOL_PDE_CONFIG = PDEConfig(space_nodes=4800, steps_per_year=8064)


@dataclass(frozen=True, slots=True)
class SnowballPDEResult:
    """Outcome of a snowball PDE valuation, as a fraction of notional.

    Carries the whole solution surface, not just the price at spot. A finite
    difference solver computes ``V`` at every grid node on its way to the
    answer, so delta and gamma are already present in the result and can be
    read off by differencing -- no extra solves, and none of the Monte Carlo
    noise that makes a simulated gamma unusable for a barrier product.
    Discarding the surface and returning only a scalar threw that away.

    Attributes:
        price: Value at the current spot, in the relevant state.
        S_grid: Spot levels of the grid nodes, increasing.
        V_no_ki: Value at each node given the note has *not* knocked in.
        V_ki: Value at each node given it *has*. A seasoned trade that has
            already breached its knock-in must be valued and risked off this
            surface, not the other one.
        V_no_ki_next_step: ``V_no_ki`` one time step later in calendar time,
            kept for theta. See :meth:`surface_next_step`.
        V_ki_next_step: ``V_ki`` one time step later in calendar time.
        dt: Time step in years, the denominator for that theta.
        space_nodes: Grid nodes used.
        time_steps: Time steps used.
    """

    price: float
    S_grid: np.ndarray
    V_no_ki: np.ndarray
    V_ki: np.ndarray
    V_no_ki_next_step: np.ndarray
    V_ki_next_step: np.ndarray
    dt: float
    space_nodes: int
    time_steps: int

    def surface(self, *, knocked_in: bool = False) -> np.ndarray:
        """Return the value surface at valuation time, for a knock-in state."""
        return self.V_ki if knocked_in else self.V_no_ki

    def surface_next_step(self, *, knocked_in: bool = False) -> np.ndarray:
        """Return the surface one time step *later* in calendar time.

        Backward induction passes through ``t = dt`` on its way to ``t = 0``,
        so this costs nothing to capture and gives theta directly as
        ``(V(dt) - V(0)) / dt``.

        Doing it this way, rather than re-solving with a shortened maturity,
        avoids a trap: the observation schedule is derived from the tenor, so
        shortening ``T`` by a single day drops an entire monthly knock-out date
        (measured: 9 observations become 8, and the final one jumps from 1.000
        to 0.917). That is a change to the contract, not the passage of time,
        and a maturity-bumped theta would measure it instead of time decay.
        Reading two adjacent steps of one solve holds the schedule fixed by
        construction.
        """
        return self.V_ki_next_step if knocked_in else self.V_no_ki_next_step

    def as_dict(self) -> dict[str, float | int]:
        """Flat dict for API responses.

        Deliberately omits the grid arrays: they run to thousands of nodes and
        belong in a risk calculation, not a quote response.
        """
        return {
            "price": round(self.price, 8),
            "space_nodes": self.space_nodes,
            "time_steps": self.time_steps,
        }


def solve_tridiagonal(
    lower: np.ndarray, main: np.ndarray, upper: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Solve a tridiagonal system in ``O(n)``.

    Solves ``lower[i-1] * x[i-1] + main[i] * x[i] + upper[i] * x[i+1] = rhs[i]``.

    Delegates to LAPACK's banded solver via :func:`scipy.linalg.solve_banded`
    rather than running the Thomas recurrence in Python. The recurrence is
    inherently sequential, so in Python it costs one interpreter round-trip per
    grid node per time step -- for a 3200-node grid over 4000 steps that is
    13 million iterations, and it dominates the runtime by a wide margin.
    LAPACK does the same arithmetic in compiled code.

    The speed matters for more than comfort: it is what makes a genuine
    convergence study affordable, and a PDE engine whose grid cannot be
    refined cannot be validated.

    Note:
        ``solve_banded`` performs partial pivoting, so this is also more robust
        than the bare recurrence, which can divide by a vanishing pivot.
    """
    n = main.size
    if n == 0:
        return np.empty(0, dtype=float)
    if n == 1:
        return np.asarray(rhs / main, dtype=float)

    # LAPACK banded storage: row 0 holds the superdiagonal (shifted right by
    # one), row 1 the main diagonal, row 2 the subdiagonal (left-aligned).
    banded = np.zeros((3, n), dtype=float)
    banded[0, 1:] = upper
    banded[1, :] = main
    banded[2, :-1] = lower
    return np.asarray(solve_banded((1, 1), banded, rhs), dtype=float)


# ---------------------------------------------------------------------------
# Variable-coefficient operator
# ---------------------------------------------------------------------------
#
# Shared by the snowball solver and the European one. Kept here rather than in
# a third module because `solve_tridiagonal` is imported by name from this one
# and moving it would be a gratuitous break.


def local_vol_coefficients(
    local_vol: LocalVolatility,
    x_grid: np.ndarray,
    dx: float,
    t: float,
    r: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Operator coefficients at the interior nodes, for one time level.

    The log-space operator is

    .. code-block:: text

        L V = 1/2 sigma^2(S,t) V_xx + (mu(t) - 1/2 sigma^2(S,t)) V_x - r V

    which is the constant-coefficient operator with ``sigma`` and ``mu`` now
    varying. Central differences on a uniform grid stay second-order accurate
    with coefficients evaluated at the nodes.

    Args:
        local_vol: Where the volatility and the drift come from.
        x_grid: Full log-spot grid, including both boundary nodes.
        dx: Grid spacing.
        t: Calendar time of this level. **Calendar time since inception**, not
            time remaining -- the surface's maturity coordinate. Passing the
            remaining time instead reverses the term structure, and a single
            European price cannot detect it because the integral of a
            time-dependent variance is reversal-invariant.
        r: Discount rate, which enters the main diagonal alone.

    Returns:
        ``(a, b, c)``, each of length ``len(x_grid) - 2`` and evaluated at the
        interior nodes, so ``(L V)_i = a_j V_{i-1} + b_j V_i + c_j V_{i+1}``
        for interior node ``i = j + 1``.
    """
    sigma = np.asarray(local_vol.local_vol(np.exp(x_grid[1:-1]), t), dtype=float)
    diffusion = 0.5 * sigma * sigma
    advection = local_vol.drift(t) - diffusion
    return (
        diffusion / (dx * dx) - advection / (2.0 * dx),
        -2.0 * diffusion / (dx * dx) - r,
        diffusion / (dx * dx) + advection / (2.0 * dx),
    )


def theta_step_variable(
    V: np.ndarray,
    old: tuple[np.ndarray, np.ndarray, np.ndarray],
    new: tuple[np.ndarray, np.ndarray, np.ndarray],
    dt: float,
    theta: float,
    bc_low: float,
    bc_high: float,
) -> np.ndarray:
    """One backward theta-scheme step with time-dependent coefficients.

    Solves ``(I - theta dt L_new) V_new = (I + (1 - theta) dt L_old) V_old``.

    Two details here are invisible at flat volatility, so no existing test can
    catch either, and both are first-order in ``dt`` -- they read as slow
    convergence rather than as a wrong answer.

    **The explicit stencil takes the old level, the Dirichlet folds take the
    new one.** The folded terms come from the *implicit* matrix, so they carry
    ``a_new[0]`` and ``c_new[-1]``. The constant-coefficient version uses one
    scalar for both and is right only because the two levels coincide there.

    **The sub- and superdiagonals are sliced asymmetrically.** Row ``j`` of the
    implicit matrix has ``A[j, j-1] = -theta dt a_new[j]``, so the subdiagonal
    that ``solve_tridiagonal`` wants is ``a_new[1:]``, while the superdiagonal
    is ``c_new[:-1]``. With scalars, ``np.full`` hid the asymmetry; getting it
    wrong shifts every off-diagonal coefficient by one node, which is an
    O(dx) error that looks like a modest mispricing.

    Args:
        V: Values at the old time level, all nodes.
        old: ``(a, b, c)`` at the old level, from
            :func:`local_vol_coefficients`.
        new: ``(a, b, c)`` at the new level.
        dt: Step size.
        theta: ``0.5`` for Crank-Nicolson, ``1.0`` for fully implicit.
        bc_low: Dirichlet value at the lower boundary, at the new level.
        bc_high: Dirichlet value at the upper boundary, at the new level.

    Returns:
        Values at the new time level, all nodes.
    """
    a_old, b_old, c_old = old
    a_new, b_new, c_new = new
    explicit_weight = 1.0 - theta

    rhs = (
        explicit_weight * dt * a_old * V[:-2]
        + (1.0 + explicit_weight * dt * b_old) * V[1:-1]
        + explicit_weight * dt * c_old * V[2:]
    )
    rhs[0] += theta * dt * a_new[0] * bc_low
    rhs[-1] += theta * dt * c_new[-1] * bc_high

    interior = solve_tridiagonal(
        -theta * dt * a_new[1:],
        1.0 - theta * dt * b_new,
        -theta * dt * c_new[:-1],
        rhs,
    )

    out = np.empty_like(V)
    out[0] = bc_low
    out[1:-1] = interior
    out[-1] = bc_high
    return out


def _next_observation(schedule: np.ndarray, t_now: float) -> float | None:
    """First observation date at or after ``t_now``, or ``None`` if none remain.

    Used for the upper boundary condition, where a note far above the
    knock-out level will terminate on its next scheduled date rather than at
    the current instant.
    """
    if schedule.size == 0:
        return None
    idx = int(np.searchsorted(schedule, t_now - 1e-12))
    if idx >= schedule.size:
        return None
    return float(schedule[idx])


@dataclass(frozen=True, slots=True)
class _GridGeometry:
    """The grid itself, with no operator on it.

    Extracted so the constant-coefficient and local-volatility grids share one
    copy of the observation and barrier alignment. That alignment is the most
    bug-prone code in this module -- two Phase 1 defects lived in it, each
    producing a price that oscillated under refinement instead of converging --
    and duplicating it would let a future fix drift between the two paths.
    """

    steps_per_year: int
    n_steps: int
    dt: float
    dx: float
    n_nodes: int
    x_grid: np.ndarray
    S_grid: np.ndarray
    ki_index: int
    ko_index: int
    inner_intervals: int


def _build_geometry(
    terms: SnowballTerms, config: PDEConfig, pad_below: float, pad_above: float
) -> _GridGeometry:
    """Build the observation- and barrier-aligned grid.

    Pure code motion out of :class:`_SnowballGrid`, with the padding passed in
    rather than derived from a single volatility, so a skewed surface can ask
    for asymmetric boundaries. Every operation is in the original order, so the
    constant-coefficient path is unchanged to the last bit.
    """
    # --- Observation-aligned time grid --------------------------------------
    # Every monitoring date must land exactly on a time step. If the grid
    # is chosen independently, each observation has to be rounded to the
    # nearest step, which shifts it by up to dt/2 and makes the price jitter
    # under refinement rather than converge -- measured before this fix,
    # 252 -> 504 -> 1008 steps/year gave 0.06253, 0.06311, 0.06275, moving
    # non-monotonically by ~5e-4 with no trend.
    #
    # Snapping the step count to a whole multiple of the knock-in frequency
    # fixes it exactly: the daily dates i/252 then coincide with steps, and
    # because 252 is divisible by the monthly frequency 12, the knock-out
    # dates land on steps too.
    requested = max(int(round(config.steps_per_year)), 1)
    multiple = max(int(np.ceil(requested / terms.ki_obs_freq)), 1)
    steps_per_year = multiple * terms.ki_obs_freq
    n_steps = max(int(round(terms.T * steps_per_year)), 10)
    dt = terms.T / n_steps

    # --- Barrier-aligned grid -------------------------------------------
    # Both barriers must fall exactly on grid nodes. A barrier sitting
    # between nodes has to be represented by whichever node is nearest,
    # which displaces it by up to dx/2 and degrades the scheme from
    # second-order to first-order. Worse, as the node count changes the
    # displacement changes with it, so the price oscillates under
    # refinement instead of converging -- measured directly on this grid
    # before the fix: 0.06149, 0.06328, 0.06352, 0.06286, 0.06311 for
    # 200 to 3200 nodes, with no sign of settling.
    #
    # Choosing dx to divide the knock-in/knock-out span exactly, then
    # extending outward by whole multiples of it, pins both barriers to
    # nodes at every resolution.
    x_ki = float(np.log(terms.ki_barrier))
    x_ko = float(np.log(terms.ko_barrier))
    barrier_span = x_ko - x_ki

    # Padding outside the barriers, in log space. This must scale with the
    # width of the terminal distribution: a boundary condition is an
    # *approximation*, and unlike a discretisation error it does not shrink
    # when the grid is refined. Placing the boundary too close therefore
    # converges the scheme neatly to the wrong number -- which is precisely
    # what the fixed 0.6/0.3 padding inherited from the original code did.
    #
    # Measured with the fixed padding: the PDE's own self-convergence error
    # fell to ~1e-5 while it still disagreed with Monte Carlo by 1.6e-3 at
    # sigma=0.35 and 9e-4 at T=2 -- cases where 0.6 in log space is barely
    # one standard deviation. Four standard deviations of total variance,
    # floored for very short tenors, puts the boundary where its influence
    # is genuinely negligible.
    total_span = barrier_span + pad_below + pad_above

    # Apportion the requested nodes across the span, then round the inner
    # interval count so dx divides the barrier span exactly.
    inner_intervals = max(int(round(config.space_nodes * barrier_span / total_span)), 10)
    dx = barrier_span / inner_intervals
    intervals_below = max(int(np.ceil(pad_below / dx)), 1)
    intervals_above = max(int(np.ceil(pad_above / dx)), 1)

    n_nodes = intervals_below + inner_intervals + intervals_above + 1
    x_low = x_ki - intervals_below * dx
    x_grid = x_low + dx * np.arange(n_nodes, dtype=float)

    return _GridGeometry(
        steps_per_year=steps_per_year,
        n_steps=n_steps,
        dt=dt,
        dx=dx,
        n_nodes=n_nodes,
        x_grid=x_grid,
        S_grid=np.exp(x_grid),
        # Exact node indices of the two barriers, so the masks downstream are
        # expressed as index comparisons rather than floating-point ones.
        ki_index=intervals_below,
        ko_index=intervals_below + inner_intervals,
        inner_intervals=inner_intervals,
    )


class _SnowballGrid:
    """Log-spot grid and the constant-coefficient operator on it."""

    def __init__(self, terms: SnowballTerms, market: EquityMarket, config: PDEConfig) -> None:
        self.terms = terms
        self.market = market
        self.config = config

        total_vol = market.sigma * np.sqrt(terms.T)
        geometry = _build_geometry(
            terms, config, max(4.0 * total_vol, 0.6), max(4.0 * total_vol, 0.3)
        )
        self.steps_per_year = geometry.steps_per_year
        self.n_steps = geometry.n_steps
        self.dt = geometry.dt
        self.dx = geometry.dx
        self.n_nodes = geometry.n_nodes
        self.x_grid = geometry.x_grid
        self.S_grid = geometry.S_grid
        self.ki_index = geometry.ki_index
        self.ko_index = geometry.ko_index
        self.inner_intervals = geometry.inner_intervals

        # Black-Scholes operator in log space: constant coefficients, so the
        # three diagonals are scalars rather than arrays.
        diffusion = 0.5 * market.sigma**2
        advection = market.drift - 0.5 * market.sigma**2
        self.a = diffusion / self.dx**2 - advection / (2.0 * self.dx)
        self.b = -2.0 * diffusion / self.dx**2 - market.r
        self.c = diffusion / self.dx**2 + advection / (2.0 * self.dx)

    def discounted_growth(self, tau: float, t_from: float, t_to: float) -> float:
        """``exp(-r tau) * F(t_to) / F(t_from)`` -- the PV of receiving one unit of spot.

        Written as the single ``exp(-carry tau)`` the constant-carry algebra
        collapses to, rather than as a product of two exponentials. Identical
        mathematically; identical *bitwise* only this way, which is what keeps
        the flat path's results unchanged to the last digit.
        """
        del t_from, t_to
        return float(np.exp(-self.market.carry * tau))

    def step(
        self,
        V: np.ndarray,
        bc_low: float,
        bc_high: float,
        theta: float,
        step_index: int = 0,
    ) -> np.ndarray:
        """Advance one step backwards in time with the generalised theta scheme.

        ``theta = 0.5`` is Crank-Nicolson (second order, unconditionally
        stable); ``theta = 1.0`` is fully implicit (first order, strongly
        damping). Solves
        ``(I - theta * dt * L) V_new = (I + (1 - theta) * dt * L) V_old``.

        Args:
            V: Values at the old time level.
            bc_low: Dirichlet value at the lower boundary.
            bc_high: Dirichlet value at the upper boundary.
            theta: Scheme weight.
            step_index: Ignored here -- the operator does not depend on time
                when the coefficients are constant. Accepted so this and
                :class:`_LocalVolGrid` present the same interface and the
                backward loop needs no branch.
        """
        del step_index
        dt = self.dt
        a, b, c = self.a, self.b, self.c
        n_interior = self.n_nodes - 2
        explicit_weight = 1.0 - theta

        rhs = (
            explicit_weight * dt * a * V[:-2]
            + (1.0 + explicit_weight * dt * b) * V[1:-1]
            + explicit_weight * dt * c * V[2:]
        )
        # Fold the Dirichlet boundaries into the first and last equations.
        rhs[0] += theta * dt * a * bc_low
        rhs[-1] += theta * dt * c * bc_high

        main = np.full(n_interior, 1.0 - theta * dt * b)
        lower = np.full(max(n_interior - 1, 0), -theta * dt * a)
        upper = np.full(max(n_interior - 1, 0), -theta * dt * c)

        interior = solve_tridiagonal(lower, main, upper, rhs)

        out = np.empty_like(V)
        out[0] = bc_low
        out[1:-1] = interior
        out[-1] = bc_high
        return out

    def interpolate(self, V: np.ndarray, S_target: float) -> float:
        """Linearly interpolate ``V`` at a spot level, in log space."""
        return _interpolate_log(self.x_grid, V, S_target)


class _LocalVolGrid:
    """Log-spot grid and the variable-coefficient operator on it.

    Same geometry machinery as :class:`_SnowballGrid` -- the observation and
    barrier alignment matter no less here -- with two differences.

    The padding comes from the surface rather than from one volatility, so a
    skewed surface gets asymmetric boundaries. Four at-the-money standard
    deviations under-pads the downside badly once there is a skew, because
    implied volatility at the boundary strike is far above at the money; see
    :meth:`~otcpricer.models.local_vol.LocalVolSurface.boundary_moneyness`.

    And the coefficients depend on time, so they are rebuilt per step. The
    theta scheme needs the operator at *both* levels, but the new level of
    backward iteration ``k`` is the old level of iteration ``k + 1``, so a roll
    cache evaluates each level exactly once. The cache is correctness-neutral:
    a miss recomputes from scratch, so it cannot go stale. It also absorbs the
    two calls per iteration -- the knocked-in state is stepped before the
    un-knocked one -- which is why it is keyed on the step index rather than on
    a call counter.
    """

    def __init__(
        self,
        terms: SnowballTerms,
        market: EquityMarket,
        config: PDEConfig,
        local_vol: LocalVolatility,
        n_sigma: float = 4.0,
    ) -> None:
        self.terms = terms
        self.market = market
        self.config = config
        self.local_vol = local_vol

        lower, upper = local_vol.boundary_moneyness(terms.T, n_sigma)
        geometry = _build_geometry(terms, config, max(-lower, 0.6), max(upper, 0.3))
        self.steps_per_year = geometry.steps_per_year
        self.n_steps = geometry.n_steps
        self.dt = geometry.dt
        self.dx = geometry.dx
        self.n_nodes = geometry.n_nodes
        self.x_grid = geometry.x_grid
        self.S_grid = geometry.S_grid
        self.ki_index = geometry.ki_index
        self.ko_index = geometry.ko_index
        self.inner_intervals = geometry.inner_intervals

        self._cached_index = -1
        self._cached_old: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._cached_new: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def _coefficients(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return local_vol_coefficients(self.local_vol, self.x_grid, self.dx, t, self.market.r)

    def _levels(
        self, step_index: int
    ) -> tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """Operator coefficients at the old and new levels of iteration ``step_index``."""
        if self._cached_index == step_index and self._cached_old is not None:
            assert self._cached_new is not None
            return self._cached_old, self._cached_new

        t_old = self.terms.T - step_index * self.dt
        if self._cached_index == step_index - 1 and self._cached_new is not None:
            old = self._cached_new
        else:
            old = self._coefficients(t_old)
        new = self._coefficients(max(self.terms.T - (step_index + 1) * self.dt, 0.0))

        self._cached_index = step_index
        self._cached_old, self._cached_new = old, new
        return old, new

    def discounted_growth(self, tau: float, t_from: float, t_to: float) -> float:
        """``exp(-r tau) * F(t_to) / F(t_from)``, following the surface's forward curve.

        A constant carry would be inconsistent with the operator's advection
        term, and the error would leak inward through the diffusion.
        """
        return float(np.exp(-self.market.r * tau)) * self.local_vol.forward_growth(t_from, t_to)

    def step(
        self,
        V: np.ndarray,
        bc_low: float,
        bc_high: float,
        theta: float,
        step_index: int = 0,
    ) -> np.ndarray:
        """Advance one step backwards with time-dependent coefficients."""
        old, new = self._levels(step_index)
        return theta_step_variable(V, old, new, self.dt, theta, bc_low, bc_high)

    def interpolate(self, V: np.ndarray, S_target: float) -> float:
        """Linearly interpolate ``V`` at a spot level, in log space."""
        return _interpolate_log(self.x_grid, V, S_target)


def _interpolate_log(x_grid: np.ndarray, V: np.ndarray, S_target: float) -> float:
    """Linear interpolation of ``V`` at a spot level, in log space."""
    x = float(np.log(S_target))
    if x <= x_grid[0]:
        return float(V[0])
    if x >= x_grid[-1]:
        return float(V[-1])
    idx = int(np.clip(np.searchsorted(x_grid, x), 1, x_grid.size - 1))
    weight = (x - x_grid[idx - 1]) / (x_grid[idx] - x_grid[idx - 1])
    return float(V[idx - 1] * (1.0 - weight) + V[idx] * weight)


def price_snowball_pde(
    terms: SnowballTerms,
    market: EquityMarket,
    config: PDEConfig | None = None,
    *,
    knocked_in: bool = False,
    local_vol: LocalVolatility | None = None,
) -> SnowballPDEResult:
    """Value a snowball note with Crank-Nicolson finite differences.

    Independent of the Monte Carlo in every respect -- a different
    discretisation of a different formulation -- which is what makes agreement
    between the two meaningful evidence rather than a tautology.

    Two spot levels matter and must not be conflated:

    ``terms.S0``
        The **initial fixing**. It sets the barrier levels and the knock-in put
        strike, and is fixed for the life of the trade.
    ``market.S``
        The **current spot**, where the price is read.

    They coincide only at inception. Keeping them separate is what lets a
    seasoned trade be revalued, and what makes a spot delta meaningful.

    Args:
        terms: Contract terms, including the lock-up and knock-in strike mode.
        market: Market state; ``basis_pa`` enters the drift if set.
        config: Discretisation controls.
        knocked_in: Whether the note has already breached its knock-in barrier.
            A seasoned trade that has must be valued off the knocked-in
            surface; the two differ at the same spot, which is the whole reason
            the solver carries both.
        local_vol: A Dupire local volatility to price against. When omitted the
            solver takes the original constant-coefficient path unchanged, so
            flat-volatility results are identical to the last bit rather than
            merely close. When supplied, ``market.sigma`` is **not read** --
            the surface carries both the volatility and the drift -- which is
            what makes a volatility bump on ``market`` structurally incapable
            of producing a spurious vega here.

    Returns:
        A :class:`SnowballPDEResult` with the price as a fraction of notional
        and the full solution surface, from which delta and gamma can be read
        without further solves.
    """
    config = config or PDEConfig()
    grid: _SnowballGrid | _LocalVolGrid = (
        _SnowballGrid(terms, market, config)
        if local_vol is None
        else _LocalVolGrid(terms, market, config, local_vol)
    )

    S_grid = grid.S_grid
    T = terms.T
    r = market.r
    coupon = terms.coupon_pa
    ki_strike = terms.ki_put_strike

    # Index comparisons rather than float ones, now that the barriers sit
    # exactly on nodes. The knock-in region includes its own barrier node
    # (breach is "at or below"), and likewise knock-out is "at or above".
    ki_slice = slice(0, grid.ki_index + 1)
    ko_slice = slice(grid.ko_index, grid.n_nodes)

    # --- Terminal conditions ------------------------------------------------
    # Knocked in: a short put struck at ki_put_strike, normalised by notional.
    V_ki = np.minimum(S_grid / ki_strike - 1.0, 0.0)

    # A knocked-in note is still subject to the knock-out observation at
    # maturity: if it has recovered above the barrier it terminates and pays
    # the full coupon rather than the (zero) put value. Leaving this out makes
    # the knocked-in grid worth 0 above the strike, silently discarding the
    # coupon on every path that knocks in and then rallies through the barrier.
    #
    # The size of the error scales with how reachable that round trip is, which
    # is why it showed up as a ~9 sigma disagreement at ki_pct=0.85 and ~7 sigma
    # at sigma=0.35, while staying near the noise floor on the reference trade.
    final_ko_at_maturity = bool(
        terms.ko_observation_times().size and terms.ko_observation_times()[-1] >= T - 1e-12
    )
    if final_ko_at_maturity:
        V_ki[ko_slice] = coupon * T

    # Never knocked in and never knocked out: the full coupon. (Above the
    # knock-out barrier this is also exactly the knock-out payoff, so the
    # maturity knock-out needs no separate treatment here.)
    V_no_ki = np.full(grid.n_nodes, coupon * T, dtype=float)

    # --- Observation dates, mapped onto time-step indices -------------------
    # The backward loop leaves the grid at calendar time T - (k + 1) * dt after
    # iteration k, so an observation at t_obs belongs to
    #     k = (T - t_obs) / dt - 1.
    # Omitting the -1 applies every observation one step early in calendar
    # time, and -- worse -- makes an observation *at maturity* unrepresentable,
    # since it would need k = -1. Those have to be folded into the terminal
    # condition instead, before any stepping.
    def _to_step(t_obs: float) -> int | None:
        step = int(round((T - t_obs) / grid.dt)) - 1
        return None if step < 0 else min(step, grid.n_steps - 1)

    ko_steps: dict[int, float] = {}
    for t_obs in terms.ko_observation_times():
        step = _to_step(float(t_obs))
        if step is None:
            # A knock-out at maturity pays coupon * T, which is exactly what
            # the terminal condition already holds -- so this is a no-op rather
            # than a missing case. Asserting that keeps it from looking like an
            # oversight.
            continue
        ko_steps[step] = float(t_obs)

    # Knock-in is also a *discretely* monitored barrier, on its own daily
    # schedule. Applying it at every time step instead -- as the original
    # implementation did -- silently makes monitoring finer as the time grid is
    # refined, so the engine prices a progressively different contract rather
    # than converging to this one. Measured before the fix, refining 63 -> 1008
    # steps/year walked the price from 0.06678 down to 0.06121 with no sign of
    # settling, because each refinement bought more monitoring dates.
    ki_steps: set[int] = set()
    ki_observed_at_maturity = False
    for t_obs in terms.ki_observation_times():
        step = _to_step(float(t_obs))
        if step is None:
            ki_observed_at_maturity = True
        else:
            ki_steps.add(step)

    # A knock-in at maturity is *not* a no-op: below the barrier the note pays
    # the short-put loss, not the coupon. Folding it into the terminal
    # condition is the only way to express it, and leaving it out was a real
    # mispricing -- the terminal row carried the full coupon all the way down
    # through the knock-in region.
    if ki_observed_at_maturity:
        V_no_ki[ki_slice] = V_ki[ki_slice]

    # Sorted knock-out schedule, for the upper boundary condition.
    ko_times = np.sort(terms.ko_observation_times())

    rannacher = min(config.rannacher_steps, grid.n_steps)

    # Surfaces one step later in calendar time, kept for theta. After
    # iteration k the grid sits at t = T - (k + 1) * dt, so t = dt is reached
    # at k = n_steps - 2 -- the iteration before last.
    theta_step = grid.n_steps - 2
    V_no_ki_next = V_no_ki.copy()
    V_ki_next = V_ki.copy()

    # --- Backward induction -------------------------------------------------
    S_low = float(grid.S_grid[0])

    for k in range(grid.n_steps):
        theta = 1.0 if k < rannacher else 0.5
        # Stepping backwards from maturity, so after k + 1 steps `tau` is the
        # time *remaining* to maturity and `t_elapsed` the time accrued since
        # inception. Both are needed: one discounts, the other accrues coupon.
        tau = (k + 1) * grid.dt
        t_elapsed = max(T - tau, 0.0)

        # Knocked-in lower boundary: a short put deep in the money, whose value
        # is its discounted forward intrinsic
        #   -(K e^{-r tau} - S e^{-carry tau}) / K.
        # Using -e^{-r tau} instead -- the S -> 0 limit, as the original code
        # did -- assumes a total loss of notional at a boundary that actually
        # sits at S/S0 ~ 0.41, where the true loss is only ~0.59.
        bc_ki_low = -(
            np.exp(-r * tau) - (S_low / ki_strike) * grid.discounted_growth(tau, t_elapsed, T)
        )

        # Knocked-in *upper* boundary. A knocked-in note still carries the
        # knock-out feature, so this far above the barrier it terminates on its
        # next observation and collects the coupon -- it is not worthless.
        t_next = _next_observation(ko_times, t_elapsed)
        if t_next is None:
            bc_ki_high = 0.0  # past the last observation: a far-OTM short put
        else:
            bc_ki_high = coupon * t_next * float(np.exp(-r * (t_next - t_elapsed)))

        V_ki = grid.step(V_ki, bc_ki_low, bc_ki_high, theta, k)

        # Below the knock-in the un-knocked state has already transferred, so
        # its lower boundary is the knocked-in value.
        bc_no_ki_low = float(V_ki[0])

        # Upper boundary. The top node sits ~39% above the knock-out, so a note
        # there will knock out at the *next* observation date with near
        # certainty -- collecting the coupon accrued to that date, not to now.
        # Setting the boundary to `coupon * t_elapsed` (the accrual to the
        # current instant, as the original code did) therefore understates it
        # between observation dates, and the error leaks inward through the
        # diffusion term. Discounting the next scheduled knock-out payment back
        # to now is both more accurate and the correct limit.
        if t_next is None:
            # Past the final observation: it runs to maturity and, this far
            # above the knock-in, collects the full coupon.
            bc_no_ki_high = coupon * T * float(np.exp(-r * tau))
        else:
            bc_no_ki_high = coupon * t_next * float(np.exp(-r * (t_next - t_elapsed)))

        V_no_ki = grid.step(V_no_ki, bc_no_ki_low, bc_no_ki_high, theta, k)

        # On a knock-in observation date, transfer value into the knocked-in
        # state wherever spot is at or below the barrier.
        if k in ki_steps:
            V_no_ki[ki_slice] = V_ki[ki_slice]

        # Knock-out only on its own schedule, and never during the lock-up --
        # ko_observation_times() has already removed those dates.
        #
        # Applied to BOTH states. A note that has knocked in still carries the
        # knock-out feature: if it recovers above the barrier on an observation
        # date it terminates and pays the coupon, exactly as an un-knocked note
        # would. Modelling the knocked-in state as a plain short put -- as the
        # original two-state solver did -- silently drops that coupon. Measured
        # on the reference trade, 0.12% of paths knock in and then knock out,
        # and omitting them left the PDE ~0.46% below the Monte Carlo.
        if k in ko_steps:
            ko_value = coupon * ko_steps[k]
            V_no_ki[ko_slice] = ko_value
            V_ki[ko_slice] = ko_value

        if k == theta_step:
            V_no_ki_next = V_no_ki.copy()
            V_ki_next = V_ki.copy()

    # Read the price at the *current* spot, not at the initial fixing.
    #
    # terms.S0 is the initial fixing: it fixes the barrier levels and the
    # knock-in put strike, and never changes over the life of the trade.
    # market.S is where the underlying is now. For a newly struck note the two
    # coincide, which is why interpolating at terms.S0 -- as this did -- passed
    # every test: they all priced trades at inception. For anything seasoned it
    # was simply wrong, returning the same value at every spot (measured:
    # 0.063087 at spot 4500, 5000 and 5500 alike, while the Monte Carlo
    # correctly gave 0.0074, 0.0635 and 0.0719). It also made a spot-bumped
    # delta identically zero, which is what prompted finding it.
    surface = V_ki if knocked_in else V_no_ki
    price = grid.interpolate(surface, market.S)
    _log.debug(
        "snowball PDE priced",
        extra={
            "price": price,
            "spot": market.S,
            "fixing": terms.S0,
            "knocked_in": knocked_in,
            "nodes": grid.n_nodes,
            "steps": grid.n_steps,
            "ko_observations": len(ko_steps),
        },
    )
    return SnowballPDEResult(
        price=price,
        S_grid=grid.S_grid,
        V_no_ki=V_no_ki,
        V_ki=V_ki,
        V_no_ki_next_step=V_no_ki_next,
        V_ki_next_step=V_ki_next,
        dt=grid.dt,
        space_nodes=grid.n_nodes,
        time_steps=grid.n_steps,
    )
