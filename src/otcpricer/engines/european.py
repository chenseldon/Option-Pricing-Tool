"""European options under local volatility -- the engine that proves the rest.

Why this exists before the snowball work
----------------------------------------
A local volatility surface is derived from an implied volatility surface, so
there is exactly one way to check it: **price the vanillas the surface was
built from, and see whether the implied volatilities come back.** Dupire's
construction guarantees they must, in continuous time. Any discrepancy is
therefore an error -- in the formula, the forward curve, the moneyness
mapping, the operator coefficients, the index alignment, or the discretisation
-- and the round trip localises it where nothing downstream could.

That makes this the single most informative test in the phase, and it is worth
a real engine rather than a test fixture. It is also useful on its own: a
vanilla priced on the fitted surface is a skew-consistent quote, which a flat
volatility cannot give.

Two solvers, because they fail differently
------------------------------------------
The PDE and the Monte Carlo share the local volatility and nothing else. The
PDE tests the variable-coefficient operator and its boundary treatment; the
Monte Carlo tests the time-stepping scheme and carries a discretisation bias
the PDE does not have. Running both against the same closed form separates
those failure modes instead of pooling them.

What makes the comparison exact
-------------------------------
The surface is anchored at the forward curve's own ``F(0)``, so the curve
reproduces every quoted forward exactly and the reference price is Black-76 at
that expiry's forward and implied volatility. Without that anchoring the test
would be measuring a spot mismatch and reporting it as a model error.

Grid notes
----------
The strike is placed exactly on a node. A European payoff has a kink there, and
a kink sitting between nodes is represented by its neighbours, which costs an
order of convergence -- the same reasoning that pins the snowball's barriers to
nodes. Rannacher start-up damps the kink for the same reason it damps the
snowball's barrier discontinuity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from otcpricer.core.exceptions import OtcPricerError
from otcpricer.core.logging import get_logger
from otcpricer.core.types import OptionType
from otcpricer.core.validation import check_positive
from otcpricer.core.vol import LocalVolatility
from otcpricer.engines.mc import MonteCarloConfig, normal_draws
from otcpricer.engines.pde import PDEConfig, local_vol_coefficients, theta_step_variable

__all__ = [
    "EuropeanMCResult",
    "EuropeanPDEResult",
    "price_european_mc",
    "price_european_pde",
]

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EuropeanPDEResult:
    """A European valuation under local volatility, with its grid evidence.

    Attributes:
        price: Value at the current spot.
        S_grid: Grid spot levels, increasing.
        V: Value at each node at valuation time.
        strike_index: Node the strike sits on. Exact, by construction.
        dx: Log-spot spacing.
        dt: Time step in years.
        space_nodes: Nodes used.
        time_steps: Steps used.
        vol_model: What produced the number. Always ``"dupire-local"`` here;
            carried so a price is never read without knowing its basis.
    """

    price: float
    S_grid: np.ndarray
    V: np.ndarray
    strike_index: int
    dx: float
    dt: float
    space_nodes: int
    time_steps: int
    vol_model: str = "dupire-local"

    def as_dict(self) -> dict[str, float | int | str]:
        """Flat dict for API responses; omits the grid arrays."""
        return {
            "price": round(self.price, 10),
            "space_nodes": self.space_nodes,
            "time_steps": self.time_steps,
            "dx": round(self.dx, 10),
            "vol_model": self.vol_model,
        }


@dataclass(frozen=True, slots=True)
class EuropeanMCResult:
    """A European valuation by local-volatility Monte Carlo.

    Attributes:
        price: Discounted mean payoff.
        std_error: Standard error, from the antithetic pair means.
        ci95: Half-width of the 95% confidence interval.
        paths: Paths simulated.
        steps: Time steps per path. Under local volatility this is an
            **accuracy** parameter, not merely a schedule: the scheme is no
            longer exact, so the price carries a discretisation bias that falls
            with the step count.
        vol_model: What produced the number.
    """

    price: float
    std_error: float
    ci95: float
    paths: int
    steps: int
    vol_model: str = "dupire-local"

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "price": round(self.price, 10),
            "std_error": round(self.std_error, 10),
            "ci95": round(self.ci95, 10),
            "paths": self.paths,
            "steps": self.steps,
            "vol_model": self.vol_model,
        }


def _payoff(option_type: OptionType, S: np.ndarray, K: float) -> np.ndarray:
    wanted = OptionType.parse(option_type)
    if wanted is OptionType.CALL:
        return np.maximum(S - K, 0.0)
    return np.maximum(K - S, 0.0)


def _time_levels(
    local_vol: LocalVolatility, T: float, n_steps: int, variance_graded: bool
) -> np.ndarray:
    """Calendar times from ``T`` down to 0, optionally graded by variance.

    Uniform steps are the wrong mesh for a Dupire surface. Local variance
    behaves like ``dw/dt ~ t^-gamma`` as ``t -> 0`` -- integrable, but
    unbounded -- so a uniform mesh resolves it only to first order no matter
    how smooth the scheme is. Measured on the 510050 fit, whose calibrated
    ``gamma`` sits at the upper bound of 0.5 and so has the sharpest
    singularity the parameterisation allows, the error fell as 1.45, 0.79,
    0.42, 0.21, 0.106, 0.051 vol points for 500 to 16000 uniform steps: clean
    first order, while space refinement had already plateaued at 0.25.

    Grading so that each step carries an equal increment of **total variance**
    is the natural time change for that singularity: it is exactly the
    substitution that removes it. ``theta(t)`` is monotone, hence invertible,
    and ``atm_total_vol`` gives its square root, so the inversion is a single
    interpolation on a fine auxiliary grid.

    The snowball solver deliberately does *not* do this. Its time grid has to
    put every knock-out and knock-in observation on an exact step, which a
    graded mesh would break -- and getting observation dates wrong was a
    measured defect in Phase 1, worth more than the convergence order. The two
    engines therefore differ on purpose, and this note is here so the
    difference does not look like an oversight.
    """
    if not variance_graded:
        return np.linspace(T, 0.0, n_steps + 1)

    # theta(t) on a fine auxiliary grid, then invert by interpolation.
    fine_t = np.linspace(0.0, T, max(20 * n_steps, 2000))
    fine_variance = np.array([local_vol.atm_total_vol(float(t)) ** 2 for t in fine_t])
    # Monotonicity is the calendar no-arbitrage condition, so this is safe;
    # the maximum guards against a flat segment confusing np.interp.
    fine_variance = np.maximum.accumulate(fine_variance)
    targets = np.linspace(fine_variance[-1], 0.0, n_steps + 1)
    levels = np.interp(targets[::-1], fine_variance, fine_t)[::-1]
    levels[0], levels[-1] = T, 0.0
    return np.asarray(levels)


def price_european_pde(
    spot: float,
    strike: float,
    T: float,
    rate: float,
    option_type: OptionType | str,
    local_vol: LocalVolatility,
    config: PDEConfig | None = None,
    *,
    n_sigma: float = 4.0,
    variance_graded: bool = True,
    min_time_steps: int = 800,
) -> EuropeanPDEResult:
    """Value a European option under local volatility by finite differences.

    Args:
        spot: Current spot.
        strike: Strike.
        T: Time to expiry in years.
        rate: Continuously compounded discount rate. The *drift* comes from
            the local volatility's own forward curve, not from here -- using
            anything else would evaluate the volatility at the wrong moneyness.
        option_type: Call or put.
        local_vol: The local volatility surface.
        config: Discretisation controls.
        n_sigma: Boundary placement, in standard deviations.
        variance_graded: Concentrate time steps where variance accumulates
            fastest. See :func:`_time_levels` for why the default is on.
        min_time_steps: Floor on the total step count, regardless of tenor.
            ``steps_per_year`` is the wrong knob here: accuracy depends on the
            total number of steps, so a short-dated option gets far too few.
            At ``steps_per_year = 1008`` a 0.08-year expiry would take 80 steps
            and err by 3.5 volatility points; 800 steps brings that to a
            fraction of one.

    Returns:
        An :class:`EuropeanPDEResult`.

    Raises:
        OtcPricerError: on non-positive spot, strike or maturity.
    """
    check_positive(spot, "spot")
    check_positive(strike, "strike")
    check_positive(T, "T")
    wanted = OptionType.parse(option_type)
    config = config or PDEConfig()

    n_steps = max(int(round(T * config.steps_per_year)), min_time_steps)
    time_levels = _time_levels(local_vol, T, n_steps, variance_graded)

    # --- Grid, with the strike pinned to a node -----------------------------
    lower, upper = local_vol.boundary_moneyness(T, n_sigma)
    x_spot, x_strike = float(np.log(spot)), float(np.log(strike))
    x_low = min(x_spot + lower, x_strike - 0.1)
    x_high = max(x_spot + upper, x_strike + 0.1)

    intervals = max(int(config.space_nodes), 50)
    dx = (x_high - x_low) / intervals
    # Shift the grid down so the payoff kink lands exactly on a node. A kink
    # between nodes is represented by its neighbours and costs an order of
    # accuracy -- the same reason the snowball grid pins its barriers.
    #
    # The shift is the distance to the *next* node below, not the remainder
    # itself: with `x_strike - x_low = m dx + rem`, subtracting `rem` leaves
    # `m dx + 2 rem`, which is no more aligned than before. Subtracting
    # `dx - rem` gives `(m + 1) dx`. The outer modulo keeps an already-aligned
    # strike untouched.
    x_low -= (dx - (x_strike - x_low) % dx) % dx
    n_nodes = int(np.ceil((x_high - x_low) / dx)) + 1
    x_grid = x_low + dx * np.arange(n_nodes, dtype=float)
    S_grid = np.exp(x_grid)
    strike_index = int(round((x_strike - x_low) / dx))

    # --- Terminal condition -------------------------------------------------
    V = _payoff(wanted, S_grid, strike)

    rannacher = min(config.rannacher_steps, n_steps)
    S_low, S_high = float(S_grid[0]), float(S_grid[-1])

    # Coefficients roll: the new level of one iteration is the old level of the
    # next, so each time level is evaluated once rather than twice.
    coefficients = local_vol_coefficients(local_vol, x_grid, dx, T, rate)

    for k in range(n_steps):
        theta = 1.0 if k < rannacher else 0.5
        t_new = float(time_levels[k + 1])  # calendar time being solved for
        dt = float(time_levels[k] - t_new)
        tau = T - t_new  # time remaining to expiry

        new_coefficients = local_vol_coefficients(local_vol, x_grid, dx, t_new, rate)

        # Dirichlet boundaries at discounted forward intrinsic. The growth
        # factor comes from the forward curve, so it stays consistent with the
        # drift in the operator; a constant carry would not.
        growth = local_vol.forward_growth(t_new, T)
        discount = float(np.exp(-rate * tau))
        if wanted is OptionType.CALL:
            bc_low = 0.0
            bc_high = discount * (S_high * growth - strike)
        else:
            bc_low = discount * (strike - S_low * growth)
            bc_high = 0.0

        V = theta_step_variable(V, coefficients, new_coefficients, dt, theta, bc_low, bc_high)
        coefficients = new_coefficients

    price = float(np.interp(np.log(spot), x_grid, V))
    _log.debug(
        "European PDE priced under local volatility",
        extra={"price": price, "nodes": n_nodes, "steps": n_steps, "strike_on_node": True},
    )
    return EuropeanPDEResult(
        price=price,
        S_grid=S_grid,
        V=V,
        strike_index=strike_index,
        dx=dx,
        dt=float(np.max(np.diff(time_levels))),
        space_nodes=n_nodes,
        time_steps=n_steps,
    )


def price_european_mc(
    spot: float,
    strike: float,
    T: float,
    rate: float,
    option_type: OptionType | str,
    local_vol: LocalVolatility,
    config: MonteCarloConfig | None = None,
    *,
    draws: np.ndarray | None = None,
) -> EuropeanMCResult:
    """Value a European option under local volatility by Monte Carlo.

    Steps in log space with the volatility frozen at the start of each step --
    log-Euler. This is where exactness is lost: the constant-coefficient
    scheme integrates geometric Brownian motion exactly over any step size,
    while a state-dependent volatility cannot be integrated in closed form. So
    the price carries a genuine discretisation bias, and ``steps_per_year``
    becomes an accuracy parameter rather than a scheduling one.

    Stating that plainly matters because the Monte Carlo's own docstring
    advertises the opposite for the flat path, and that claim does not carry
    over.

    Args:
        spot: Current spot.
        strike: Strike.
        T: Time to expiry in years.
        rate: Discount rate. The drift comes from the local volatility's
            forward curve.
        option_type: Call or put.
        local_vol: The local volatility surface.
        config: Simulation controls.
        draws: Pre-generated normals, for common random numbers across bumps.

    Returns:
        A :class:`EuropeanMCResult`.

    Raises:
        OtcPricerError: on non-positive spot, strike or maturity.
    """
    check_positive(spot, "spot")
    check_positive(strike, "strike")
    check_positive(T, "T")
    wanted = OptionType.parse(option_type)
    config = config or MonteCarloConfig()

    n_steps = max(int(round(T * config.steps_per_year)), 1)
    dt = T / n_steps
    if draws is None:
        draws = normal_draws(n_steps, config.paths, config.seed, config.antithetic)
    elif draws.shape != (n_steps, config.paths):
        raise OtcPricerError(f"draws must have shape {(n_steps, config.paths)}, got {draws.shape}")

    log_spot = np.full(draws.shape[1], float(np.log(spot)), dtype=float)
    root_dt = float(np.sqrt(dt))
    for step in range(n_steps):
        t = step * dt
        sigma = np.asarray(local_vol.local_vol(np.exp(log_spot), t), dtype=float)
        log_spot = (
            log_spot
            + (local_vol.drift(t) - 0.5 * sigma * sigma) * dt
            + sigma * root_dt * draws[step]
        )

    discounted = np.exp(-rate * T) * _payoff(wanted, np.exp(log_spot), strike)
    price = float(np.mean(discounted))

    # Pair means when antithetic: each pair is one independent observation, so
    # using the flat path count would understate the error. Matches the
    # estimator in engines.mc.
    if config.antithetic and discounted.size >= 4:
        half = discounted.size // 2
        pair_means = 0.5 * (discounted[:half] + discounted[half : 2 * half])
        std_error = float(np.std(pair_means, ddof=1) / np.sqrt(half))
    elif discounted.size > 1:
        std_error = float(np.std(discounted, ddof=1) / np.sqrt(discounted.size))
    else:
        std_error = 0.0

    return EuropeanMCResult(
        price=price,
        std_error=std_error,
        ci95=1.959963985 * std_error,
        paths=int(discounted.size),
        steps=n_steps,
    )
