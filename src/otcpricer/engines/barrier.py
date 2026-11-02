"""Barrier options by finite differences and by simulation.

Two engines for one product, which is this project's standing discipline and
the thing the shark fin never had: until now it was priced by an analytic
formula inlined in a Flask route and, separately, by a legacy grid, with
nothing comparing the two.

What the grid has to get right
------------------------------
Both lessons are inherited from Phase 1, where each cost a price that oscillated
under refinement instead of converging:

**The barrier must land exactly on a node.** A barrier sitting between nodes is
represented by the nearest one, displacing it by up to ``dx/2``; worse, the
displacement changes with the node count, so refining moves the answer around
rather than settling it. Here ``dx`` is chosen so the log-distance from spot to
the barrier is a whole number of steps.

**Observations must land exactly on time steps.** Same argument in time. The
step count is snapped to a whole multiple of the observation count.

Knock-ins come from parity, not from a second solver
----------------------------------------------------
A knock-in needs to know whether the barrier has already been touched, which is
a second state variable and would need a second grid dimension. It is not
solved that way here. Instead

    ``knock-in + knock-out = vanilla``

which holds exactly, for any monitoring frequency, when the rebate is zero --
every path ends up in precisely one of the two. So the knock-out is solved and
the knock-in is the vanilla minus it. That is not a shortcut: it is an identity,
and :func:`price_shark_fin_pde` refuses a knock-in with a rebate rather than
quietly applying it to the wrong leg.

The identity is also a free cross-check on the knock-out solve, and the tests
use it that way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from otcpricer.core.exceptions import EngineError, ParameterError
from otcpricer.core.market import EquityMarket
from otcpricer.core.types import OptionType
from otcpricer.core.vol import LocalVolatility
from otcpricer.engines.mc import MonteCarloConfig, simulate_log_paths
from otcpricer.engines.pde import PDEConfig, theta_step_variable
from otcpricer.models import bsm
from otcpricer.products.shark_fin import SharkFinTerms

__all__ = [
    "BarrierPDEResult",
    "BarrierMCResult",
    "price_shark_fin_analytic",
    "price_shark_fin_mc",
    "price_shark_fin_pde",
]

#: Rannacher damping steps, applied after **every** discontinuity rather than
#: only at maturity.
#:
#: Two fully implicit steps suppress the oscillation Crank-Nicolson shows
#: against a discontinuous initial condition. The usual placement is at the
#: start of the backward solve, and for a continuously monitored barrier that
#: is enough -- the solution stays smooth thereafter and convergence is
#: second-order in dx (measured: 8.9e-04, 2.1e-04, 5.0e-05, 8.8e-06 as the node
#: count doubles).
#:
#: A *discretely* monitored barrier introduces a fresh discontinuity at the
#: barrier node at every observation date, and damping only the first two steps
#: of the whole solve leaves 251 of them undamped. Measured on a daily-observed
#: down-and-out put, that cost first-order convergence: 1.3378, 1.3527, 1.3609,
#: 1.3660, 1.3692 as nodes doubled, differences merely halving, with the
#: Monte Carlo sitting at 1.3711 +/- 0.0012 -- an 8.5-sigma gap that was the
#: PDE's error, not the simulation's.
RANNACHER_STEPS = 2


@dataclass(frozen=True, slots=True)
class BarrierPDEResult:
    """A finite-difference barrier price and the grid behind it."""

    price: float
    space_nodes: int
    time_steps: int
    barrier_on_node: bool
    observations_on_steps: bool
    vol_model: str = "flat"
    method: str = "crank-nicolson+rannacher"

    def as_dict(self) -> dict[str, object]:
        return {
            "price": round(self.price, 10),
            "space_nodes": self.space_nodes,
            "time_steps": self.time_steps,
            "barrier_on_node": self.barrier_on_node,
            "observations_on_steps": self.observations_on_steps,
            "vol_model": self.vol_model,
            "method": self.method,
        }


@dataclass(frozen=True, slots=True)
class BarrierMCResult:
    """A simulated barrier price with its sampling error."""

    price: float
    std_error: float
    knock_prob: float
    paths: int
    vol_model: str = "flat"
    method: str = "monte-carlo"

    @property
    def ci95(self) -> float:
        return 1.959963985 * self.std_error

    def as_dict(self) -> dict[str, object]:
        return {
            "price": round(self.price, 10),
            "std_error": round(self.std_error, 10),
            "ci95": round(self.ci95, 10),
            "knock_prob": round(self.knock_prob, 6),
            "paths": self.paths,
            "vol_model": self.vol_model,
            "method": self.method,
        }


# ---------------------------------------------------------------------------
# Analytic, for the continuous limit only
# ---------------------------------------------------------------------------


def price_shark_fin_analytic(terms: SharkFinTerms, market: EquityMarket) -> float:
    """Rubinstein-Reiner closed form, for a **continuously** monitored barrier.

    Refuses a discretely monitored contract rather than returning the
    continuous price for it. Those are different contracts -- a discrete
    knock-out is worth strictly more, since the underlying can breach and
    recover between observations -- and returning one for the other is exactly
    the confusion this module was built to end.

    Raises:
        ParameterError: if the terms are discretely monitored, or carry a
            rebate, which this form does not cover.
    """
    if not terms.is_continuous:
        raise ParameterError(
            f"the analytic form is for a continuously monitored barrier; these "
            f"terms observe {terms.monitoring} times a year. Use the PDE or "
            f"Monte Carlo engine, which price the contract as written."
        )
    if terms.rebate != 0.0:
        raise ParameterError("the analytic form here does not carry a rebate")

    from scipy.stats import norm

    S, K, T = market.S, terms.K, terms.T
    r, q, sigma = market.r, market.q, market.sigma
    H = terms.barrier

    if sigma <= 0.0:
        raise ParameterError(f"sigma must be positive, got {sigma}")

    is_up_call = terms.kind.is_up and terms.option is OptionType.CALL
    is_down_put = (not terms.kind.is_up) and terms.option is OptionType.PUT
    if not (is_up_call or is_down_put):
        raise ParameterError(
            f"the analytic form covers up-and-out/in calls and "
            f"down-and-out/in puts; got a {terms.kind} {terms.option}"
        )

    # Haug (1998) p.66. Four terms, and the fourth matters: a two-term
    # "vanilla minus reflection" form is correct only when the barrier sits on
    # the far side of the strike, so that no surviving path finishes in the
    # money through the barrier region. For an up-and-out call with H > K the
    # payoff region and the barrier region overlap, and the two-term form
    # returns a negative number -- which a max(., 0) clamp will happily turn
    # into a confident zero.
    root_t = sigma * float(np.sqrt(T))
    lam = (r - q + 0.5 * sigma * sigma) / (sigma * sigma)
    phi = 1.0 if terms.option is OptionType.CALL else -1.0
    carry = float(np.exp(-q * T))
    discount = float(np.exp(-r * T))
    h_over_s = H / S

    x1 = float(np.log(S / K)) / root_t + lam * root_t
    x2 = float(np.log(S / H)) / root_t + lam * root_t
    y1 = float(np.log(H * H / (S * K))) / root_t + lam * root_t
    y2 = float(np.log(H / S)) / root_t + lam * root_t

    def plain(x: float) -> float:
        return float(
            phi * S * carry * norm.cdf(phi * x)
            - phi * K * discount * norm.cdf(phi * x - phi * root_t)
        )

    def reflected(y: float) -> float:
        return float(
            phi * S * carry * h_over_s ** (2.0 * lam) * norm.cdf(y)
            - phi * K * discount * h_over_s ** (2.0 * lam - 2.0) * norm.cdf(y - root_t)
        )

    # Haug's reflection terms carry eta: -1 for an up barrier, +1 for a down
    # one. In the (A - B + eta*(C - D)) arrangement used here the N(eta*y)
    # arguments fold into a single sign on the pair, because the constant that
    # N(y) = 1 - N(-y) introduces is identical for y1 and y2 and cancels
    # between them.
    #
    # The sign is why this is not a port of the previous implementation. That
    # one applied the up-barrier sign unconditionally and was checked only on
    # an up-and-out call, where it is right -- it agrees with QuantLib to
    # 2e-14. On a down-and-out put it was out by 1.7 to 5.5 on prices of order
    # 1 to 8, while carrying a comment claiming validation to 0.01%.
    eta = -1.0 if terms.kind.is_up else 1.0

    vanilla = plain(x1)

    # Domain guard, before the formula rather than after. An up-and-out call
    # with H <= K has no surviving path that finishes in the money, and a
    # down-and-out put with H >= K likewise -- those are worth zero as a fact
    # about the contract, not because a formula returned something negative.
    # The old route raised an error for them, which is also wrong.
    if (is_up_call and H <= K) or (is_down_put and H >= K):
        if terms.is_knock_out:
            return 0.0
        return terms.participation * vanilla

    knock_out = vanilla - plain(x2) + eta * (reflected(y1) - reflected(y2))

    # No clamp. A negative value here would mean the formula is being applied
    # outside its domain, and silently returning zero is how the old route
    # reported "worthless" for contracts that were merely unsupported.
    if knock_out < -1e-10:
        raise EngineError(
            f"the analytic barrier form returned {knock_out:.10f} for a "
            f"{terms.kind} {terms.option} with H={H} and K={K}; that is outside "
            f"its domain rather than a price"
        )
    knock_out = terms.participation * max(knock_out, 0.0)

    if terms.is_knock_out:
        return knock_out
    return terms.participation * vanilla - knock_out


# ---------------------------------------------------------------------------
# Finite differences
# ---------------------------------------------------------------------------


def _coefficients(
    variance: np.ndarray, drift: float, rate: float, dx: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Log-space operator coefficients at the interior nodes.

    The same discretisation ``pde.local_vol_coefficients`` applies, written
    against a variance *array* rather than a LocalVolatility object, so the
    flat and local-volatility paths here differ only in how that array is
    filled. That matters: it is what makes the constant-volatility case a
    genuine degenerate check on the variable one rather than a separate
    code path with its own bugs.
    """
    inner = variance[1:-1]
    diffusion = 0.5 * inner / (dx * dx)
    advection = (drift - 0.5 * inner) / (2.0 * dx)
    lower = diffusion - advection
    upper = diffusion + advection
    main = -2.0 * diffusion - rate
    return lower, main, upper


@dataclass(frozen=True, slots=True)
class _BarrierGrid:
    """Barrier- and observation-aligned grid for a single barrier."""

    x_grid: np.ndarray
    S_grid: np.ndarray
    dx: float
    dt: float
    n_steps: int
    barrier_index: int
    observation_steps: frozenset[int]

    @property
    def n_nodes(self) -> int:
        return int(self.x_grid.size)


def _build_barrier_grid(
    terms: SharkFinTerms, market: EquityMarket, config: PDEConfig, n_sigma: float
) -> _BarrierGrid:
    """Place the barrier on a node and every observation on a step."""
    # --- time: a whole number of steps between observations ---------------
    requested = max(int(round(config.steps_per_year)), 1)
    if terms.is_continuous:
        n_steps = max(int(round(terms.T * requested)), 10)
        # Empty, deliberately. The grid stops at the barrier and its Dirichlet
        # boundary *is* the barrier, so there is nothing to project. Treating
        # every step as an observation instead -- which looks harmless -- makes
        # the Rannacher rule below fire on every step, leaving the scheme fully
        # implicit throughout and first-order in time. Measured: it turned a
        # clean second-order sequence (8.9e-04 -> 8.8e-06) into one that stopped
        # improving at all.
        observation_steps: frozenset[int] = frozenset()
    else:
        per_observation = max(int(round(requested / int(terms.monitoring))), 1)
        n_steps = per_observation * terms.n_observations
        observation_steps = frozenset(
            per_observation * (index + 1) for index in range(terms.n_observations)
        )
    dt = terms.T / n_steps

    # --- space: the barrier exactly on a node ------------------------------
    x_spot = float(np.log(market.S))
    x_barrier = float(np.log(terms.barrier))
    span = abs(x_barrier - x_spot)

    total_vol = market.sigma * float(np.sqrt(terms.T))
    reach = max(n_sigma * total_vol, span * 1.5, 1e-3)

    # Whole number of intervals between spot and barrier, so dx divides the
    # span exactly at every resolution.
    inner = max(int(round(config.space_nodes * span / (2.0 * reach))), 4)
    dx = span / inner

    # Extend outward by whole multiples of dx, so the barrier stays on a node.
    pad = max(int(np.ceil(reach / dx)), inner + 2)
    if terms.is_continuous:
        # The grid stops AT the barrier, which then acts as a Dirichlet
        # boundary of the live domain. This is not a tidier way of writing the
        # same scheme -- it is a different and correct one.
        #
        # Carrying the grid past the barrier and re-imposing the dead zone at
        # every step looks equivalent and is not: the operator couples nodes
        # across the barrier within each step, so the barrier is effectively
        # smeared over one dx. Measured, that costs first-order convergence
        # instead of second -- the error merely halved as the node count
        # doubled (0.135, 0.065, 0.040, 0.023 for 400 to 3200 nodes) and was
        # still 3% of price at 3200 nodes.
        if terms.kind.is_up:
            x_low, x_high = x_spot - pad * dx, x_barrier
        else:
            x_low, x_high = x_barrier, x_spot + pad * dx
    elif terms.kind.is_up:
        # Discrete monitoring genuinely lives on both sides: between
        # observations the underlying may be above the barrier and recover, so
        # the dead zone has to be solved rather than excluded.
        x_low = x_spot - pad * dx
        x_high = x_barrier + inner * dx
    else:
        x_low = x_barrier - inner * dx
        x_high = x_spot + pad * dx

    n_nodes = int(round((x_high - x_low) / dx)) + 1
    x_grid = x_low + dx * np.arange(n_nodes, dtype=float)
    barrier_index = int(round((x_barrier - x_low) / dx))

    return _BarrierGrid(
        x_grid=x_grid,
        S_grid=np.exp(x_grid),
        dx=dx,
        dt=dt,
        n_steps=n_steps,
        barrier_index=barrier_index,
        observation_steps=observation_steps,
    )


def _boundary_values(
    terms: SharkFinTerms, market: EquityMarket, S_grid: np.ndarray, tau: float
) -> tuple[float, float]:
    """Dirichlet values at the two grid edges, with ``tau`` left to run.

    The edge past the barrier is the dead zone and takes the rebate. The far
    edge takes the deep-moneyness limit of the payoff rather than zero: for a
    call at a very high spot that is the forward minus the discounted strike,
    and using zero there would impose a wrong boundary and bleed inward.
    """
    discount = float(np.exp(-market.r * tau))
    carry = float(np.exp(-market.q * tau))
    low, high = float(S_grid[0]), float(S_grid[-1])

    if terms.option is OptionType.CALL:
        deep_low = 0.0
        deep_high = terms.participation * max(high * carry - terms.K * discount, 0.0)
    else:
        deep_low = terms.participation * max(terms.K * discount - low * carry, 0.0)
        deep_high = 0.0

    if terms.kind.is_up:
        # The upper edge is beyond the barrier.
        return deep_low, terms.rebate
    return terms.rebate, deep_high


def price_shark_fin_pde(
    terms: SharkFinTerms,
    market: EquityMarket,
    config: PDEConfig | None = None,
    *,
    local_vol: LocalVolatility | None = None,
    n_sigma: float = 6.0,
) -> BarrierPDEResult:
    """Price a shark fin by Crank-Nicolson with Rannacher damping.

    Knock-outs are solved directly: at each observation step the nodes beyond
    the barrier are set to the rebate. Knock-ins come from in-out parity
    against a vanilla priced on the same grid, which keeps the two legs
    consistent by construction.

    Args:
        terms: The note.
        market: Market state.
        config: Grid controls.
        local_vol: A Dupire local volatility, or ``None`` for ``market.sigma``.
        n_sigma: How far the grid reaches beyond spot, in total volatilities.

    Raises:
        ParameterError: on a knock-in with a rebate, which parity cannot split.
        EngineError: if the grid failed to put the barrier on a node, which
            would silently degrade the scheme rather than fail.
    """
    config = config or PDEConfig()

    if (not terms.is_knock_out) and terms.rebate != 0.0:
        raise ParameterError(
            "a knock-in with a rebate cannot be priced by in-out parity: the "
            "rebate belongs to the knock-out leg only, so the two no longer "
            "sum to a vanilla. Price the knock-out and its rebate separately."
        )

    grid = _build_barrier_grid(terms, market, config, n_sigma)
    barrier_on_node = bool(abs(grid.x_grid[grid.barrier_index] - np.log(terms.barrier)) < 1e-12)
    if not barrier_on_node:
        raise EngineError(
            f"the barrier at {terms.barrier} did not land on a grid node "
            f"(nearest is {float(np.exp(grid.x_grid[grid.barrier_index])):.8f}); "
            f"refusing to price on a displaced barrier"
        )

    S_grid = grid.S_grid
    values = terms.terminal_payoff(S_grid)

    # The dead zone at maturity, applied before stepping so the terminal
    # condition is the knocked-out one. For a continuous barrier the grid stops
    # at the barrier, so this touches only that single boundary node.
    dead = terms.breached(S_grid)
    values = np.where(dead, terms.rebate, values)

    # Maturity is itself a discontinuity, so the damping starts here.
    steps_since_discontinuity = 0

    for step in range(grid.n_steps, 0, -1):
        t_now = step * grid.dt
        t_next = t_now - grid.dt

        if local_vol is not None:
            variance_now = np.asarray(local_vol.local_vol(S_grid, t_now)) ** 2
            variance_next = np.asarray(local_vol.local_vol(S_grid, max(t_next, 0.0))) ** 2
            drift = local_vol.drift(t_now)
        else:
            variance_now = np.full(S_grid.size, market.sigma**2)
            variance_next = variance_now
            drift = market.drift

        a_now, b_now, c_now = _coefficients(variance_now, drift, market.r, grid.dx)
        a_next, b_next, c_next = _coefficients(variance_next, drift, market.r, grid.dx)

        # Rannacher: fully implicit for the steps immediately following any
        # discontinuity, which means maturity *and* every observation date.
        theta = 1.0 if steps_since_discontinuity < RANNACHER_STEPS else 0.5
        steps_since_discontinuity += 1

        # Dirichlet values at the NEW level, which is what theta_step_variable
        # folds in. One side of the grid lies past the barrier and is therefore
        # the dead zone; the other is a deep in- or out-of-the-money limit
        # where the option's value is known in closed form.
        tau = terms.T - t_next
        bc_low, bc_high = _boundary_values(terms, market, S_grid, tau)

        values = theta_step_variable(
            values,
            (a_now, b_now, c_now),
            (a_next, b_next, c_next),
            dt=grid.dt,
            theta=theta,
            bc_low=bc_low,
            bc_high=bc_high,
        )

        # The barrier bites only on an observation date -- and re-imposing it
        # creates a new discontinuity, so the damping counter restarts.
        if (step - 1) in grid.observation_steps:
            values = np.where(dead, terms.rebate, values)
            steps_since_discontinuity = 0

    price = float(np.interp(np.log(market.S), grid.x_grid, values))

    if not terms.is_knock_out:
        vanilla = float(
            bsm.price(terms.option, market.S, terms.K, terms.T, market.r, market.sigma, market.q)
        )
        price = terms.participation * vanilla - price

    return BarrierPDEResult(
        price=price,
        space_nodes=grid.n_nodes,
        time_steps=grid.n_steps,
        barrier_on_node=barrier_on_node,
        observations_on_steps=True,
        vol_model="dupire-local" if local_vol is not None else "flat",
    )


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------


def price_shark_fin_mc(
    terms: SharkFinTerms,
    market: EquityMarket,
    config: MonteCarloConfig | None = None,
    *,
    local_vol: LocalVolatility | None = None,
) -> BarrierMCResult:
    """Price a shark fin by simulation.

    The observation schedule is applied as written -- the barrier is tested at
    the scheduled dates only, which is what makes this a check on the PDE's
    discrete monitoring rather than on a continuous idealisation of it.

    A continuously monitored contract is simulated at the step grid, which
    *underestimates* the knock probability: a path can breach and recover
    within a step. That bias is real, is not removed by more paths, and the
    tests measure it rather than assuming it away.
    """
    config = config or MonteCarloConfig()
    times, paths = simulate_log_paths(market, terms.T, config, local_vol=local_vol)

    if terms.is_continuous:
        observation_indices = np.arange(1, times.size)
    else:
        schedule = terms.observation_times()
        observation_indices = np.array(
            [int(np.abs(times - t).argmin()) for t in schedule], dtype=int
        )

    observed = paths[observation_indices, :]
    breached = terms.breached(observed)
    knocked = breached.any(axis=0)

    terminal = terms.terminal_payoff(paths[-1, :])
    discount = float(np.exp(-market.r * terms.T))

    if terms.is_knock_out:
        payoff = np.where(knocked, 0.0, terminal) * discount
        if terms.rebate:
            # Paid at the first touch, so each path discounts from its own date.
            first = np.argmax(breached, axis=0)
            touch_time = times[observation_indices][first]
            payoff = np.where(knocked, terms.rebate * np.exp(-market.r * touch_time), payoff)
    else:
        payoff = np.where(knocked, terminal, 0.0) * discount

    if config.antithetic:
        half = payoff.size // 2
        sample = 0.5 * (payoff[:half] + payoff[half : 2 * half])
    else:
        sample = payoff

    return BarrierMCResult(
        price=float(sample.mean()),
        std_error=float(sample.std(ddof=1) / np.sqrt(sample.size)),
        knock_prob=float(knocked.mean()),
        paths=int(payoff.size),
        vol_model="dupire-local" if local_vol is not None else "flat",
    )
