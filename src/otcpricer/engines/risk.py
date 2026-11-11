"""Greeks for snowball notes, by PDE grid differentiation and by Monte Carlo.

Why the PDE is the production method
------------------------------------
A finite difference solver computes ``V`` at every grid node on its way to the
price, so the value *function* is already in hand. Delta and gamma are then
differences on that grid: no extra solves, no bumping, and -- decisively -- no
Monte Carlo noise.

That last point is not a convenience. A snowball's payoff is discontinuous at
both barriers, and a simulated gamma divides the second difference of a
discontinuous estimator by ``h^2``. The noise is amplified beyond use, and no
amount of path count or step tuning fixes it, because the difficulty is in the
estimator rather than the sample size. The measured gap bears this out: the
PDE gamma converges smoothly to -1.868e-07 while the Monte Carlo finite
difference reports -2.245e-07, 17% away, with the error dominated by variance.

Monte Carlo is kept as an *independent* check -- a different formulation,
sampled rather than solved -- because two methods agreeing is the only reason
to believe either. Phase 1 established the prices that way; this does the same
for the risk.

What the three legacy methods got wrong
---------------------------------------
The replaced implementation reported three Greek sets and described them as
mutually validating. They disagreed in sign:

=====================  ===========  ===========
Method                 Legacy delta  Correct
=====================  ===========  ===========
Resimulation FD        -8.4e-05      +4.9e-05
Pathwise               +1.0e-05      +4.9e-05
Likelihood Ratio       +9.0e-06      +4.9e-05
=====================  ===========  ===========

All three were wrong. Every one was built on the Monte Carlo payoff that
carried the knock-in strike and discounting defects corrected earlier, and the
likelihood ratio carried three further errors of its own: a sign error in the
volatility score, a delta score taken from the terminal distribution when the
payoff is path-dependent, and a missing discount-factor term in rho. Those are
recorded in ``docs/models/greeks.md``.

Why pathwise differentiation is not offered
-------------------------------------------
Pathwise differentiation requires the payoff to be Lipschitz in the parameter,
so that the derivative and the expectation can be exchanged. A snowball's is
not: on a knock-out the value jumps from its continuation value to
``coupon * t``, and on a knock-in the entire leg switches. The method is
therefore invalid here, and the legacy code's ad-hoc "conditional MC
smoothing" was patching a method that does not apply rather than fixing a bug.

The sound alternative is a mixed estimator -- pathwise on the smooth part,
likelihood ratio on the jump -- which is materially more work and more error
surface. It is deliberately not implemented. A method honestly excluded is
worth more than a biased one presented as validated; see ``docs/models``.

Unit conventions match the vanilla Greeks in :mod:`otcpricer.models.bsm`:
delta per unit of spot, gamma per unit squared, vega per percentage point of
volatility, theta per calendar day, rho per percentage point of rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from otcpricer.core.exceptions import EngineError
from otcpricer.core.logging import get_logger
from otcpricer.core.market import EquityMarket
from otcpricer.core.types import Greeks
from otcpricer.core.vol import LocalVolatility
from otcpricer.engines.mc import (
    MonteCarloConfig,
    SnowballMCResult,
    normal_draws,
    price_snowball_mc,
)
from otcpricer.engines.pde import (
    LOCAL_VOL_PDE_CONFIG,
    PDEConfig,
    SnowballPDEResult,
    price_snowball_pde,
)
from otcpricer.products.snowball import SnowballTerms

__all__ = [
    "DEFAULT_CARRY_BUMP",
    "DEFAULT_RATE_BUMP",
    "DEFAULT_SPOT_BUMP_REL",
    "DEFAULT_VOL_BUMP",
    "RiskComparison",
    "SnowballRisk",
    "compare_snowball_risk",
    "grid_derivative_curves",
    "snowball_risk_mc",
    "snowball_risk_pde",
]

_log = get_logger(__name__)

#: Volatility bump for vega, in absolute vol units (1 percentage point).
DEFAULT_VOL_BUMP = 0.01
#: Rate bump for rho, in absolute units (25 basis points).
DEFAULT_RATE_BUMP = 0.0025

#: Absolute carry bump for ``carry_rho``, 25bp a year.
#:
#: Matched to :data:`DEFAULT_RATE_BUMP` so the two sensitivities are directly
#: comparable -- they are the same size of move in opposite halves of the
#: drift, and reading them side by side is how a desk sees that a basis move
#: is not a rate move.
DEFAULT_CARRY_BUMP = 0.0025
#: Relative spot bump for the Monte Carlo finite differences (1%).
#:
#: Larger than one would choose for a smooth payoff, deliberately. Common
#: random numbers remove most of the variance from a delta estimate but the
#: payoff still steps as paths cross barriers, so too small a bump leaves the
#: difference dominated by that discreteness.
DEFAULT_SPOT_BUMP_REL = 0.01

#: Scaling from per-unit to per-percentage-point, matching the vanilla Greeks.
_PER_PERCENT = 0.01
#: Calendar days in a year, for the daily theta convention.
_DAYS_PER_YEAR = 365.0


@dataclass(frozen=True, slots=True)
class SnowballRisk:
    """Greeks for a snowball note, with provenance.

    Attributes:
        greeks: The five sensitivities, in the conventions above.
        method: How they were produced, so a number is never read without
            knowing what produced it.
        std_error: Standard error per Greek where the method is sampled;
            empty for the deterministic PDE.
        price: The value the Greeks were taken around, as a cross-check that
            the risk and the quote come from the same state.
        carry_rho: Sensitivity to **total carry**, per percentage point a year,
            or ``None`` if it was not computed.

            One field, not two. ``q`` and ``basis_pa`` reach every engine only
            through their sum ``market.carry``, so ``dV/dq`` and
            ``dV/dbasis_pa`` are the same derivative -- measured at a relative
            difference of 1.5e-15, i.e. identical to the last bit. Reporting
            both would manufacture an independence the model does not have, and
            a desk hedging each separately would double-count.

            It is here rather than in :class:`~otcpricer.core.types.Greeks`
            because that is a fixed five-field record shared with the vanilla
            products, and a basis is not a vanilla concept.
    """

    greeks: Greeks
    method: str
    price: float
    std_error: dict[str, float] | None = None
    carry_rho: float | None = None

    #: Decimal places for serialisation. Generous on purpose: these are
    #: fractions of notional, so gamma lands around 1e-07 and the 8 places
    #: used for prices would leave it with two significant figures.
    DECIMALS = 12

    def as_dict(self) -> dict[str, object]:
        """Flat dict for API responses."""
        payload: dict[str, object] = {
            "method": self.method,
            "price": round(self.price, 8),
            **self.greeks.as_dict(self.DECIMALS),
        }
        if self.carry_rho is not None:
            payload["carry_rho"] = round(self.carry_rho, self.DECIMALS)
            payload["carry_rho_note"] = (
                "Sensitivity to total carry (q + basis_pa) per percentage "
                "point. There is deliberately no separate dividend "
                "sensitivity: the engines read only the sum, so dV/dq and "
                "dV/dbasis_pa are the same derivative to 1.5e-15."
            )
        if self.std_error:
            payload["std_error"] = {k: round(v, 10) for k, v in self.std_error.items()}
        return payload


def grid_derivative_curves(S_grid: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Delta and gamma at **every node** of a log-spot grid.

    The grid is uniform in ``x = log S``, so the chain rule gives

    * ``dV/dS   = (1/S) * dV/dx``
    * ``d2V/dS2 = (1/S^2) * (d2V/dx2 - dV/dx)``

    Separated out because these curves were always being computed and then
    thrown away: :func:`_grid_derivatives` formed them across the whole grid
    and returned two interpolated scalars. A Greeks surface needs the curves,
    and deriving them a second time somewhere else would be the start of two
    deltas that disagree in the last digits -- so the scalar now comes from
    this, rather than this being a copy of the scalar.

    Validated against a closed-form case: the error halves exactly on each
    refinement (8.71e-07, 4.36e-07, 2.18e-07, 1.09e-07 for delta), which is
    the first-order convergence the barrier treatment allows.
    """
    dx = float(np.log(S_grid[1]) - np.log(S_grid[0]))
    first = np.gradient(V, dx, edge_order=2)
    second = np.gradient(first, dx, edge_order=2)
    return first / S_grid, (second - first) / (S_grid * S_grid)


def _grid_derivatives(S_grid: np.ndarray, V: np.ndarray, spot: float) -> tuple[float, float]:
    """Return ``(delta, gamma)`` at ``spot``, off the curves above.

    Interpolated rather than read from the nearest node. Spot generally falls
    between nodes, and snapping to the nearest one would introduce an error of
    up to half a grid step that does not shrink as the grid refines.
    """
    delta_curve, gamma_curve = grid_derivative_curves(S_grid, V)
    x = np.log(S_grid)
    log_spot = float(np.log(spot))
    return (
        float(np.interp(log_spot, x, delta_curve)),
        float(np.interp(log_spot, x, gamma_curve)),
    )


def _pde_value(
    terms: SnowballTerms,
    market: EquityMarket,
    config: PDEConfig,
    knocked_in: bool,
    local_vol: LocalVolatility | None = None,
) -> float:
    """Price once with the PDE, for a bumped-parameter re-solve."""
    return price_snowball_pde(
        terms, market, config, knocked_in=knocked_in, local_vol=local_vol
    ).price


def _vega(
    terms: SnowballTerms,
    market: EquityMarket,
    config: PDEConfig,
    knocked_in: bool,
    local_vol: LocalVolatility | None,
    vol_bump: float,
) -> float:
    """Central-difference vega, per percentage point of volatility.

    Which volatility is bumped depends on where it comes from. On the flat path
    it is ``market.sigma``. On the local-volatility path ``market.sigma`` is
    not read at all, so bumping it would return a number produced by the grid
    changing shape rather than by the market moving -- the surface is shifted
    instead, at the money, which is the sensitivity a desk hedges.

    The two are *different derivatives* and should not be compared. A parallel
    shift of the at-the-money level also changes curvature at every maturity,
    because SSVI's ``phi`` depends on ``theta``; that is what re-calibrating to
    a shifted at-the-money level would give, and it is the right answer, but it
    is not the flat path's vega measured more carefully.
    """
    if local_vol is None:
        up = _pde_value(terms, market.with_shift(vol_abs=vol_bump), config, knocked_in)
        down = _pde_value(terms, market.with_shift(vol_abs=-vol_bump), config, knocked_in)
    else:
        up = _pde_value(terms, market, config, knocked_in, local_vol.with_atm_shift(vol_bump))
        down = _pde_value(terms, market, config, knocked_in, local_vol.with_atm_shift(-vol_bump))
    return (up - down) / (2.0 * vol_bump) * _PER_PERCENT


def snowball_risk_pde(
    terms: SnowballTerms,
    market: EquityMarket,
    config: PDEConfig | None = None,
    *,
    knocked_in: bool = False,
    vol_bump: float = DEFAULT_VOL_BUMP,
    rate_bump: float = DEFAULT_RATE_BUMP,
    carry_bump: float = DEFAULT_CARRY_BUMP,
    local_vol: LocalVolatility | None = None,
) -> SnowballRisk:
    """Greeks from the Crank-Nicolson solver. The production method.

    Delta and gamma come from the solution surface the solver already built, so
    they cost nothing beyond the price. Vega and rho need the parameter changed,
    so they are central differences over re-solves -- clean, because the solver
    is deterministic and contributes no noise of its own.

    Theta is read from two adjacent time steps of the *same* solve rather than
    from a shortened maturity. See
    :meth:`~otcpricer.engines.pde.SnowballPDEResult.surface_next_step` for why:
    shortening the tenor by a day silently deletes a monthly knock-out
    observation, so a maturity-bumped theta measures a contract change instead
    of time decay.

    Args:
        terms: Contract terms.
        market: Market state. ``market.S`` is current spot; the barriers stay
            pinned to ``terms.S0``, which is what makes delta a hedge ratio
            rather than the derivative of a re-struck contract.
        config: Discretisation controls.
        knocked_in: Risk the knocked-in surface, for a seasoned trade that has
            already breached its barrier.
        vol_bump: Absolute volatility bump for vega.
        rate_bump: Absolute rate bump for rho.
        carry_bump: Absolute carry bump for ``carry_rho``.

    Returns:
        A :class:`SnowballRisk` with no standard errors: the method is
        deterministic.
    """
    config = config or PDEConfig()
    base: SnowballPDEResult = price_snowball_pde(
        terms, market, config, knocked_in=knocked_in, local_vol=local_vol
    )

    # Delta and gamma: already in the solution.
    delta, gamma = _grid_derivatives(base.S_grid, base.surface(knocked_in=knocked_in), market.S)

    # Theta: one step of the same backward induction, so the observation
    # schedule is identical by construction. V advances forward in calendar
    # time, hence (V_next - V_now).
    x = np.log(base.S_grid)
    log_spot = float(np.log(market.S))
    value_now = float(np.interp(log_spot, x, base.surface(knocked_in=knocked_in)))
    value_next = float(np.interp(log_spot, x, base.surface_next_step(knocked_in=knocked_in)))
    theta = (value_next - value_now) / base.dt / _DAYS_PER_YEAR

    # Vega and rho: central differences over re-solves.
    vega = _vega(terms, market, config, knocked_in, local_vol, vol_bump)

    # Rho moves the discounting *and* the drift. On the local-volatility path
    # the drift lives on the surface's forward curve, so the surface has to be
    # shifted too -- omitting it leaves a discount-only rho, which on a
    # one-year snowball turned +0.0036 into -0.0001, a sign change rather than
    # a small distortion.
    rate_up = _pde_value(
        terms,
        market.with_shift(rate_abs=rate_bump),
        config,
        knocked_in,
        None if local_vol is None else local_vol.with_rate_shift(rate_bump),
    )
    rate_down = _pde_value(
        terms,
        market.with_shift(rate_abs=-rate_bump),
        config,
        knocked_in,
        None if local_vol is None else local_vol.with_rate_shift(-rate_bump),
    )
    rho = (rate_up - rate_down) / (2.0 * rate_bump) * _PER_PERCENT

    # Carry rho: the drift half of rho, without the discounting half. The
    # basis is now measured rather than assumed (marketdata/carry.py), and the
    # measurement moves: the spot-to-far carry has a standard deviation of
    # 2.1% to 3.9% a year across the panel, against a rate that moved 70bp over
    # three years. So this is the larger of the two exposures by an order of
    # magnitude, and it had no Greek at all.
    #
    # On the local-volatility path the surface carries the drift, so the
    # surface is shifted too -- with_carry_shift, not with_rate_shift, which
    # moves rate and drift together and would leave the carry unchanged.
    carry_up = _pde_value(
        terms,
        market.with_shift(carry_abs=carry_bump),
        config,
        knocked_in,
        None if local_vol is None else local_vol.with_carry_shift(carry_bump),
    )
    carry_down = _pde_value(
        terms,
        market.with_shift(carry_abs=-carry_bump),
        config,
        knocked_in,
        None if local_vol is None else local_vol.with_carry_shift(-carry_bump),
    )
    carry_rho = (carry_up - carry_down) / (2.0 * carry_bump) * _PER_PERCENT

    risk = SnowballRisk(
        greeks=Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho),
        method="pde-grid" if local_vol is None else "pde-grid+ssvi-parallel-vega",
        price=base.price,
        carry_rho=carry_rho,
    )
    _log.debug("snowball PDE risk", extra={"spot": market.S, **risk.greeks.as_dict()})
    return risk


def snowball_risk_mc(
    terms: SnowballTerms,
    market: EquityMarket,
    config: MonteCarloConfig | None = None,
    *,
    spot_bump_rel: float = DEFAULT_SPOT_BUMP_REL,
    vol_bump: float = DEFAULT_VOL_BUMP,
    rate_bump: float = DEFAULT_RATE_BUMP,
    carry_bump: float = DEFAULT_CARRY_BUMP,
    local_vol: LocalVolatility | None = None,
) -> SnowballRisk:
    """Greeks by Monte Carlo resimulation, as an independent check on the PDE.

    Every valuation reuses one set of normal draws -- common random numbers.
    Without that, the difference of two independent simulations is swamped by
    sampling noise; with it, the two runs differ only through the parameter,
    and most of the variance cancels.

    The barriers are *not* bumped with spot. Only ``market`` is perturbed, and
    the barriers live on ``terms``, so this measures the hedge ratio: how the
    note responds when the market moves and the contract stays put. Bumping the
    initial fixing instead would re-strike the barriers and answer a different,
    far less useful question.

    Gamma is returned but should be treated as an order-of-magnitude check
    only. It divides a second difference of a discontinuous estimator by
    ``h^2``, which amplifies variance; the PDE value is the one to use.

    Args:
        terms: Contract terms.
        market: Market state.
        config: Simulation controls.
        spot_bump_rel: Relative spot bump.
        vol_bump: Absolute volatility bump.
        rate_bump: Absolute rate bump.
        carry_bump: Absolute carry bump for ``carry_rho``.

    Returns:
        A :class:`SnowballRisk` with per-Greek standard errors propagated from
        the underlying runs. Theta's is the error of a *paired* difference;
        every other is a conservative quadrature bound.
    """
    config = config or MonteCarloConfig()

    # One shared draw set for every valuation below. simulate_log_paths
    # requires shape (steps, paths), and supplying draws forces a single batch,
    # so the caller is accepting that memory cost knowingly.
    n_steps = max(int(round(terms.T * config.steps_per_year)), 1)
    draws = normal_draws(n_steps, config.paths, config.seed, config.antithetic)

    def value(shifted: EquityMarket, surface: LocalVolatility | None = None) -> tuple[float, float]:
        result = price_snowball_mc(
            terms, shifted, config, draws=draws, local_vol=surface or local_vol
        )
        return result.price, result.std_error

    # The base run keeps its per-path values, because the paired theta below
    # needs them and re-running the same simulation to collect them would
    # double the cost of the most expensive call in the function.
    base_paired = price_snowball_mc(
        terms, market, config, draws=draws, local_vol=local_vol, keep_path_values=True
    )
    base_price, base_se = base_paired.price, base_paired.std_error

    spot_bump = market.S * spot_bump_rel
    up_price, up_se = value(market.with_shift(spot_rel=spot_bump_rel))
    down_price, down_se = value(market.with_shift(spot_rel=-spot_bump_rel))

    delta = (up_price - down_price) / (2.0 * spot_bump)
    gamma = (up_price - 2.0 * base_price + down_price) / (spot_bump * spot_bump)

    # Same reasoning as the PDE path: on the local-volatility path the market's
    # sigma does not drive the diffusion, so the surface is shifted instead.
    if local_vol is None:
        vol_up, vol_up_se = value(market.with_shift(vol_abs=vol_bump))
        vol_down, vol_down_se = value(market.with_shift(vol_abs=-vol_bump))
    else:
        vol_up, vol_up_se = value(market, local_vol.with_atm_shift(vol_bump))
        vol_down, vol_down_se = value(market, local_vol.with_atm_shift(-vol_bump))
    vega = (vol_up - vol_down) / (2.0 * vol_bump) * _PER_PERCENT

    rate_up, rate_up_se = value(
        market.with_shift(rate_abs=rate_bump),
        None if local_vol is None else local_vol.with_rate_shift(rate_bump),
    )
    rate_down, rate_down_se = value(
        market.with_shift(rate_abs=-rate_bump),
        None if local_vol is None else local_vol.with_rate_shift(-rate_bump),
    )
    rho = (rate_up - rate_down) / (2.0 * rate_bump) * _PER_PERCENT

    carry_up, carry_up_se = value(
        market.with_shift(carry_abs=carry_bump),
        None if local_vol is None else local_vol.with_carry_shift(carry_bump),
    )
    carry_down, carry_down_se = value(
        market.with_shift(carry_abs=-carry_bump),
        None if local_vol is None else local_vol.with_carry_shift(-carry_bump),
    )
    carry_rho = (carry_up - carry_down) / (2.0 * carry_bump) * _PER_PERCENT

    # Theta, from a seasoned valuation rather than a shortened tenor.
    #
    # Shortening T deletes a knock-out observation -- nine become eight on the
    # reference trade -- so a maturity-bumped theta measures a contract change
    # and nan was the honest answer. Advancing the *valuation date* changes no
    # term of the contract: the schedule and the coupon accrual stay written
    # from inception, and only the simulation's start, the surviving
    # observations and the discounting origin move.
    #
    # The step is one of the simulation's own steps, so dt is identical in both
    # runs and the seasoned run reuses every increment but the first. That
    # makes the difference a genuinely paired estimator rather than two
    # independent means.
    theta_step = 1.0 / config.steps_per_year
    seasoned = price_snowball_mc(
        terms,
        market,
        config,
        draws=draws[1:],
        local_vol=local_vol,
        t0=theta_step,
        keep_path_values=True,
    )
    theta, theta_se = _paired_theta(base_paired, seasoned, theta_step, config.antithetic)

    # Common random numbers correlate the bumped runs, so adding their errors
    # in quadrature overstates the result -- the shared noise largely cancels
    # in the difference. Kept as a deliberately conservative upper bound.
    def difference_error(se_a: float, se_b: float, scale: float) -> float:
        return float(np.hypot(se_a, se_b) / scale)

    std_error = {
        "delta": difference_error(up_se, down_se, 2.0 * spot_bump),
        "gamma": difference_error(np.hypot(up_se, down_se), 2.0 * base_se, spot_bump**2),
        "vega": difference_error(vol_up_se, vol_down_se, 2.0 * vol_bump / _PER_PERCENT),
        "rho": difference_error(rate_up_se, rate_down_se, 2.0 * rate_bump / _PER_PERCENT),
        "carry_rho": difference_error(carry_up_se, carry_down_se, 2.0 * carry_bump / _PER_PERCENT),
        # Paired, not quadrature. The two runs share every increment but one,
        # so adding their errors in quadrature overstates the difference's
        # error by a factor of four or more -- it reached 12,412% of the value
        # at a fine step, which says nothing about the estimate at all.
        "theta": theta_se,
    }

    risk = SnowballRisk(
        greeks=Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho),
        method=(
            "mc-resimulation-crn" if local_vol is None else "mc-resimulation-crn+ssvi-parallel-vega"
        ),
        price=base_price,
        std_error=std_error,
        carry_rho=carry_rho,
    )
    _log.debug("snowball MC risk", extra={"spot": market.S, **risk.greeks.as_dict()})
    return risk


@dataclass(frozen=True, slots=True)
class RiskComparison:
    """Side-by-side PDE and Monte Carlo risk, with a verdict per Greek."""

    pde: SnowballRisk
    mc: SnowballRisk
    verdicts: dict[str, dict[str, object]]

    @property
    def agree(self) -> bool:
        """True when every compared Greek is inside its error budget."""
        return all(bool(v["within_budget"]) for v in self.verdicts.values())

    def as_dict(self) -> dict[str, object]:
        """Flat dict for API responses."""
        return {
            "pde": self.pde.as_dict(),
            "mc": self.mc.as_dict(),
            "verdicts": self.verdicts,
            "agree": self.agree,
        }


def _paired_theta(
    base: SnowballMCResult,
    seasoned: SnowballMCResult,
    step: float,
    antithetic: bool,
) -> tuple[float, float]:
    """Theta and its own standard error from two paired runs.

    The error comes from the per-path *difference*, which is the only honest
    way to report it: under common random numbers the two runs share most of
    their noise, so quadrature on their separate errors describes a pair of
    independent runs that were never performed. Measured on the reference
    trade, quadrature gives 347% of the value where this gives 89%.

    Antithetic pairs are averaged before the variance is taken, exactly as
    :func:`price_snowball_mc` does for the price: the two halves of a pair are
    negatively correlated by construction, so treating them as independent
    samples would overstate the error again, in the other direction from
    quadrature.

    Returns:
        ``(theta, std_error)`` per calendar day.
    """
    if base.path_values is None or seasoned.path_values is None:
        raise EngineError("a paired theta needs per-path values from both runs")

    difference = np.asarray(seasoned.path_values - base.path_values, dtype=float)
    scale = step * _DAYS_PER_YEAR
    theta = float(difference.mean()) / scale

    samples = difference
    if antithetic:
        half = difference.size // 2
        if half:
            samples = 0.5 * (difference[:half] + difference[half : 2 * half])
    std_error = float(np.std(samples, ddof=1) / np.sqrt(samples.size)) / scale
    return theta, std_error


def _check_supplied_risk(
    supplied: SnowballRisk,
    terms: SnowballTerms,
    market: EquityMarket,
    config: PDEConfig,
    local_vol: LocalVolatility | None,
) -> None:
    """Refuse a result that was taken around a different contract.

    A :class:`SnowballRisk` does not carry the terms it came from, so this
    cannot be proved. What it can do is reprice the contract that *was* passed
    in and compare, which costs one solve against the seven being saved and
    catches the realistic mistake -- a caller reusing a result after changing
    the coupon, the barriers or the market.

    The tolerance is the grid's own reproducibility rather than a round
    number: the same inputs through the same solver give the same price, so
    anything beyond a few ULP means different inputs.
    """
    fresh = price_snowball_pde(terms, market, config, local_vol=local_vol).price
    if abs(fresh - supplied.price) > 1e-12 * max(abs(fresh), 1.0):
        raise EngineError(
            f"the supplied risk was taken around a price of {supplied.price:.12g}, "
            f"but these terms and market price to {fresh:.12g}. Reconciling two "
            f"engines against different contracts would report agreement that "
            f"means nothing -- recompute the risk, or let this function do it."
        )


def compare_snowball_risk(
    terms: SnowballTerms,
    market: EquityMarket,
    *,
    pde_config: PDEConfig | None = None,
    mc_config: MonteCarloConfig | None = None,
    local_vol: LocalVolatility | None = None,
    pde: SnowballRisk | None = None,
    mc: SnowballRisk | None = None,
) -> RiskComparison:
    """Run both methods and report whether they agree, Greek by Greek.

    The budget for each Greek is built from measured quantities rather than
    chosen: the Monte Carlo run's own standard error, widened to cover the
    PDE's first-order discretisation bias. Reporting the budget alongside the
    verdict matters, because a pass carried entirely by a wide sampling error
    is weaker evidence than a pass with both methods tight, and the caller
    should be able to tell the two apart.

    Gamma is compared on a relative tolerance rather than an absolute one, and
    a generous one. The Monte Carlo estimator for it is intrinsically noisy on
    a discontinuous payoff, so the useful question is whether the two agree in
    sign and magnitude, not to three digits.

    Theta is compared in a block of its own. It is the last of the Greeks to
    get a second engine, and the comparison is weaker than the others in a way
    worth seeing rather than averaging away: the two are difference quotients
    over horizons four times apart, and three sampling errors come to about
    80% of the value. Both facts are reported alongside the verdict.

    Passing risk that is already computed
    -------------------------------------
    ``pde`` and ``mc`` let a caller hand in a result instead of paying for it
    twice. The cost is not marginal: a PDE risk computation is **seven solves**
    -- the base, plus a pair each for vega, rho and carry_rho, since delta and
    gamma come off the surface the base solve already built.

    ``/api/snowball_report`` was paying it twice. It computed the PDE risk for
    the report's own risk section and then called this function, which computed
    the identical thing again. Fourteen solves where seven would do, and on the
    local-volatility path a solve is 4.4 seconds.

    What is *not* allowed is handing in a result computed against different
    inputs, which would produce a reconciliation of two unrelated numbers while
    looking like a pass. There is no way to verify that from here -- a
    ``SnowballRisk`` does not carry the terms it came from -- so the price it
    was taken around is checked against a fresh price of the contract supplied,
    which catches the realistic version of the mistake.

    Args:
        terms: Contract terms.
        market: Market state.
        pde_config: Grid for the PDE side. Defaults by path; see
            :data:`~otcpricer.engines.pde.LOCAL_VOL_PDE_CONFIG`.
        mc_config: Simulation controls for the Monte Carlo side.
        local_vol: A Dupire surface to compare on, instead of ``market.sigma``.
        pde: An already-computed PDE risk for *these* terms and market.
        mc: An already-computed Monte Carlo risk, likewise.

    Returns:
        A :class:`RiskComparison`.

    Raises:
        EngineError: if a supplied result was taken around a materially
            different price from the contract passed in.
    """
    # The flat grid's budget does not carry over to a Dupire surface; see
    # LOCAL_VOL_PDE_CONFIG for the measured convergence behind this.
    if pde_config is None:
        pde_config = (
            PDEConfig(space_nodes=1600, steps_per_year=1008)
            if local_vol is None
            else LOCAL_VOL_PDE_CONFIG
        )
    mc_config = mc_config or MonteCarloConfig(paths=200_000)

    if pde is None:
        pde = snowball_risk_pde(terms, market, pde_config, local_vol=local_vol)
    else:
        _check_supplied_risk(pde, terms, market, pde_config, local_vol)
    if mc is None:
        mc = snowball_risk_mc(terms, market, mc_config, local_vol=local_vol)

    def read(risk: SnowballRisk, name: str) -> float:
        """One Greek off a result, whether it lives on Greeks or beside it.

        ``carry_rho`` is on :class:`SnowballRisk` rather than in
        :class:`~otcpricer.core.types.Greeks`, which is a fixed five-field
        record shared with the vanilla products. The lookup is widened here so
        the budget logic stays in one place rather than being copied for one
        sensitivity.
        """
        if name == "carry_rho":
            return float("nan") if risk.carry_rho is None else float(risk.carry_rho)
        return float(getattr(risk.greeks, name))

    verdicts: dict[str, dict[str, object]] = {}
    for name in ("delta", "vega", "rho", "carry_rho"):
        pde_value = read(pde, name)
        mc_value = read(mc, name)
        sampling = float((mc.std_error or {}).get(name, 0.0))
        # Three sampling errors, floored at 2% of the PDE value so the budget
        # stays meaningful where the Monte Carlo happens to be very precise but
        # the PDE still carries its own bias.
        budget = max(3.0 * sampling, 0.02 * abs(pde_value))
        gap = abs(pde_value - mc_value)
        verdicts[name] = {
            "pde": pde_value,
            "mc": mc_value,
            "gap": gap,
            "budget": budget,
            "mc_std_error": sampling,
            "within_budget": gap <= budget,
        }

    # Gamma is judged against the Monte Carlo's own sampling error, exactly as
    # the others are. Requiring the *signs* to match would be incoherent: the
    # estimator is declared unreliable here, and when the true gamma is small
    # relative to its sampling error the sign is simply not determined. On a
    # trade with a remote knock-in, gamma is around -2.9e-08 against a Monte
    # Carlo standard error near 2e-07, so the simulated sign is a coin toss
    # while the two values are entirely consistent.
    pde_gamma = float(pde.greeks.gamma)
    mc_gamma = float(mc.greeks.gamma)
    gamma_sampling = float((mc.std_error or {}).get("gamma", 0.0))
    gamma_gap = abs(pde_gamma - mc_gamma)
    gamma_budget = max(3.0 * gamma_sampling, 0.5 * abs(pde_gamma))
    verdicts["gamma"] = {
        "pde": pde_gamma,
        "mc": mc_gamma,
        "gap": gamma_gap,
        "budget": gamma_budget,
        "mc_std_error": gamma_sampling,
        "signs_agree": (pde_gamma > 0) == (mc_gamma > 0),
        "within_budget": gamma_gap <= gamma_budget,
        "note": (
            "Monte Carlo gamma divides a second difference of a discontinuous "
            "estimator by h^2, so its sampling error is large and can exceed "
            "the value itself. Judged against that error, not on sign. Use the "
            "PDE value."
        ),
    }

    # Theta gets its own block, for two reasons that the main loop's budget
    # would hide.
    #
    # First, the two numbers are *different difference quotients* of the same
    # derivative. The PDE reads two adjacent levels of its own backward solve,
    # so its step is the grid's dt -- 1/1008 of a year at the standard config.
    # The Monte Carlo advances the valuation date by one of *its* steps, 1/252.
    # Both estimate dV/dt, over horizons four times apart, so they differ by an
    # O(h) term before either is wrong about anything.
    #
    # Second, three sampling errors come to roughly 80% of theta's value at
    # 200,000 paths. A pass inside that budget is weak evidence, and reporting
    # it beside delta's -- where the budget is 27% of the value -- would make
    # the two look equally established. The ratio is published so a reader can
    # see which it is.
    pde_theta = float(pde.greeks.theta)
    mc_theta = float(mc.greeks.theta)
    theta_sampling = float((mc.std_error or {}).get("theta", 0.0))
    theta_gap = abs(pde_theta - mc_theta)
    theta_budget = 3.0 * theta_sampling
    verdicts["theta"] = {
        "pde": pde_theta,
        "mc": mc_theta,
        "gap": theta_gap,
        "budget": theta_budget,
        "mc_std_error": theta_sampling,
        "mc_error_share_of_value": (theta_budget / abs(mc_theta) if mc_theta else float("inf")),
        "signs_agree": (pde_theta > 0) == (mc_theta > 0),
        "within_budget": theta_gap <= theta_budget,
        "note": (
            "Different difference quotients of the same derivative: the PDE "
            "steps by its grid's dt (1/1008 y), the simulation by one of its "
            "own steps (1/252 y), so an O(h) gap is expected before either is "
            "wrong. The simulated theta comes from advancing the valuation "
            "date, not from shortening the tenor -- shortening it deletes a "
            "knock-out observation and measures a contract change instead of "
            "decay. Its error is a paired difference under common random "
            "numbers, and at 3 standard errors it is a large fraction of the "
            "value: see mc_error_share_of_value before reading the pass as "
            "strong."
        ),
    }

    return RiskComparison(pde=pde, mc=mc, verdicts=verdicts)
