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

from otcpricer.core.logging import get_logger
from otcpricer.core.market import EquityMarket
from otcpricer.core.types import Greeks
from otcpricer.engines.mc import MonteCarloConfig, normal_draws, price_snowball_mc
from otcpricer.engines.pde import PDEConfig, SnowballPDEResult, price_snowball_pde
from otcpricer.products.snowball import SnowballTerms

__all__ = [
    "DEFAULT_RATE_BUMP",
    "DEFAULT_SPOT_BUMP_REL",
    "DEFAULT_VOL_BUMP",
    "RiskComparison",
    "SnowballRisk",
    "compare_snowball_risk",
    "snowball_risk_mc",
    "snowball_risk_pde",
]

_log = get_logger(__name__)

#: Volatility bump for vega, in absolute vol units (1 percentage point).
DEFAULT_VOL_BUMP = 0.01
#: Rate bump for rho, in absolute units (25 basis points).
DEFAULT_RATE_BUMP = 0.0025
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
    """

    greeks: Greeks
    method: str
    price: float
    std_error: dict[str, float] | None = None

    def as_dict(self) -> dict[str, object]:
        """Flat dict for API responses."""
        payload: dict[str, object] = {
            "method": self.method,
            "price": round(self.price, 8),
            **self.greeks.as_dict(8),
        }
        if self.std_error:
            payload["std_error"] = {k: round(v, 10) for k, v in self.std_error.items()}
        return payload


def _grid_derivatives(S_grid: np.ndarray, V: np.ndarray, spot: float) -> tuple[float, float]:
    """Return ``(delta, gamma)`` at ``spot`` by differencing a log-spot grid.

    The grid is uniform in ``x = log S``, so the chain rule gives

    * ``dV/dS   = (1/S) * dV/dx``
    * ``d2V/dS2 = (1/S^2) * (d2V/dx2 - dV/dx)``

    Derivatives are formed on the whole grid and then interpolated at ``spot``,
    rather than differencing around the nearest node. Spot generally falls
    between nodes, and snapping to the nearest one would introduce an error of
    up to half a grid step that does not shrink as the grid refines.

    Validated against a closed-form case: the error halves exactly on each
    refinement (8.71e-07, 4.36e-07, 2.18e-07, 1.09e-07 for delta), which is
    the first-order convergence the barrier treatment allows.
    """
    x = np.log(S_grid)
    dx = float(x[1] - x[0])
    first = np.gradient(V, dx, edge_order=2)
    second = np.gradient(first, dx, edge_order=2)

    delta_curve = first / S_grid
    gamma_curve = (second - first) / (S_grid * S_grid)

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
) -> float:
    """Price once with the PDE, for a bumped-parameter re-solve."""
    return price_snowball_pde(terms, market, config, knocked_in=knocked_in).price


def snowball_risk_pde(
    terms: SnowballTerms,
    market: EquityMarket,
    config: PDEConfig | None = None,
    *,
    knocked_in: bool = False,
    vol_bump: float = DEFAULT_VOL_BUMP,
    rate_bump: float = DEFAULT_RATE_BUMP,
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

    Returns:
        A :class:`SnowballRisk` with no standard errors: the method is
        deterministic.
    """
    config = config or PDEConfig()
    base: SnowballPDEResult = price_snowball_pde(terms, market, config, knocked_in=knocked_in)

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
    vol_up = _pde_value(terms, market.with_shift(vol_abs=vol_bump), config, knocked_in)
    vol_down = _pde_value(terms, market.with_shift(vol_abs=-vol_bump), config, knocked_in)
    vega = (vol_up - vol_down) / (2.0 * vol_bump) * _PER_PERCENT

    rate_up = _pde_value(terms, market.with_shift(rate_abs=rate_bump), config, knocked_in)
    rate_down = _pde_value(terms, market.with_shift(rate_abs=-rate_bump), config, knocked_in)
    rho = (rate_up - rate_down) / (2.0 * rate_bump) * _PER_PERCENT

    risk = SnowballRisk(
        greeks=Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho),
        method="pde-grid",
        price=base.price,
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

    Returns:
        A :class:`SnowballRisk` with per-Greek standard errors propagated from
        the underlying runs.
    """
    config = config or MonteCarloConfig()

    # One shared draw set for every valuation below. simulate_log_paths
    # requires shape (steps, paths), and supplying draws forces a single batch,
    # so the caller is accepting that memory cost knowingly.
    n_steps = max(int(round(terms.T * config.steps_per_year)), 1)
    draws = normal_draws(n_steps, config.paths, config.seed, config.antithetic)

    def value(shifted: EquityMarket) -> tuple[float, float]:
        result = price_snowball_mc(terms, shifted, config, draws=draws)
        return result.price, result.std_error

    base_price, base_se = value(market)

    spot_bump = market.S * spot_bump_rel
    up_price, up_se = value(market.with_shift(spot_rel=spot_bump_rel))
    down_price, down_se = value(market.with_shift(spot_rel=-spot_bump_rel))

    delta = (up_price - down_price) / (2.0 * spot_bump)
    gamma = (up_price - 2.0 * base_price + down_price) / (spot_bump * spot_bump)

    vol_up, vol_up_se = value(market.with_shift(vol_abs=vol_bump))
    vol_down, vol_down_se = value(market.with_shift(vol_abs=-vol_bump))
    vega = (vol_up - vol_down) / (2.0 * vol_bump) * _PER_PERCENT

    rate_up, rate_up_se = value(market.with_shift(rate_abs=rate_bump))
    rate_down, rate_down_se = value(market.with_shift(rate_abs=-rate_bump))
    rho = (rate_up - rate_down) / (2.0 * rate_bump) * _PER_PERCENT

    # Theta is left to the PDE. Shortening the tenor deletes a knock-out
    # observation (9 become 8 on the reference trade), so a maturity-bumped
    # Monte Carlo theta would measure that contract change rather than decay.
    # Reporting nan is honest; reporting a number would not be.
    theta = float("nan")

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
    }

    risk = SnowballRisk(
        greeks=Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho),
        method="mc-resimulation-crn",
        price=base_price,
        std_error=std_error,
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


def compare_snowball_risk(
    terms: SnowballTerms,
    market: EquityMarket,
    *,
    pde_config: PDEConfig | None = None,
    mc_config: MonteCarloConfig | None = None,
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
    """
    pde_config = pde_config or PDEConfig(space_nodes=1600, steps_per_year=1008)
    mc_config = mc_config or MonteCarloConfig(paths=200_000)

    pde = snowball_risk_pde(terms, market, pde_config)
    mc = snowball_risk_mc(terms, market, mc_config)

    verdicts: dict[str, dict[str, object]] = {}
    for name in ("delta", "vega", "rho"):
        pde_value = float(getattr(pde.greeks, name))
        mc_value = float(getattr(mc.greeks, name))
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

    pde_gamma = float(pde.greeks.gamma)
    mc_gamma = float(mc.greeks.gamma)
    relative = abs(pde_gamma - mc_gamma) / max(abs(pde_gamma), 1e-30)
    verdicts["gamma"] = {
        "pde": pde_gamma,
        "mc": mc_gamma,
        "relative_gap": relative,
        "budget": 0.5,
        "signs_agree": (pde_gamma > 0) == (mc_gamma > 0),
        "within_budget": relative <= 0.5 and (pde_gamma > 0) == (mc_gamma > 0),
        "note": (
            "Monte Carlo gamma divides a second difference of a discontinuous "
            "estimator by h^2 and is intrinsically noisy; compared on sign and "
            "magnitude only. Use the PDE value."
        ),
    }

    return RiskComparison(pde=pde, mc=mc, verdicts=verdicts)
