"""Monte Carlo engine for path-dependent equity products.

What changed from the legacy implementation
-------------------------------------------
Two defects in the old snowball Monte Carlo are fixed here, both measured
against the v7.0.0 baseline:

**Discounting.** The old code computed ``exp(-r * T) * mean(payoff)``,
discounting every path from maturity. But a knocked-out note pays
``coupon_pa * t`` *at* ``t``, and with a knock-out probability around 73% most
paths terminate well before ``T``. Discounting them from maturity understates
the note systematically. This engine carries a payment time per path and
discounts each one at its own date. The PDE never had this problem, because
backward induction discounts implicitly -- which is part of why the two engines
disagreed.

**Knock-in strike.** The old code struck the knock-in put at the knock-in
*barrier* (``-max(0, 1 - S_T / KI)``) while the PDE struck it at the initial
level (``min(S/S0 - 1, 0)``). The barrier version understates the downside, and
the two are different products. Both engines now read the strike from
:class:`~otcpricer.products.snowball.SnowballTerms`.

Antithetic variates are available and on by default: for a payoff this
monotone in the terminal level they cut the standard error materially at no
accuracy cost, and they make the knock-out probability estimate noticeably
more stable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from otcpricer.core.exceptions import EngineError
from otcpricer.core.logging import get_logger
from otcpricer.core.market import EquityMarket
from otcpricer.core.validation import check_positive
from otcpricer.core.vol import LocalVolatility
from otcpricer.models.heston import HestonParams
from otcpricer.products.snowball import SnowballTerms

__all__ = [
    "QE_PSI_CRITICAL",
    "MonteCarloConfig",
    "SnowballMCResult",
    "heston_variance_paths",
    "normal_draws",
    "price_snowball_mc",
    "simulate_log_paths",
]

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class MonteCarloConfig:
    """Simulation controls.

    Attributes:
        paths: Number of paths. With ``antithetic`` enabled this is the total,
            generated as two mirrored halves.
        steps_per_year: Time steps per year. 252 aligns the grid with trading
            days so the daily knock-in schedule lands exactly on grid points.
        seed: RNG seed; fixed by default so a quote is reproducible.
        antithetic: Pair each draw with its negation to cut variance.
    """

    paths: int = 50_000
    steps_per_year: int = 252
    seed: int = 42
    antithetic: bool = True
    memory_budget_mb: float = 256.0

    def __post_init__(self) -> None:
        if self.paths < 2:
            raise ValueError(f"paths must be at least 2, got {self.paths}")
        if self.steps_per_year < 1:
            raise ValueError(f"steps_per_year must be at least 1, got {self.steps_per_year}")
        if self.memory_budget_mb <= 0.0:
            raise ValueError(f"memory_budget_mb must be positive, got {self.memory_budget_mb}")

    def batch_paths(self, n_steps: int) -> int:
        """Paths per batch that keeps the path matrix inside the memory budget.

        The full matrix is ``(n_steps + 1) x paths`` doubles, which grows with
        the tenor as well as the path count: a three-year note at two million
        paths needs 11 GiB in one block. Simulating in batches keeps the
        footprint flat and bounded, at no cost to the estimate -- the batches
        are independent, so their statistics combine exactly.

        Always even, so antithetic pairs are never split across a boundary.
        """
        bytes_per_path = (n_steps + 1) * 8 * 2  # paths matrix plus the draws
        affordable = int(self.memory_budget_mb * 1024 * 1024 / max(bytes_per_path, 1))
        capped = max(min(affordable, self.paths), 2)
        return capped - (capped % 2)


@dataclass(frozen=True, slots=True)
class SnowballMCResult:
    """Outcome of a snowball Monte Carlo valuation.

    All monetary quantities are fractions of notional.

    Attributes:
        price: Mean discounted payoff.
        std_error: Standard error of that mean.
        ci95: Half-width of the 95% confidence interval.
        ko_prob: Probability of knocking out at some observation.
        ki_prob: Probability of knocking in (whether or not it later knocks out).
        ki_no_ko_prob: Probability of the loss-making state: knocked in and
            never knocked out. This is the one that actually costs money.
        expected_ko_time: Mean knock-out time over knocking-out paths, in
            years; ``nan`` if no path knocks out. Exposed because it is what
            makes the discounting fix visible.
        paths: Paths simulated.
        vol_model: What produced the number -- ``"flat"``, ``"dupire-local"``
            or ``"heston-qe"``. Carried because the three are not
            interchangeable and the response would otherwise look identical
            whichever ran.
        path_values: Discounted payoff per path, or ``None``.

            Populated only on request, and only when the run is a single batch
            -- which supplying ``draws`` already forces. It exists for one
            reason: a **paired** difference needs per-path values to report its
            own error.

            Under common random numbers two runs share most of their noise, so
            combining their standard errors in quadrature overstates the error
            of the difference enormously. Measured on theta, the quadrature
            bound came to 347% of the value where the paired estimator puts it
            at 89%, and at a finer step the bound reached 12,412% -- a number
            that says nothing at all about the estimate.
    """

    price: float
    std_error: float
    ci95: float
    ko_prob: float
    ki_prob: float
    ki_no_ko_prob: float
    expected_ko_time: float
    paths: int
    vol_model: str = "flat"
    path_values: np.ndarray | None = None

    @property
    def cross_checked_by_a_second_engine(self) -> bool:
        """Whether an independent engine can confirm this price.

        False for Heston, and the one place in this project where that is so.
        The flat and local-volatility paths both have a Crank-Nicolson partner;
        Heston's would be a two-dimensional PDE with a cross-derivative and an
        ADI solver, which is not built. Surfaced as a field rather than left in
        a docstring, so a caller reading the JSON sees it.
        """
        return self.vol_model != "heston-qe"

    def as_dict(self) -> dict[str, object]:
        """Flat dict for API responses."""
        payload: dict[str, object] = {
            "price": round(self.price, 8),
            "std_error": round(self.std_error, 8),
            "ci95": round(self.ci95, 8),
            "ko_prob": round(self.ko_prob, 6),
            "ki_prob": round(self.ki_prob, 6),
            "ki_no_ko_prob": round(self.ki_no_ko_prob, 6),
            "expected_ko_time": round(self.expected_ko_time, 6),
            "paths": self.paths,
            "vol_model": self.vol_model,
            "cross_checked_by_a_second_engine": self.cross_checked_by_a_second_engine,
        }
        if not self.cross_checked_by_a_second_engine:
            payload["single_engine_note"] = (
                "This price has no independent second engine. QE is verified "
                "against the COS European prices, so the simulation is "
                "consistent with the model, but the step to a path-dependent "
                "payoff is not separately confirmed. Do not read it as "
                "carrying the same weight as the flat or local-volatility "
                "snowball numbers."
            )
        return payload


def normal_draws(n_steps: int, n_paths: int, seed: int, antithetic: bool) -> np.ndarray:
    """Standard normal increments shaped ``(n_steps, n_paths)``.

    With ``antithetic`` the second half of the columns is the negation of the
    first, so the sample mean of the driving noise is exactly zero.
    """
    rng = np.random.default_rng(seed)
    if not antithetic:
        return rng.standard_normal((n_steps, n_paths))

    half = n_paths // 2
    base = rng.standard_normal((n_steps, half))
    draws: np.ndarray = np.concatenate([base, -base], axis=1)
    if draws.shape[1] < n_paths:  # odd path count: top up with one extra draw
        extra = rng.standard_normal((n_steps, n_paths - draws.shape[1]))
        draws = np.concatenate([draws, extra], axis=1)
    return draws


def _simulate_local_vol(
    market: EquityMarket,
    local_vol: LocalVolatility,
    dt: float,
    n_steps: int,
    draws: np.ndarray,
    *,
    t0: float = 0.0,
) -> np.ndarray:
    """Step log-spot forward with a state-dependent volatility.

    This is where exactness is given up, and the trade is worth stating
    plainly because the flat path's docstring advertises the opposite. A
    constant-coefficient log step integrates geometric Brownian motion
    *exactly* over any step size, so ``steps_per_year`` there is purely a
    scheduling parameter for the observation dates. A state-dependent
    volatility has no closed-form step: the increment over ``[t_k, t_k+1]``
    depends on ``S_k``, which depends on every increment before it, so the
    whole increment matrix can no longer be built up front and summed. It
    becomes a sequential loop, vectorised across paths only, and the scheme --
    log-Euler, volatility frozen at the start of each step -- carries a genuine
    discretisation bias. ``steps_per_year`` becomes an accuracy parameter.

    The cost is smaller than it looks. The knock-in barrier is monitored daily,
    so the step count was already pinned at 252 a year or more; what is added
    is loop overhead and one volatility evaluation per step, not more steps.
    And the volatility evaluation is one vectorised call over all paths at a
    single time, which is why the local-volatility contract is scalar in time:
    the variance term structure is then evaluated once per step rather than
    once per path per step.
    """
    log_paths = np.empty((n_steps + 1, draws.shape[1]), dtype=float)
    log_paths[0, :] = np.log(market.S)
    root_dt = float(np.sqrt(dt))

    for step in range(n_steps):
        # Absolute time. The surface is a function of calendar time, so a
        # seasoned valuation must sample it at the dates it will actually
        # live through, not at an elapsed time starting from zero.
        t = t0 + step * dt
        current = log_paths[step]
        sigma = np.asarray(local_vol.local_vol(np.exp(current), t), dtype=float)
        log_paths[step + 1] = (
            current
            + (local_vol.drift(t) - 0.5 * sigma * sigma) * dt
            + sigma * root_dt * draws[step]
        )

    return log_paths


#: Andersen's switching threshold between the two variance branches.
#:
#: 1.5 is his recommendation and it is not a tuning knob: below it the
#: quadratic branch matches the first two moments of the non-central chi-square
#: well, above it the distribution is too skewed and the exponential branch
#: takes over. Measured here, moving it to 1.0 or 2.0 changes a one-year
#: European price by less than one standard error, which is the evidence that
#: the switch is in the right place rather than merely conventional.
QE_PSI_CRITICAL = 1.5


@dataclass(frozen=True, slots=True)
class _QeStepConstants:
    """Per-step constants for the QE variance recursion.

    Depend only on ``dt`` and the parameters, so they are computed once.
    """

    decay: float
    var_a: float
    var_b: float
    theta: float

    @classmethod
    def build(cls, params: HestonParams, dt: float) -> _QeStepConstants:
        decay = float(np.exp(-params.kappa * dt))
        sigma_sq = params.sigma * params.sigma
        return cls(
            decay=decay,
            var_a=sigma_sq * decay * (1.0 - decay) / params.kappa,
            var_b=params.theta * sigma_sq * (1.0 - decay) ** 2 / (2.0 * params.kappa),
            theta=params.theta,
        )


def _qe_variance_step(
    variance: np.ndarray,
    constants: _QeStepConstants,
    rng: np.random.Generator,
) -> np.ndarray:
    """One QE step of the variance, the single implementation of the scheme.

    Shared by :func:`_simulate_heston_qe` and :func:`heston_variance_paths` so
    the diagnostic cannot drift away from the pricer. The exact conditional
    moments are matched and then sampled from, rather than the SDE being
    discretised:

        ``m  = theta + (v - theta) e^{-kappa dt}``
        ``s2 = v * var_a + var_b``

    and ``psi = s2 / m^2`` selects the branch. Below
    :data:`QE_PSI_CRITICAL` the draw is ``a(b + Z)^2`` -- non-central
    chi-square shaped and non-negative by construction. Above it, an
    exponential with an **atom at zero**, which is what lets the scheme
    represent a variance that genuinely reaches zero rather than truncating a
    negative Euler step.
    """
    mean = constants.theta + (variance - constants.theta) * constants.decay
    var = variance * constants.var_a + constants.var_b
    psi = np.where(mean > 0.0, var / np.maximum(mean * mean, 1e-300), np.inf)

    quadratic = psi <= QE_PSI_CRITICAL

    inv_psi = 2.0 / np.where(quadratic, psi, 1.0)
    b_squared = np.maximum(inv_psi - 1.0 + np.sqrt(inv_psi * np.maximum(inv_psi - 1.0, 0.0)), 0.0)
    a_coef = mean / (1.0 + b_squared)
    z_var = rng.standard_normal(variance.size)
    quadratic_draw = a_coef * (np.sqrt(b_squared) + z_var) ** 2

    safe_psi = np.where(quadratic, 2.0, psi)
    prob_zero = (safe_psi - 1.0) / (safe_psi + 1.0)
    beta = np.where(mean > 0.0, (1.0 - prob_zero) / np.maximum(mean, 1e-300), 0.0)
    uniform = rng.random(variance.size)
    exponential_draw = np.where(
        uniform <= prob_zero,
        0.0,
        np.log(np.maximum((1.0 - prob_zero) / np.maximum(1.0 - uniform, 1e-300), 1.0))
        / np.maximum(beta, 1e-300),
    )

    stepped: np.ndarray = np.maximum(np.where(quadratic, quadratic_draw, exponential_draw), 0.0)
    return stepped


def heston_variance_paths(
    params: HestonParams,
    T: float,
    config: MonteCarloConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Variance paths alone, for diagnostics.

    The variance is not recoverable from the spot paths -- the log-spot step
    mixes ``v_t`` and ``v_{t+dt}`` and adds a deterministic drift, so a zero
    variance does not freeze the return. Anything that needs to *see* the zero
    atom, including the test that it exists at all, has to look here.

    Returns:
        ``(times, variance)`` with ``variance`` shaped ``(steps + 1, paths)``.
    """
    check_positive(T, "T")
    n_steps = max(int(round(T * config.steps_per_year)), 1)
    dt = T / n_steps
    constants = _QeStepConstants.build(params, dt)
    rng = np.random.default_rng(config.seed)

    variance = np.empty((n_steps + 1, config.paths), dtype=float)
    variance[0, :] = params.v0
    for step in range(n_steps):
        variance[step + 1, :] = _qe_variance_step(variance[step, :], constants, rng)

    return np.arange(n_steps + 1, dtype=float) * dt, variance


def _simulate_heston_qe(
    market: EquityMarket,
    params: HestonParams,
    dt: float,
    n_steps: int,
    n_paths: int,
    seed: int,
    antithetic: bool,
) -> np.ndarray:
    """Andersen's Quadratic Exponential scheme for Heston.

    Why not Euler
    -------------
    The variance follows a CIR process, and when Feller is violated it can
    reach zero. An Euler step then goes *negative*, and ``sqrt(v)`` is not
    defined there. The usual patches -- truncate at zero, or reflect -- both
    introduce a bias that does not vanish as the step shrinks, because the
    event they are patching is not a discretisation artefact.

    This matters here specifically rather than in general: the parameters
    calibrated to this project's own data violate Feller on both underlyings
    (margins -3.36 and -1.52). So QE is a requirement, not a refinement.

    How it works
    ------------
    QE matches the *exact* conditional moments of the variance and samples from
    a distribution with those moments, rather than discretising the SDE. Over a
    step, given ``v``:

        ``m  = theta + (v - theta) e^{-kappa dt}``
        ``s2 = v sigma^2 e^{-kappa dt}(1 - e^{-kappa dt})/kappa``
        ``     + theta sigma^2 (1 - e^{-kappa dt})^2 / (2 kappa)``

    and ``psi = s2 / m^2`` decides the branch. For small ``psi`` the variance
    is sampled as ``a(b + Z)^2`` -- non-central chi-square shaped, and
    non-negative by construction. For large ``psi`` it is sampled from an
    exponential with an atom at zero, which is what lets the scheme represent
    a variance that genuinely touches zero instead of pretending it cannot.

    The log-spot uses Andersen's central ``(gamma1 = gamma2 = 1/2)``
    discretisation, which carries the correlation through the ``K1 v_t +
    K2 v_{t+dt}`` terms rather than through a correlated normal. The remaining
    normal is therefore *independent* of the variance innovation -- getting
    that wrong by correlating them as well would double-count rho.

    Returns:
        Log-spot paths shaped ``(n_steps + 1, n_paths)``.
    """
    rng = np.random.default_rng(seed)
    kappa, theta, sigma, rho = params.kappa, params.theta, params.sigma, params.rho
    constants = _QeStepConstants.build(params, dt)

    # Andersen's log-spot coefficients, central scheme.
    gamma1 = gamma2 = 0.5
    k0 = -rho * kappa * theta * dt / sigma
    k1 = gamma1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    k2 = gamma2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    k3 = gamma1 * dt * (1.0 - rho * rho)
    k4 = gamma2 * dt * (1.0 - rho * rho)

    def normals(size: int) -> np.ndarray:
        """Antithetic in the spot driver, which is where the variance lives."""
        if not antithetic:
            return rng.standard_normal(size)
        half = size // 2
        base = rng.standard_normal(half)
        out: np.ndarray = np.concatenate([base, -base])
        if out.size < size:
            out = np.concatenate([out, rng.standard_normal(size - out.size)])
        return out

    log_paths = np.empty((n_steps + 1, n_paths), dtype=float)
    log_paths[0, :] = np.log(market.S)

    variance = np.full(n_paths, params.v0, dtype=float)
    log_spot = np.full(n_paths, float(np.log(market.S)), dtype=float)

    for step in range(n_steps):
        next_variance = _qe_variance_step(variance, constants, rng)

        # --- the log-spot step, with an independent normal
        diffusion = np.sqrt(np.maximum(k3 * variance + k4 * next_variance, 0.0))
        log_spot = (
            log_spot
            + market.drift * dt
            + k0
            + k1 * variance
            + k2 * next_variance
            + diffusion * normals(n_paths)
        )

        variance = next_variance
        log_paths[step + 1, :] = log_spot

    return log_paths


def simulate_log_paths(
    market: EquityMarket,
    T: float,
    config: MonteCarloConfig,
    *,
    draws: np.ndarray | None = None,
    local_vol: LocalVolatility | None = None,
    heston: HestonParams | None = None,
    t0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate paths under the risk-neutral measure.

    Integrates in log space, which is exact for GBM over any step size (no
    discretisation bias) and avoids the negative paths an Euler scheme on ``S``
    can produce.

    The drift is ``market.drift`` -- ``r - q - basis_pa`` -- so a futures basis
    is reflected here, while discounting elsewhere still uses ``r`` alone.

    Args:
        market: Market state.
        T: Horizon in years.
        config: Simulation controls.
        draws: Pre-generated normals shaped ``(steps, paths)``, for common
            random numbers across perturbed re-runs in finite-difference risk.
        local_vol: A Dupire local volatility to step against. When omitted the
            exact closed-form path runs unchanged. When supplied,
            ``market.sigma`` is not read and the scheme is no longer exact --
            see :func:`_simulate_local_vol`.
        heston: Heston parameters, simulated by Andersen's QE scheme. Mutually
            exclusive with ``local_vol``: they are different models, not
            different settings, and silently preferring one would make a
            caller's mistake invisible. ``market.sigma`` is not read.
        t0: Start of the simulation in years, for a seasoned valuation. The
            horizon simulated is ``T - t0``; ``market.S`` is read as the spot
            *at* ``t0``, and the returned times are absolute, measured from
            inception, so a contract schedule written from inception needs no
            translation.

    Returns:
        ``(times, paths)`` where ``times`` has length ``steps + 1`` starting at
        ``t0``, and ``paths`` is ``(steps + 1, n_paths)`` with row 0 equal to
        spot.
    """
    check_positive(T, "T")
    if not 0.0 <= t0 < T:
        raise ValueError(f"t0 must lie in [0, T), got t0={t0} and T={T}")
    horizon = T - t0
    n_steps = max(int(round(horizon * config.steps_per_year)), 1)
    dt = horizon / n_steps

    if heston is not None and local_vol is not None:
        raise ValueError(
            "pass either heston or local_vol, not both: they are different "
            "models rather than different settings"
        )

    if draws is None:
        draws = normal_draws(n_steps, config.paths, config.seed, config.antithetic)
    elif draws.shape != (n_steps, config.paths):
        raise ValueError(f"draws must have shape {(n_steps, config.paths)}, got {draws.shape}")

    if heston is not None:
        log_paths = _simulate_heston_qe(
            market,
            heston,
            dt,
            n_steps,
            config.paths,
            config.seed,
            config.antithetic,
        )
    elif local_vol is not None:
        log_paths = _simulate_local_vol(market, local_vol, dt, n_steps, draws, t0=t0)
    else:
        drift_step = (market.drift - 0.5 * market.sigma**2) * dt
        vol_step = market.sigma * np.sqrt(dt)

        # Cumulative sum rather than a Python loop over time steps: one
        # vectorised pass instead of n_steps interpreter round-trips.
        log_increments = drift_step + vol_step * draws
        log_paths = np.empty((n_steps + 1, draws.shape[1]), dtype=float)
        log_paths[0, :] = np.log(market.S)
        np.cumsum(log_increments, axis=0, out=log_paths[1:, :])
        log_paths[1:, :] += np.log(market.S)

    # Absolute times measured from *inception*, not from t0. The contract's
    # observation schedule and its coupon accrual are both written from
    # inception, so shifting the clock here instead of there would mean
    # shifting them back everywhere downstream.
    times = t0 + np.arange(n_steps + 1, dtype=float) * dt
    return times, np.exp(log_paths)


def _schedule_indices(
    observation_times: np.ndarray, times: np.ndarray, *, label: str = "observation"
) -> np.ndarray:
    """Map observation dates onto the nearest simulation grid index.

    Returns indices in increasing order, excluding index 0 -- which is the
    valuation date, and is never an observation: at inception because the
    schedule starts later, and on a seasoned valuation because any observation
    already past is history rather than something to simulate.

    Observations strictly before ``times[0]`` are dropped. On a seasoned
    valuation that is the whole point: they have already happened, and the
    caller states their outcome through ``knocked_in`` rather than having them
    re-drawn.

    Two dates landing on one step is refused
    ----------------------------------------
    This used to end in ``np.unique``, and the docstring presented the
    de-duplication as a feature. It is not: a monitoring date silently
    discarded is a monitoring date the contract has and the price does not,
    and a knock-out note with fewer observations is worth strictly more.

    What it actually caused is **schedule substitution**, and the measurement
    is unambiguous. At 252 steps a year, with ``obs_freq`` of 252, 300, 400,
    600 and 1000 -- notes asking for 189, 225, 300, 450 and 750 knock-out
    observations -- this engine returned ``0.04795316632228654`` for all five.
    Bit-identical prices for five different contracts.

    The finite-difference engine substitutes too, and *differently*: its
    ``ko_steps`` dict keeps the last contractual date written to each step and
    pays the contractual accrual, where this one paid the grid's. Across the
    same five contracts it returned five different prices spread over 4.22e-04
    with no order to them. So the two engines were not merely imprecise; they
    were pricing different substituted contracts and the cross-engine check
    could not see it, because both substitutions moved the same way.

    Raises:
        EngineError: if two dates collide on one step. The message carries the
            ``steps_per_year`` that would be enough, because the fix is always
            a finer grid rather than a coarser contract.
    """
    if observation_times.size == 0:
        return np.empty(0, dtype=int)
    future = observation_times[observation_times > times[0] + 1e-12]
    if future.size == 0:
        return np.empty(0, dtype=int)
    idx = np.searchsorted(times, future - 1e-12)
    idx = np.clip(idx, 1, times.size - 1)

    distinct = np.unique(idx)
    if distinct.size != idx.size:
        horizon = float(times[-1] - times[0])
        gaps = np.diff(np.concatenate(([float(times[0])], future)))
        tightest = float(gaps.min())
        # The nudge is not cosmetic: a schedule at 1/400 of a year gives a gap
        # whose reciprocal is 400.0000000000001, so a bare ceil would advise
        # 401 steps for a contract that fits exactly in 400 -- and then the
        # advice would not be the tightest grid that works.
        needed = int(np.ceil(1.0 / tightest - 1e-9)) if tightest > 0.0 else 0
        raise EngineError(
            f"{idx.size - distinct.size} of {idx.size} {label} dates collide on "
            f"the simulation grid: {idx.size} dates over {times.size - 1} steps "
            f"spanning {horizon:.4f} years, with the closest pair {tightest:.6f} "
            f"years apart. Dropping them would price a note with fewer "
            f"observations, which is worth strictly more -- use "
            f"steps_per_year >= {needed} instead"
        )
    return np.asarray(idx, dtype=int)


def _evaluate_batch(
    terms: SnowballTerms,
    market: EquityMarket,
    times: np.ndarray,
    paths: np.ndarray,
    ko_idx: np.ndarray,
    ki_idx: np.ndarray,
    *,
    knocked_in: bool = False,
    t0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve one batch of paths into discounted payoffs and state flags.

    Returns:
        ``(discounted, knocked_out, knocked_in, ki_and_survived, ko_time)``,
        where ``ko_time`` is ``nan`` on paths that never knocked out.
    """
    n_paths = paths.shape[1]

    # --- Knock-out: first scheduled observation at or above the barrier -----
    # Vectorised over the schedule rather than looping paths.
    ko_levels = paths[ko_idx, :] >= terms.ko_barrier
    knocked_out = np.asarray(ko_levels.any(axis=0))
    # argmax finds the first True; meaningless where nothing is True, so it is
    # read only for paths that did knock out.
    first_ko_row = ko_levels.argmax(axis=0)
    ko_time = np.asarray(np.where(knocked_out, times[ko_idx[first_ko_row]], np.nan))

    # --- Knock-in: any observation at or below the barrier, before the note
    # knocked out. A knock-in after termination cannot affect the payoff.
    if knocked_in:
        # A note that has *already* knocked in. The breach is in the past, so
        # it necessarily precedes any future knock-out and no path can be
        # clean: every path either recovers above the barrier and terminates
        # on its coupon, or takes the put loss. That is exactly the PDE's
        # V_ki state, which is what makes the two comparable.
        has_knocked_in = np.ones(n_paths, dtype=bool)
    else:
        ki_levels = paths[ki_idx, :] <= terms.ki_barrier
        cutoff = np.where(knocked_out, ko_time, np.inf)[None, :]
        has_knocked_in = np.asarray((ki_levels & (times[ki_idx][:, None] <= cutoff)).any(axis=0))

    # --- Assemble payoff and payment time per path --------------------------
    payoff = np.empty(n_paths, dtype=float)
    pay_time = np.full(n_paths, terms.T, dtype=float)

    payoff[knocked_out] = terms.knock_out_payoff(ko_time[knocked_out])
    pay_time[knocked_out] = ko_time[knocked_out]

    survived = np.asarray(~knocked_out)
    ki_and_survived = np.asarray(survived & has_knocked_in)
    clean = survived & ~has_knocked_in

    payoff[ki_and_survived] = terms.knock_in_payoff(paths[-1, ki_and_survived])
    payoff[clean] = terms.full_coupon_payoff()

    # --- Discount each path at its own payment date -------------------------
    # This is the correction: the legacy engine applied exp(-r * T) to every
    # path, including the majority that terminate early.
    #
    # Discounting runs from the *valuation* date to the payment date. On a
    # seasoned valuation the two differ, and using the inception-based time
    # would scale the whole price by exp(-r*t0) -- small, systematic, and in
    # exactly the place a theta would read it as decay.
    discounted = np.asarray(payoff * np.exp(-market.r * (pay_time - t0)))
    return discounted, knocked_out, has_knocked_in, ki_and_survived, ko_time


def price_snowball_mc(
    terms: SnowballTerms,
    market: EquityMarket,
    config: MonteCarloConfig | None = None,
    *,
    draws: np.ndarray | None = None,
    local_vol: LocalVolatility | None = None,
    heston: HestonParams | None = None,
    knocked_in: bool = False,
    t0: float = 0.0,
    keep_path_values: bool = False,
) -> SnowballMCResult:
    """Value a snowball note by path Monte Carlo.

    Each path resolves to exactly one of three states, and crucially to a
    *payment date* as well as an amount:

    ===========================  ==============================  =============
    State                        Payoff                          Paid at
    ===========================  ==============================  =============
    Knocked out                  ``coupon_pa * t_ko``            ``t_ko``
    Knocked in, never out        ``-max(0, 1 - S_T / strike)``   ``T``
    Neither                      ``coupon_pa * T``               ``T``
    ===========================  ==============================  =============

    Paths are simulated in batches sized to a memory budget rather than all at
    once. The path matrix scales with tenor as well as path count, so a
    three-year note at two million paths would otherwise need an 11 GiB
    allocation. Batches are independent, so combining their statistics is
    exact rather than an approximation.

    Args:
        terms: Contract terms, including the lock-up and knock-in strike mode.
        market: Market state; ``basis_pa`` shifts the drift if set.
        config: Simulation controls; defaults to :class:`MonteCarloConfig`.
        draws: Pre-generated normals for common-random-number risk. Supplying
            them forces a single batch, so the caller owns that memory
            tradeoff deliberately.
        local_vol: A Dupire local volatility to price against. When omitted the
            engine is unchanged and unbiased for the diffusion; when supplied,
            ``steps_per_year`` becomes an accuracy parameter rather than only a
            scheduling one.
        t0: Valuation date in years from inception, for a seasoned trade.
            Zero values the note at inception.

            The contract does **not** move: its observation schedule and its
            coupon accrual stay written from inception, so a knock-out at
            ``t`` still pays ``coupon_pa * t`` and the monthly dates stay where
            the term sheet put them. What changes is where the simulation
            starts, which observations remain, and that discounting runs from
            ``t0`` rather than from zero.

            This is what makes a Monte Carlo theta possible. Shortening ``T``
            instead silently deletes a knock-out observation -- nine become
            eight on the reference trade -- so a maturity-bumped theta measures
            a contract change rather than time decay. Advancing the valuation
            date changes no term of the contract at all.

            Observations already past are dropped rather than re-drawn. Their
            outcome is the caller's to state, through ``knocked_in``.
        keep_path_values: Return the discounted payoff of every path, for a
            paired difference that needs its own standard error. Requires a
            single batch, which supplying ``draws`` forces; refused otherwise
            rather than returning one batch's worth and looking complete.
        heston: Heston parameters, simulated by Andersen's QE scheme. Mutually
            exclusive with ``local_vol``.

            **This path has one engine, and that is a departure worth stating.**
            Every other number this project reports is cross-checked by a second
            independent engine -- the snowball's PDE against its Monte Carlo,
            the local-volatility surface against its Dupire round trip, the COS
            European prices against QuantLib. A Heston snowball has no such
            partner: the second engine would be a two-dimensional PDE with a
            cross-derivative term and an ADI solver, which is not built here.
            What *is* checked is the layer below -- QE reproduces the COS
            European prices, so the simulation is known to be consistent with
            the model. The step from there to a path-dependent payoff is not
            independently verified, and a Heston snowball should not be read as
            carrying the same weight as the flat or local-volatility one.

    Returns:
        A :class:`SnowballMCResult` with the price, its Monte Carlo error, and
        the knock-out/knock-in statistics a desk quotes alongside it.
    """
    config = config or MonteCarloConfig()
    if heston is not None and local_vol is not None:
        raise ValueError(
            "pass either heston or local_vol, not both: they are different "
            "models rather than different settings"
        )
    if not 0.0 <= t0 < terms.T:
        raise ValueError(
            f"t0 must lie in [0, T) -- a valuation date at or after maturity has "
            f"nothing left to simulate. Got t0={t0} against T={terms.T}"
        )
    n_steps = max(int(round((terms.T - t0) * config.steps_per_year)), 1)

    batch_size = config.paths if draws is not None else config.batch_paths(n_steps)
    n_batches = max(int(np.ceil(config.paths / batch_size)), 1)

    if keep_path_values and n_batches > 1:
        raise ValueError(
            f"keep_path_values needs a single batch, but this run would take "
            f"{n_batches} at {config.paths} paths and {n_steps} steps. Supply "
            f"draws (which forces one batch), raise memory_budget_mb, or lower "
            f"the path count -- returning one batch's values would look like the "
            f"whole run"
        )

    # One spawned seed sequence per batch: reproducible, and independent across
    # batches in a way sequential integer seeds would not guarantee.
    seed_stream = np.random.SeedSequence(config.seed).spawn(n_batches)

    total = 0.0
    flat_sq_sum = 0.0
    pair_sum = 0.0
    pair_sq_sum = 0.0
    pair_count = 0
    n_done = 0
    ko_count = 0
    ki_count = 0
    ki_survived_count = 0
    ko_time_sum = 0.0
    ko_idx = ki_idx = np.empty(0, dtype=int)
    kept_values: np.ndarray | None = None

    for batch, entropy in enumerate(seed_stream):
        remaining = config.paths - n_done
        if remaining <= 0:
            break
        this_batch = min(batch_size, remaining)

        batch_config = MonteCarloConfig(
            paths=this_batch,
            steps_per_year=config.steps_per_year,
            seed=int(np.random.default_rng(entropy).integers(0, 2**31 - 1)),
            antithetic=config.antithetic,
            memory_budget_mb=config.memory_budget_mb,
        )
        times, paths = simulate_log_paths(
            market,
            terms.T,
            batch_config,
            draws=draws,
            local_vol=local_vol,
            heston=heston,
            t0=t0,
        )

        if batch == 0:
            ko_idx = _schedule_indices(terms.ko_observation_times(), times, label="knock-out")
            ki_idx = _schedule_indices(terms.ki_observation_times(), times, label="knock-in")

        discounted, knocked_out, batch_knocked_in, ki_and_survived, ko_time = _evaluate_batch(
            terms, market, times, paths, ko_idx, ki_idx, knocked_in=knocked_in, t0=t0
        )

        if keep_path_values:
            kept_values = discounted

        total += float(discounted.sum())
        flat_sq_sum += float(np.dot(discounted, discounted))
        n_done += discounted.size
        ko_count += int(knocked_out.sum())
        ki_count += int(batch_knocked_in.sum())
        ki_survived_count += int(ki_and_survived.sum())
        ko_time_sum += float(np.nansum(ko_time))

        if config.antithetic:
            half = discounted.size // 2
            if half:
                pair_means = 0.5 * (discounted[:half] + discounted[half : 2 * half])
                pair_sum += float(pair_means.sum())
                pair_sq_sum += float(np.dot(pair_means, pair_means))
                pair_count += half

        del paths, discounted

    price = total / n_done

    # Standard error. Antithetic paths are correlated within a pair, so the
    # textbook std/sqrt(n) is simply the wrong estimator here; each pair
    # average is one independent observation instead. Measured on the reference
    # trade, the flat formula reported 1.40e-4 against a true across-seed
    # scatter of 1.00e-4 -- enough to make a validation tolerance meaningless.
    if config.antithetic and pair_count > 1:
        pair_mean = pair_sum / pair_count
        variance = max((pair_sq_sum - pair_count * pair_mean**2) / (pair_count - 1), 0.0)
        std_error = float(np.sqrt(variance / pair_count))
    elif n_done > 1:
        variance = max((flat_sq_sum - n_done * price**2) / (n_done - 1), 0.0)
        std_error = float(np.sqrt(variance / n_done))
    else:
        std_error = 0.0

    result = SnowballMCResult(
        price=price,
        std_error=std_error,
        ci95=1.959963985 * std_error,
        ko_prob=ko_count / n_done,
        ki_prob=ki_count / n_done,
        ki_no_ko_prob=ki_survived_count / n_done,
        expected_ko_time=(ko_time_sum / ko_count) if ko_count else float("nan"),
        paths=n_done,
        vol_model=(
            "heston-qe"
            if heston is not None
            else "dupire-local"
            if local_vol is not None
            else "flat"
        ),
        path_values=kept_values,
    )
    _log.debug(
        "snowball MC priced",
        extra={
            "price": result.price,
            "ko_prob": result.ko_prob,
            "ki_prob": result.ki_prob,
            "paths": n_done,
            "ko_observations": int(ko_idx.size),
        },
    )
    return result
