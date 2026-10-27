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

from otcpricer.core.logging import get_logger
from otcpricer.core.market import EquityMarket
from otcpricer.core.validation import check_positive
from otcpricer.products.snowball import SnowballTerms

__all__ = [
    "MonteCarloConfig",
    "SnowballMCResult",
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
    """

    price: float
    std_error: float
    ci95: float
    ko_prob: float
    ki_prob: float
    ki_no_ko_prob: float
    expected_ko_time: float
    paths: int

    def as_dict(self) -> dict[str, float | int]:
        """Flat dict for API responses."""
        return {
            "price": round(self.price, 8),
            "std_error": round(self.std_error, 8),
            "ci95": round(self.ci95, 8),
            "ko_prob": round(self.ko_prob, 6),
            "ki_prob": round(self.ki_prob, 6),
            "ki_no_ko_prob": round(self.ki_no_ko_prob, 6),
            "expected_ko_time": round(self.expected_ko_time, 6),
            "paths": self.paths,
        }


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
    draws = np.concatenate([base, -base], axis=1)
    if draws.shape[1] < n_paths:  # odd path count: top up with one extra draw
        extra = rng.standard_normal((n_steps, n_paths - draws.shape[1]))
        draws = np.concatenate([draws, extra], axis=1)
    return draws


def simulate_log_paths(
    market: EquityMarket,
    T: float,
    config: MonteCarloConfig,
    *,
    draws: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate geometric Brownian motion paths under the risk-neutral measure.

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

    Returns:
        ``(times, paths)`` where ``times`` has length ``steps + 1`` starting at
        0, and ``paths`` is ``(steps + 1, n_paths)`` with row 0 equal to spot.
    """
    check_positive(T, "T")
    n_steps = max(int(round(T * config.steps_per_year)), 1)
    dt = T / n_steps

    if draws is None:
        draws = normal_draws(n_steps, config.paths, config.seed, config.antithetic)
    elif draws.shape != (n_steps, config.paths):
        raise ValueError(f"draws must have shape {(n_steps, config.paths)}, got {draws.shape}")

    drift_step = (market.drift - 0.5 * market.sigma**2) * dt
    vol_step = market.sigma * np.sqrt(dt)

    # Cumulative sum rather than a Python loop over time steps: one vectorised
    # pass instead of n_steps interpreter round-trips.
    log_increments = drift_step + vol_step * draws
    log_paths = np.empty((n_steps + 1, draws.shape[1]), dtype=float)
    log_paths[0, :] = np.log(market.S)
    np.cumsum(log_increments, axis=0, out=log_paths[1:, :])
    log_paths[1:, :] += np.log(market.S)

    times = np.arange(n_steps + 1, dtype=float) * dt
    return times, np.exp(log_paths)


def _schedule_indices(observation_times: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Map observation dates onto the nearest simulation grid index.

    Returns unique indices in increasing order, excluding index 0 (inception is
    never an observation).
    """
    if observation_times.size == 0:
        return np.empty(0, dtype=int)
    idx = np.searchsorted(times, observation_times - 1e-12)
    idx = np.clip(idx, 1, times.size - 1)
    return np.unique(idx)


def _evaluate_batch(
    terms: SnowballTerms,
    market: EquityMarket,
    times: np.ndarray,
    paths: np.ndarray,
    ko_idx: np.ndarray,
    ki_idx: np.ndarray,
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
    ki_levels = paths[ki_idx, :] <= terms.ki_barrier
    cutoff = np.where(knocked_out, ko_time, np.inf)[None, :]
    knocked_in = np.asarray((ki_levels & (times[ki_idx][:, None] <= cutoff)).any(axis=0))

    # --- Assemble payoff and payment time per path --------------------------
    payoff = np.empty(n_paths, dtype=float)
    pay_time = np.full(n_paths, terms.T, dtype=float)

    payoff[knocked_out] = terms.knock_out_payoff(ko_time[knocked_out])
    pay_time[knocked_out] = ko_time[knocked_out]

    survived = np.asarray(~knocked_out)
    ki_and_survived = np.asarray(survived & knocked_in)
    clean = survived & ~knocked_in

    payoff[ki_and_survived] = terms.knock_in_payoff(paths[-1, ki_and_survived])
    payoff[clean] = terms.full_coupon_payoff()

    # --- Discount each path at its own payment date -------------------------
    # This is the correction: the legacy engine applied exp(-r * T) to every
    # path, including the majority that terminate early.
    discounted = np.asarray(payoff * np.exp(-market.r * pay_time))
    return discounted, knocked_out, knocked_in, ki_and_survived, ko_time


def price_snowball_mc(
    terms: SnowballTerms,
    market: EquityMarket,
    config: MonteCarloConfig | None = None,
    *,
    draws: np.ndarray | None = None,
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

    Returns:
        A :class:`SnowballMCResult` with the price, its Monte Carlo error, and
        the knock-out/knock-in statistics a desk quotes alongside it.
    """
    config = config or MonteCarloConfig()
    n_steps = max(int(round(terms.T * config.steps_per_year)), 1)

    batch_size = config.paths if draws is not None else config.batch_paths(n_steps)
    n_batches = max(int(np.ceil(config.paths / batch_size)), 1)

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
        times, paths = simulate_log_paths(market, terms.T, batch_config, draws=draws)

        if batch == 0:
            ko_idx = _schedule_indices(terms.ko_observation_times(), times)
            ki_idx = _schedule_indices(terms.ki_observation_times(), times)

        discounted, knocked_out, knocked_in, ki_and_survived, ko_time = _evaluate_batch(
            terms, market, times, paths, ko_idx, ki_idx
        )

        total += float(discounted.sum())
        flat_sq_sum += float(np.dot(discounted, discounted))
        n_done += discounted.size
        ko_count += int(knocked_out.sum())
        ki_count += int(knocked_in.sum())
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
