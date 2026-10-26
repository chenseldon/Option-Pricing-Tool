"""Snowball (autocallable) contract terms and observation schedule.

Separating the *contract* from the *engine* is what makes the PDE and the Monte
Carlo comparable. In the legacy code each engine carried its own idea of the
payoff, and they silently disagreed: the Monte Carlo struck the knock-in put at
the knock-in barrier while the PDE struck it at the initial price. Those are
different products, so the "independent PDE validation" the project advertised
was comparing two unrelated numbers (measured: 0.046047 versus 0.037848, 18%
apart). One shared terms object removes the possibility.

Contract conventions, as traded in the A-share market
----------------------------------------------------
* **Knock-out** is observed *monthly*. On a knock-out the note terminates early
  and pays the coupon accrued to that date, ``coupon_pa * t``.
* **Knock-in** is observed *daily* -- a far finer schedule than knock-out,
  which is why the two frequencies are separate parameters.
* A **lock-up** period (锁定期), typically the first three months, suspends
  knock-out observation. The note cannot terminate during it, which lowers the
  early-termination probability and is worth real money. The legacy model had
  no such parameter and so began observing from month one.
* If the note knocks in and never knocks out, the holder absorbs the fall as a
  **short put struck at the initial price** ``S0``. Some variants strike that
  put at the knock-in barrier instead, capping the loss, hence
  :class:`~otcpricer.core.types.KIStrikeMode`.
* If neither barrier is touched, the holder receives the full coupon
  ``coupon_pa * T``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from otcpricer.core.types import KIStrikeMode
from otcpricer.core.validation import check_positive, validate_snowball

__all__ = ["SnowballTerms"]


@dataclass(frozen=True, slots=True)
class SnowballTerms:
    """Terms of a snowball autocallable note.

    Prices are expressed as a fraction of notional, so a result of ``0.04``
    means 4% of notional.

    Attributes:
        S0: Initial fixing level of the underlying.
        T: Tenor in years.
        ko_pct: Knock-out level as a fraction of ``S0`` (``1.03`` is 103%).
        ki_pct: Knock-in level as a fraction of ``S0`` (``0.75`` is 75%).
        coupon_pa: Annualised coupon rate (``0.20`` is 20% p.a.).
        obs_freq: Knock-out observations per year; 12 is monthly.
        ki_obs_freq: Knock-in observations per year; 252 is each trading day.
        lockup_months: Months at the start during which knock-out is not
            observed. Defaults to 3, the standard A-share structure.
        ki_strike_mode: Strike of the knock-in short put. Defaults to ``S0``,
            the market convention.
    """

    S0: float
    T: float
    ko_pct: float = 1.03
    ki_pct: float = 0.75
    coupon_pa: float = 0.20
    obs_freq: int = 12
    ki_obs_freq: int = 252
    lockup_months: int = 3
    ki_strike_mode: KIStrikeMode = KIStrikeMode.S0

    def __post_init__(self) -> None:
        check_positive(self.S0, "S0")
        check_positive(self.T, "T")
        # Market params are the engine's concern; validate only the structure
        # here, using placeholder market values so the shared validator can be
        # reused without duplicating the structural rules.
        validate_snowball(
            S0=self.S0,
            T=self.T,
            r=0.0,
            sigma=0.2,
            q=0.0,
            ko_pct=self.ko_pct,
            ki_pct=self.ki_pct,
            coupon_pa=self.coupon_pa,
            obs_freq=self.obs_freq,
            ki_obs_freq=self.ki_obs_freq,
            lockup_months=self.lockup_months,
        )

    # -- Barrier levels ----------------------------------------------------

    @property
    def ko_barrier(self) -> float:
        """Absolute knock-out level."""
        return self.S0 * self.ko_pct

    @property
    def ki_barrier(self) -> float:
        """Absolute knock-in level."""
        return self.S0 * self.ki_pct

    @property
    def ki_put_strike(self) -> float:
        """Strike of the knock-in short put, per :attr:`ki_strike_mode`."""
        if self.ki_strike_mode is KIStrikeMode.S0:
            return self.S0
        return self.ki_barrier

    @property
    def lockup_years(self) -> float:
        """Lock-up length in years."""
        return self.lockup_months / 12.0

    # -- Observation schedules --------------------------------------------

    def ko_observation_times(self) -> np.ndarray:
        """Knock-out observation dates, in years from inception.

        Observations fall on exact ``1 / obs_freq`` boundaries, and any date at
        or before the end of the lock-up is dropped. With the default monthly
        frequency and a three-month lock-up on a one-year note this yields
        months 4 through 12, i.e. nine observations rather than twelve.

        Returns:
            Strictly increasing array of observation times; empty only if the
            lock-up consumes the whole tenor, which the validator forbids.
        """
        count = int(round(self.T * self.obs_freq))
        raw = np.arange(1, count + 1, dtype=float) / self.obs_freq
        # Guard against float error placing the final observation just past T.
        raw = raw[raw <= self.T + 1e-9]
        return raw[raw > self.lockup_years + 1e-9]

    def ki_observation_times(self) -> np.ndarray:
        """Knock-in observation dates, in years from inception.

        Knock-in is monitored for the whole life of the note, lock-up included:
        the lock-up suspends early *termination*, not downside monitoring.
        """
        count = int(round(self.T * self.ki_obs_freq))
        raw = np.arange(1, count + 1, dtype=float) / self.ki_obs_freq
        return raw[raw <= self.T + 1e-9]

    # -- Payoff legs -------------------------------------------------------

    def knock_out_payoff(self, t_observed: np.ndarray | float) -> np.ndarray | float:
        """Coupon accrued to the knock-out date, as a fraction of notional.

        Paid *at* ``t_observed``, not at maturity -- which is the whole reason
        the engines must carry payment times alongside payoffs.
        """
        return self.coupon_pa * t_observed

    def full_coupon_payoff(self) -> float:
        """Coupon for a note that neither knocks out nor knocks in."""
        return self.coupon_pa * self.T

    def knock_in_payoff(self, S_final: np.ndarray | float) -> np.ndarray | float:
        """Loss on the knock-in short put, as a (negative) fraction of notional.

        ``-max(0, 1 - S_T / strike)``, so a terminal level at or above the
        strike pays nothing and the loss grows linearly below it.
        """
        strike = self.ki_put_strike
        return -np.maximum(0.0, 1.0 - np.asarray(S_final, dtype=float) / strike)

    def describe(self) -> dict[str, object]:
        """Human-readable summary, for API responses and validation reports."""
        return {
            "S0": self.S0,
            "T": self.T,
            "ko_barrier": self.ko_barrier,
            "ki_barrier": self.ki_barrier,
            "ko_pct": self.ko_pct,
            "ki_pct": self.ki_pct,
            "coupon_pa": self.coupon_pa,
            "obs_freq": self.obs_freq,
            "ki_obs_freq": self.ki_obs_freq,
            "lockup_months": self.lockup_months,
            "ki_strike_mode": str(self.ki_strike_mode),
            "ki_put_strike": self.ki_put_strike,
            "n_ko_observations": int(self.ko_observation_times().size),
        }
