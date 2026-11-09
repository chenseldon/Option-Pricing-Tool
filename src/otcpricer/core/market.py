"""Market state container.

The legacy code threaded ``(S, r, sigma, q)`` through every signature
positionally, which made adding a fifth input mean touching every call site.
Bundling them also makes room for the one input the old model was missing
entirely: the index futures basis.

Why the basis matters here
--------------------------
A CNY desk hedging a snowball or an autocallable does not trade the index --
it trades the index future, which in the A-share market persistently trades at
a *discount* to spot (贴水). That discount is a first-order driver of the
coupon the desk can quote, alongside volatility, and the legacy model had no
way to express it.

Mechanically, a futures discount raises the carry the underlying is assumed to
pay. If the future trades below spot then ``F = S e^{(r - q_eff) T} < S``, so
``q_eff > r``: the basis behaves as an addition to the dividend yield in the
*drift*, while discounting still happens at ``r``. Those two roles are
separate, which is exactly why a single ``q`` could not represent it.

The direction is worth stating because it is easy to get backwards. A larger
discount lowers the drift, which makes the note worth less to the client, which
lets the desk quote a *higher* coupon for the same price -- matching the market
observation that a deeper 贴水 supports a richer snowball coupon.

The sign was over-constrained
-----------------------------
``basis_pa`` was required to be non-negative, which reads the market view --
"the A-share index futures persistently trade at a discount" -- as a law rather
than as a tendency. Measured on the CFFEX panel in
:mod:`otcpricer.marketdata.carry`, IH futures trade *above* their index on
40.9% of days and the resulting carry is negative on 10.3% of them. So the
check now bounds the magnitude and leaves the sign to the data.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from otcpricer.core.validation import (
    MAX_ABS_CARRY,
    check_in_range,
    check_positive,
    validate_market_params,
)

__all__ = ["EquityMarket"]


@dataclass(frozen=True, slots=True)
class EquityMarket:
    """Observable market state for an equity underlying.

    Attributes:
        S: Spot price of the underlying.
        r: Continuously compounded risk-free rate, used for discounting.
        sigma: Annualised volatility.
        q: Continuous dividend yield.
        basis_pa: Annualised index futures basis, as a *positive* decimal for
            a future trading **below** spot (贴水) and negative for a premium.
            Enters the drift only, never the discount factor. Defaults to 0 so
            existing behaviour is unchanged unless a caller opts in.

            Negative values are admissible, and were not before: see
            :data:`~otcpricer.core.validation.MAX_ABS_CARRY`. Only the
            magnitude is bounded, to catch a basis handed over in percent.
    """

    S: float
    r: float
    sigma: float
    q: float = 0.0
    basis_pa: float = 0.0

    def __post_init__(self) -> None:
        validate_market_params(self.S, 1.0, self.r, self.sigma, self.q)
        # Magnitude only. The sign was constrained here until the futures panel
        # was measured; a premium is a real market state on two IH days in five.
        check_in_range(
            abs(self.basis_pa),
            "|basis_pa|",
            0.0,
            MAX_ABS_CARRY,
            include_low=True,
        )

    @property
    def carry(self) -> float:
        """Total yield deducted from the drift: ``q + basis_pa``.

        This is the quantity that enters the risk-neutral drift as
        ``r - carry``. Discounting deliberately does not use it.
        """
        return self.q + self.basis_pa

    @property
    def drift(self) -> float:
        """Risk-neutral drift of the underlying, ``r - q - basis_pa``."""
        return self.r - self.carry

    def forward(self, T: float) -> float:
        """Forward price to ``T``, consistent with the hedging instrument.

        With a non-zero basis this is the *futures*-implied forward, which is
        the price the desk can actually transact at, not the theoretical
        dividend-only forward.
        """
        import numpy as np

        check_positive(T, "T")
        return float(self.S * np.exp(self.drift * T))

    def with_shift(
        self,
        *,
        spot_rel: float = 0.0,
        vol_abs: float = 0.0,
        rate_abs: float = 0.0,
        carry_abs: float = 0.0,
        div_abs: float = 0.0,
    ) -> EquityMarket:
        """Return a shifted copy, for finite-difference risk and stress grids.

        ``carry_abs`` and ``div_abs`` move ``basis_pa`` and ``q`` respectively,
        and they reach the price through :attr:`carry` -- the same sum. They
        are therefore the *same* shift, and both exist only so a test can apply
        them separately and show that they are. One Greek is reported, not two;
        see :func:`~otcpricer.engines.risk.snowball_risk_pde`.

        Args:
            spot_rel: Relative spot shift (``0.01`` is +1%).
            vol_abs: Absolute volatility shift (``0.01`` is +1 vol point).
            rate_abs: Absolute rate shift (``0.0001`` is +1bp). Moves the
                discounting as well as the drift.
            carry_abs: Absolute shift to ``basis_pa``. Drift only.
            div_abs: Absolute shift to ``q``. Drift only, and indistinguishable
                from ``carry_abs`` in every price.
        """
        return replace(
            self,
            S=self.S * (1.0 + spot_rel),
            sigma=max(self.sigma + vol_abs, 1e-8),
            r=self.r + rate_abs,
            basis_pa=self.basis_pa + carry_abs,
            q=max(self.q + div_abs, 0.0),
        )

    def as_dict(self) -> dict[str, float | str]:
        """Flat dict for API responses and reports.

        Reports the carry and the drift as well as the three inputs that make
        them, because the sum is what both engines actually read: ``q`` and
        ``basis_pa`` reach a price only through :attr:`carry`, so a response
        listing them separately without their sum invites a reader to treat
        them as two exposures.
        """
        return {
            "S": self.S,
            "r": self.r,
            "sigma": self.sigma,
            "q": self.q,
            "basis_pa": self.basis_pa,
            "carry": self.carry,
            "drift": self.drift,
            "carry_note": (
                "carry = q + basis_pa is the single quantity the engines "
                "read; dV/dq and dV/dbasis_pa are the same derivative."
            ),
        }
