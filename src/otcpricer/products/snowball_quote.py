"""Quoting a snowball: solving for the coupon, not pricing a given one.

A desk does not ask "what is this note worth at a 20% coupon". It asks "what
coupon makes this note worth par" -- that number *is* the quote. Until now
nothing in this project could answer it: ``coupon_pa`` was an input everywhere,
and the only par-solve in the codebase was :mod:`otcpricer.products.irs` for
swaps.

The price is exactly affine in the coupon
-----------------------------------------
This is the whole design, so it is worth stating precisely. The knock-out and
knock-in events depend only on the spot path against fixed barriers -- **never
on the coupon**. And every coupon-bearing term in both engines is linear in it:

- ``knock_out_payoff`` is ``coupon * t`` and ``full_coupon_payoff`` is
  ``coupon * T`` (:mod:`otcpricer.products.snowball`);
- the knock-in leg ``-max(0, 1 - S/K)`` does not involve the coupon at all;
- in the PDE, the terminal conditions, both upper boundaries and the knock-out
  overwrite are each ``coupon`` times something coupon-free, and the operator
  and the knock-in transfer are linear.

So ``price(c) = A*c + B`` holds **exactly**, with

.. code-block:: text

    A = E[ discount * (t_ko on knock-out paths + T on clean paths) ]
    B = E[ discount * knock-in loss ]                    (negative)

and the par coupon is ``-B / A``. Two pricer calls would give it exactly:
``c = 0`` returns ``B``, ``c = 1`` returns ``A + B``.

``A`` is the snowball's **coupon annuity** -- the discounted expected time the
note spends alive and paying -- and that is the same object
:mod:`otcpricer.products.irs` already uses to get a par swap rate analytically
rather than by bisection. The symmetry is exact, down to the comment there
about the annuity being the whole sensitivity rather than one discount factor.

A third solve, not two
----------------------
This takes **three** solves, at ``c = 0``, ``1`` and ``0.5``. The extra 50% buys
something worth far more than it costs: ``|0.5*A + B - price(0.5)|`` is computed
every time and *raises* if it exceeds :data:`MAX_AFFINITY_RESIDUAL`. So the
property the module is built on is an enforced invariant rather than a
documented assumption, and a future change to ``pde.py`` that introduced a
coupon-dependent grid or boundary would fail here immediately instead of
silently returning a wrong quote.

Measured on the reference trade at a 1600x1008 grid, the residual against that
straight line across six coupons from 0 to 1 is **7.8e-16** -- machine
precision -- and the price at the solved coupon is **8.5e-17**. It is equally
exact on the knocked-in surface (1.3e-16) and with the knock-in put struck at
the barrier (8.3e-17). The Monte Carlo's residual is **identically zero**,
because a fixed seed gives bit-identical paths across coupons.

The answer carries the grid's own bias
--------------------------------------
Exact in the coupon does not mean exact in the price. The par coupon inherits
the finite-difference grid's first-order convergence:

===============  ============
grid             par coupon
===============  ============
400 x 252            7.2902%
800 x 504            7.1441%
1600 x 1008          7.1187%
3200 x 2016          7.1069%
6400 x 4032          7.1009%
===============  ============

Differences of -1.46e-3, -2.50e-4, -1.20e-4, -6.1e-5 -- halving, so the
Richardson limit is about **7.0948%**. At 1600x1008 the quote is therefore
**2.4e-04 high**, which is 1.4 Monte Carlo standard errors at 400,000 paths,
i.e. visible rather than negligible. :data:`PAR_COUPON_PDE_CONFIG` is the grid
at which that bias drops below the simulation's noise floor.

Why that matters beyond elegance
--------------------------------
A reference implementation (``carrey325/Autocall-pricing-and-risk-engine``)
solves this by bisection over ``[0.01, 0.40]``. On this product that is roughly
thirty wasted solves for an answer two give exactly -- and worse, an iterative
solver converges to a *tolerance*, so it can never reveal that the underlying
relationship is linear. Here the affinity is itself the test: three coupons
must be collinear to machine precision, which is a far stronger statement than
any tolerance on a root.

The basis is a first-order driver of the answer
-----------------------------------------------
Measured on the reference trade at 20% volatility:

=========================  ============
futures basis              par coupon
=========================  ============
0 (the old default)              7.12%
IO, measured ~0.6%               7.49%
MO, measured ~7.7%           **13.40%**
=========================  ============

So the basis nearly doubles the quotable coupon, which is exactly what the
market view -- "the deeper the futures discount, the higher the coupon" --
asserts, now measured rather than repeated. :func:`decompose_par_coupon`
splits the quote into its volatility and basis parts for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from otcpricer.core.exceptions import CalibrationError
from otcpricer.core.logging import get_logger
from otcpricer.core.market import EquityMarket
from otcpricer.core.vol import LocalVolatility
from otcpricer.engines.pde import PDEConfig, price_snowball_pde
from otcpricer.products.snowball import SnowballTerms

__all__ = [
    "MAX_AFFINITY_RESIDUAL",
    "MIN_COUPON_ANNUITY",
    "PAR_COUPON_PDE_CONFIG",
    "CouponDecomposition",
    "CouponSolution",
    "decompose_par_coupon",
    "solve_par_coupon",
    "verify_par_coupon_mc",
]

_log = get_logger(__name__)

#: Smallest coupon-leg value worth inverting.
#:
#: ``A`` is the present value of one unit of coupon, so it is the probability-
#: weighted, discounted time the note spends alive and paying. It vanishes only
#: when the note can neither knock out nor survive clean -- an unreachable
#: knock-out barrier together with a certain knock-in. Then no coupon whatsoever
#: makes the note worth par, and dividing by it would hand back a confident
#: ``inf`` rather than saying so.
MIN_COUPON_ANNUITY = 1e-12

#: Largest acceptable departure from exact affinity in the coupon.
#:
#: Measured at 1.1e-16 to 7.8e-16 across coupons, parameter sets and both the
#: clean and knocked-in surfaces, so 1e-12 leaves four decades of headroom for
#: a platform difference while still catching any real break. Exceeding it
#: means a coupon-dependent term has appeared somewhere in the engine, and the
#: whole basis of this module is gone -- so it raises rather than warns.
MAX_AFFINITY_RESIDUAL = 1e-12

#: Grid at which the par coupon's own discretisation bias falls below the
#: Monte Carlo noise floor.
#:
#: The par coupon converges first-order like the price it comes from: 7.2902%,
#: 7.1441%, 7.1187%, 7.1069%, 7.1009% as the grid doubles from 400x252, with a
#: Richardson limit near 7.0948%. At 1600x1008 the bias is 2.4e-04 -- 1.4
#: standard errors at 400,000 paths -- so a cross-engine test run there is
#: testing the grid rather than the agreement.
PAR_COUPON_PDE_CONFIG = PDEConfig(space_nodes=3200, steps_per_year=2016)


@dataclass(frozen=True, slots=True)
class CouponSolution:
    """A solved coupon and the two legs it came from.

    Attributes:
        par_coupon: The coupon making the note worth ``target``.
        coupon_annuity: ``A``, the present value of one unit of coupon a year.
            The snowball's analogue of a swap annuity, and readable as the
            discounted expected time the note stays alive and paying.
        knock_in_value: ``B``, the present value of the knock-in leg. Negative
            for any note whose downside is reachable.
        target: The price the coupon was solved to.
        verified_price: The note repriced at ``par_coupon``. Should be
            ``target`` to machine precision; carried so a caller can see that
            rather than trust it.
        affinity_residual: How far the midpoint solve fell from the line
            through the other two -- the measured departure from the property
            this whole module rests on, reported rather than assumed.
        method: How it was obtained.
    """

    par_coupon: float
    coupon_annuity: float
    knock_in_value: float
    target: float
    verified_price: float
    affinity_residual: float = 0.0
    method: str = "affine-exact"

    @property
    def residual(self) -> float:
        """How far the repriced note sits from the target."""
        return self.verified_price - self.target

    @property
    def knock_in_cost_in_coupon(self) -> float:
        """The knock-in leg expressed as coupon, which is what it costs to quote.

        ``-B/A`` is the par coupon when the target is zero, so this *is* the
        quote on a note struck at par -- read as "the coupon the downside
        obliges you to pay".
        """
        return -self.knock_in_value / self.coupon_annuity

    def price_at(self, coupon_pa: float) -> float:
        """The note's price at any coupon, without solving again.

        ``A*c + B``. Because the relationship is exact rather than
        approximate, the two legs are a *complete* description of the note
        across every coupon -- so a caller walking a quote ladder pays for
        three solves once instead of one solve per rung.
        """
        return self.coupon_annuity * coupon_pa + self.knock_in_value

    def as_dict(self) -> dict[str, object]:
        return {
            "par_coupon": round(self.par_coupon, 10),
            "par_coupon_pct": round(self.par_coupon * 100.0, 6),
            "coupon_annuity": round(self.coupon_annuity, 10),
            "knock_in_value": round(self.knock_in_value, 10),
            "target": self.target,
            "verified_price": round(self.verified_price, 12),
            "residual": round(self.residual, 14),
            "affinity_residual": float(f"{self.affinity_residual:.3e}"),
            "method": self.method,
            "note": (
                "The price is exactly affine in the coupon, so this is solved "
                "from three pricer calls rather than by root-finding, and the "
                "residual is at machine precision rather than at a tolerance. "
                "The third call is the one that checks the affinity instead of "
                "assuming it; affinity_residual is what it measured."
            ),
        }


def solve_par_coupon(
    terms: SnowballTerms,
    market: EquityMarket,
    config: PDEConfig | None = None,
    *,
    target: float = 0.0,
    knocked_in: bool = False,
    local_vol: LocalVolatility | None = None,
) -> CouponSolution:
    """Solve for the coupon that makes a snowball worth ``target``.

    Exact, from three finite-difference solves, because the price is affine
    in the coupon -- see the module docstring. ``terms.coupon_pa`` is ignored:
    it is the unknown here.

    Two solves are arithmetically sufficient. The third, at ``c = 0.5``, is
    what turns the affinity from a documented assumption into an invariant
    checked on every call.

    Args:
        terms: Contract terms. Its ``coupon_pa`` is overwritten internally.
        market: Market state. ``basis_pa`` matters a great deal -- it roughly
            doubles the answer at the measured MO level.
        config: Grid controls, passed through unchanged.
        target: The price to solve to. Zero is a note struck at par.
        knocked_in: Solve for a note that has already knocked in.
        local_vol: A Dupire local volatility to solve against.

    Returns:
        A :class:`CouponSolution`.

    Raises:
        CalibrationError: if the coupon annuity is worthless, so no coupon
            reaches the target; or if the price is no longer affine in the
            coupon, which would make this route silently wrong rather than
            merely imprecise.
    """

    def price_at_coupon(coupon: float) -> float:
        # The coupon is overridden from outside the engine, deliberately.
        # Giving price_snowball_pde a coupon argument would mean touching the
        # function whose output tests/test_snowball_local_vol.py pins
        # bit-exactly; this way nothing here can reach it.
        return price_snowball_pde(
            replace(terms, coupon_pa=coupon),
            market,
            config,
            knocked_in=knocked_in,
            local_vol=local_vol,
        ).price

    at_zero = price_at_coupon(0.0)
    at_one = price_at_coupon(1.0)
    at_half = price_at_coupon(0.5)

    knock_in_value = at_zero
    coupon_annuity = at_one - at_zero

    # The invariant, measured rather than assumed. Were a coupon-dependent
    # grid, boundary or schedule ever to appear in the engine, -B/A would stop
    # being the par coupon -- and would be returned with full confidence and
    # no symptom at all. This is the cheapest available guard against that,
    # and it is the entire reason for the third solve.
    predicted = 0.5 * coupon_annuity + knock_in_value
    affinity_residual = abs(predicted - at_half)
    if affinity_residual > MAX_AFFINITY_RESIDUAL:
        raise CalibrationError(
            f"the snowball price is no longer affine in the coupon: pricing at "
            f"c=0.5 gave {at_half:.12g}, while the line through c=0 and c=1 "
            f"predicts {predicted:.12g} -- a gap of {affinity_residual:.3e} "
            f"against a limit of {MAX_AFFINITY_RESIDUAL:.0e}. Solving by this "
            f"route assumes exact affinity, so the answer would be wrong "
            f"rather than imprecise. Something in the pricing path has become "
            f"coupon-dependent."
        )

    if coupon_annuity <= MIN_COUPON_ANNUITY:
        raise CalibrationError(
            f"the coupon annuity is {coupon_annuity:.3e} per unit of coupon, "
            f"so no coupon makes this note worth {target}. That happens when the "
            f"note can neither knock out nor survive clean -- check whether "
            f"ko_pct={terms.ko_pct} is reachable and ki_pct={terms.ki_pct} is "
            f"avoidable at sigma={market.sigma} over T={terms.T}"
        )

    par_coupon = (target - knock_in_value) / coupon_annuity
    verified = price_at_coupon(par_coupon)

    _log.info(
        "par coupon %.6f%% (annuity %.6f, knock-in leg %.6f, residual %.2e, affinity %.2e)",
        par_coupon * 100.0,
        coupon_annuity,
        knock_in_value,
        verified - target,
        affinity_residual,
    )

    return CouponSolution(
        par_coupon=par_coupon,
        coupon_annuity=coupon_annuity,
        knock_in_value=knock_in_value,
        target=target,
        verified_price=verified,
        affinity_residual=affinity_residual,
        method="affine-exact" if local_vol is None else "affine-exact+dupire-local",
    )


def verify_par_coupon_mc(
    solution: CouponSolution,
    terms: SnowballTerms,
    market: EquityMarket,
    mc_config: object | None = None,
    *,
    knocked_in: bool = False,
    local_vol: LocalVolatility | None = None,
) -> dict[str, object]:
    """Reprice at the solved coupon on the Monte Carlo, as an independent check.

    The solver is exact *given the finite-difference price*, so what it cannot
    establish is that the finite-difference price is right. This does: a
    different engine, sharing nothing with the first but the contract, must
    also return ``target`` at the same coupon.

    Measured on the reference trade at 400,000 paths, the Monte Carlo returns
    ``+8.8e-05 +/- 1.7e-04``, i.e. **0.54 standard errors** from zero.

    Returns:
        A dict with the simulated price, its standard error, the z-score
        against the target, and a pass/fail at three standard errors.
    """
    from otcpricer.engines.mc import MonteCarloConfig, price_snowball_mc

    resolved = mc_config if isinstance(mc_config, MonteCarloConfig) else MonteCarloConfig()
    simulated = price_snowball_mc(
        replace(terms, coupon_pa=solution.par_coupon),
        market,
        resolved,
        local_vol=local_vol,
        knocked_in=knocked_in,
    )

    z = (
        (simulated.price - solution.target) / simulated.std_error
        if simulated.std_error > 0.0
        else float("nan")
    )
    return {
        "mc_price": round(simulated.price, 10),
        "mc_std_error": round(simulated.std_error, 10),
        "target": solution.target,
        "z_score": round(z, 4),
        "passes": bool(abs(z) < 3.0),
        "paths": simulated.paths,
        "vol_model": simulated.vol_model,
        "note": (
            "An independent engine repricing at the solved coupon. The solver "
            "is exact given the PDE price; this is what establishes that the "
            "PDE price itself is right."
        ),
    }


@dataclass(frozen=True, slots=True)
class CouponDecomposition:
    """A quotable coupon split into what pays for it.

    Attributes:
        par_coupon: The full quote, at the market's actual basis.
        without_basis: The quote the same note would carry at a zero basis.
        basis_pa: The basis used.
        volatility_part: ``without_basis`` -- what the volatility alone buys.
        basis_part: The remainder, attributable to the futures discount.
    """

    par_coupon: float
    without_basis: float
    basis_pa: float

    @property
    def volatility_part(self) -> float:
        return self.without_basis

    @property
    def basis_part(self) -> float:
        return self.par_coupon - self.without_basis

    @property
    def basis_share(self) -> float:
        """Fraction of the quote the basis is responsible for."""
        return self.basis_part / self.par_coupon if self.par_coupon else float("nan")

    def as_dict(self) -> dict[str, object]:
        return {
            "par_coupon": round(self.par_coupon, 10),
            "par_coupon_pct": round(self.par_coupon * 100.0, 6),
            "volatility_part_pct": round(self.volatility_part * 100.0, 6),
            "basis_part_pct": round(self.basis_part * 100.0, 6),
            "basis_share": round(self.basis_share, 6),
            "basis_pa": round(self.basis_pa, 8),
            "note": (
                "The split is the difference between solving at the market's "
                "basis and at a zero basis. It is not additive in any deeper "
                "sense -- the two legs interact through the drift -- but it is "
                "the comparison a desk actually makes when the futures curve "
                "moves."
            ),
        }


def decompose_par_coupon(
    terms: SnowballTerms,
    market: EquityMarket,
    config: PDEConfig | None = None,
    *,
    knocked_in: bool = False,
    local_vol: LocalVolatility | None = None,
) -> CouponDecomposition:
    """Split the quotable coupon into its volatility and basis parts.

    Four finite-difference solves: two par-coupon solves, each costing two.

    The market view is that a snowball coupon is paid for by volatility, the
    futures discount and cash management, and that "the deeper the discount,
    the higher the coupon". The first two of those are now measurable here --
    and the effect is large rather than marginal. At the reference trade's 20%
    volatility the quote runs 7.12% at a zero basis and 13.40% at the measured
    MO basis, so the discount accounts for **47%** of the coupon.
    """
    full = solve_par_coupon(terms, market, config, knocked_in=knocked_in, local_vol=local_vol)
    flat_basis = solve_par_coupon(
        terms,
        replace(market, basis_pa=0.0),
        config,
        knocked_in=knocked_in,
        local_vol=local_vol,
    )
    return CouponDecomposition(
        par_coupon=full.par_coupon,
        without_basis=flat_basis.par_coupon,
        basis_pa=market.basis_pa,
    )
