"""Quote report for a snowball note: what it pays, and what that rests on.

The flagship product was the last one without a report, which is backwards: the
surface, the local volatility and Heston all had one while the thing actually
quoted did not.

Ordering, and why it is this one
-------------------------------
Same principle as the other reports in this package -- lead with what could
make the number wrong, not with the number. For a snowball the ranking is
measured rather than assumed:

1. **Carry coverage.** The listed futures reach about half a year while the
   notes run one to three, so 48% to 84% of the carry is an assumption. And
   carry is not a refinement: at the CSI 1000's measured level it more than
   doubles the quotable coupon, 6.54% against 15.04%. The largest input is
   also the least observed, so it goes first.
2. **The quote** and its split into volatility and carry.
3. **Price, probabilities and expected knock-out time** -- what the note does.
4. **Risk**, each Greek labelled with its method and whether a second engine
   confirms it.
5. **Two-engine reconciliation**, including the two surfaces that only
   acquired a second engine in this phase: the knocked-in state and theta.

What the report refuses to do
-----------------------------
It does not print a number without its provenance. Every Greek carries the
method that produced it, theta carries the fact that its two estimates are
difference quotients over horizons four times apart, and the carry section
prints the measured fraction before the level. A reader who stops after the
first screen should come away knowing the quote's weakest input, not its
headline.
"""

from __future__ import annotations

from otcpricer.core.market import EquityMarket
from otcpricer.engines.mc import SnowballMCResult
from otcpricer.engines.risk import RiskComparison, SnowballRisk
from otcpricer.marketdata.carry import CarryMeasurement
from otcpricer.products.snowball import SnowballTerms
from otcpricer.products.snowball_quote import CouponDecomposition, CouponSolution
from otcpricer.reports.layout import rule, wrap

__all__ = ["snowball_report"]


def _header(terms: SnowballTerms, market: EquityMarket) -> list[str]:
    return [
        rule("="),
        "SNOWBALL QUOTE REPORT",
        rule("="),
        f"  spot / fixing  {market.S:.6g} / {terms.S0:.6g}",
        f"  tenor          {terms.T:.4f} years",
        f"  knock-out      {terms.ko_pct:.4%} of fixing   ({terms.obs_freq} observations a year)",
        f"  knock-in       {terms.ki_pct:.4%} of fixing"
        f"   ({terms.ki_obs_freq} observations a year)",
        f"  lock-up        {terms.lockup_months} months",
        f"  rate           {market.r:.4%}",
        f"  volatility     {market.sigma:.4%}",
        f"  carry          {market.carry:.4%}   (q {market.q:.4%} + basis {market.basis_pa:.4%})",
        "",
    ]


def _carry_section(
    carry: CarryMeasurement | None,
    market: EquityMarket,
    terms: SnowballTerms,
) -> list[str]:
    lines = ["CARRY COVERAGE -- READ THIS FIRST", rule()]

    if carry is None:
        lines += [
            f"  carry in use         {market.carry:.4%} a year",
            "  measured from        nothing -- supplied by the caller",
            "",
        ]
        lines += wrap(
            "No futures measurement was supplied, so the carry above is an "
            "assumption over the whole life of the note rather than over part "
            "of it. That is not a small thing: at the carry measured on the "
            "CSI 1000 board the quotable coupon more than doubles, 6.54% "
            "against 15.04%. Pass a CarryMeasurement to see which part of this "
            "number the futures actually support."
        )
        lines.append("")
        return lines

    measured = carry.measured_fraction(terms.T)
    lines += [
        f"  root / index         {carry.root} / {carry.index_code}",
        f"  as of                {carry.as_of.isoformat()}",
        f"  contracts used       {len(carry.maturities)}"
        f"   ({carry.rejected_near} rejected as too near)",
        f"  board reaches        {carry.observed_reach:.4f} years",
        f"  note runs            {terms.T:.4f} years",
        f"  MEASURED FRACTION    {measured:.1%} of the note's life",
        f"  tail rule            {carry.tail}  ->  {carry.tail_carry:.4%} a year",
        "",
        f"  near carry           {carry.near_carry:.4%}   (to {carry.maturities[0]:.4f}y)",
        f"  far carry            {carry.far_carry:.4%}   (to {carry.observed_reach:.4f}y)",
        f"  term structure gap   {carry.term_structure_gap:+.4%}",
        f"  average over tenor   {carry.average_carry_to(terms.T):.4%}",
        "",
    ]

    lines += wrap(
        f"{1.0 - measured:.0%} of this note's carry is not measured. The listed "
        f"futures stop at {carry.observed_reach:.2f} years; beyond that the "
        f"carry is whatever {carry.tail} says, here {carry.tail_carry:.2%} a "
        f"year. This is the single largest assumption in the quote -- larger "
        f"than the volatility on a three-year note -- because the carry moves "
        f"the drift and the drift is what decides whether the note knocks out "
        f"or knocks in."
    )
    lines.append("")
    lines += wrap(
        "The board observes a drift, not a carry: log(F/S)/tau = r - q - basis "
        "is one number against three unknowns. So this is total carry, fed as "
        f"q=0 and basis_pa=carry at r={carry.rate:.2%}. Nothing here separates "
        "the dividend yield from the financing spread, and the engines never "
        "needed the split -- they read only the sum."
    )
    lines.append("")

    if abs(carry.term_structure_gap) > 0.02:
        lines += wrap(
            f"The measured term structure is not flat: {carry.term_structure_gap:+.2%} "
            f"between the near and far contracts. A single scalar carry is "
            f"therefore an average rather than a level, and the average above "
            f"is the one weighted by how long each part applies."
        )
        lines.append("")
    return lines


def _quote_section(
    solution: CouponSolution | None,
    decomposition: CouponDecomposition | None,
) -> list[str]:
    if solution is None:
        return []
    lines = [
        "THE QUOTE",
        rule(),
        f"  par coupon           {solution.par_coupon:.4%} a year",
        f"  coupon annuity       {solution.coupon_annuity:.6f}   (discounted expected time alive)",
        f"  knock-in leg         {solution.knock_in_value:.6f}",
        f"  solved to            {solution.target:.6g}",
        f"  residual             {solution.residual:+.3e}",
        f"  affinity residual    {solution.affinity_residual:.3e}",
        f"  method               {solution.method}",
        "",
    ]
    lines += wrap(
        "The price is exactly affine in the coupon, so this is solved from "
        "three finite-difference calls rather than by root-finding -- and the "
        "third call is there to check the affinity on every quote rather than "
        "assume it. The residual is at machine precision because the solve is "
        "exact, not because a tolerance was met."
    )
    lines.append("")

    if decomposition is not None:
        lines += [
            "  what pays for it",
            f"    volatility         {decomposition.volatility_part:.4%}",
            f"    carry              {decomposition.basis_part:.4%}"
            f"   ({decomposition.basis_share:.1%} of the quote)",
            "",
        ]
        lines += wrap(
            "The split is a difference of two solves, not a sensitivity times a "
            "level: d(par)/d(carry) runs 0.60 near zero against 1.20 at the "
            "measured CSI 1000 carry, so attributing by either slope alone "
            "misstates the carry's share by a third. The two legs also interact "
            "through the drift, so this is the comparison a desk makes when the "
            "futures curve moves rather than an identity."
        )
        lines.append("")
    return lines


def _value_section(mc: SnowballMCResult | None, price: float) -> list[str]:
    lines = ["VALUE AND OUTCOMES", rule(), f"  price                {price:+.8f}"]
    if mc is None:
        lines += [
            "",
            *wrap(
                "No simulation was supplied, so the outcome probabilities are "
                "unavailable. They come from the Monte Carlo rather than the "
                "PDE because they are path statistics, not a price."
            ),
            "",
        ]
        return lines
    lines += [
        f"  simulated price      {mc.price:+.8f}  +/- {mc.std_error:.2e}",
        f"  knock-out            {mc.ko_prob:.2%}",
        f"  knock-in             {mc.ki_prob:.2%}",
        f"  knocked in, not out  {mc.ki_no_ko_prob:.2%}   <- the loss state",
        f"  expected KO time     {mc.expected_ko_time:.4f} years",
        f"  paths                {mc.paths:,}",
        f"  volatility model     {mc.vol_model}",
        "",
    ]
    if not mc.cross_checked_by_a_second_engine:
        lines += wrap(
            "This price has one engine. A Heston snowball's second engine would "
            "be a two-dimensional PDE with a cross-derivative term, which is "
            "not built here -- so it does not carry the weight the flat and "
            "local-volatility prices do."
        )
        lines.append("")
    return lines


def _risk_section(risk: SnowballRisk) -> list[str]:
    g = risk.greeks
    lines = [
        "RISK",
        rule(),
        f"  method               {risk.method}",
        f"  delta                {g.delta:+.8f}   per unit of spot",
        f"  gamma                {g.gamma:+.3e}   per unit of spot squared",
        f"  vega                 {g.vega:+.8f}   per volatility point",
        f"  theta                {g.theta:+.8f}   per calendar day",
        f"  rho                  {g.rho:+.8f}   per rate percentage point",
    ]
    if risk.carry_rho is not None:
        lines.append(f"  carry rho            {risk.carry_rho:+.8f}   per carry percentage point")
    lines.append("")
    if risk.carry_rho is not None:
        drift_half = -risk.carry_rho
        discount_half = g.rho + risk.carry_rho
        lines += wrap(
            "There is one carry sensitivity and deliberately no separate "
            "dividend one: q and basis_pa reach the engines only through their "
            "sum, so dV/dq and dV/dbasis_pa are the same derivative -- the two "
            "bumped prices come back bit-identical. Reporting both would "
            "manufacture an independence the model does not have."
        )
        lines.append("")
        lines += wrap(
            f"rho moves the discount factor and the drift; carry rho moves the "
            f"drift alone. So the drift half of the rate exposure is "
            f"{drift_half:+.6f} and the discounting half is {discount_half:+.6f} "
            f"-- the drift dominates, which is why a basis move is not a rate "
            f"move even though both enter through the same term."
        )
        lines.append("")
    return lines


def _as_float(value: object) -> float:
    """Narrow one verdict field.

    A verdict dict mixes floats, booleans and a prose note, so it is typed as
    ``dict[str, object]`` at the source. Asserting here rather than casting
    means a verdict that ever started carrying a string in a numeric slot
    fails loudly instead of being formatted as one.
    """
    assert isinstance(value, int | float), f"expected a number, got {value!r}"
    return float(value)


def _reconciliation_section(comparison: RiskComparison | None) -> list[str]:
    if comparison is None:
        return []
    lines = ["TWO-ENGINE RECONCILIATION", rule()]
    lines.append(f"  {'greek':<12}{'pde':>14}{'mc':>14}{'gap':>12}{'budget':>12}  ok")
    for name, verdict in comparison.verdicts.items():
        ok = "yes" if verdict["within_budget"] else "NO"
        lines.append(
            f"  {name:<12}{_as_float(verdict['pde']):>14.3e}"
            f"{_as_float(verdict['mc']):>14.3e}"
            f"{_as_float(verdict['gap']):>12.2e}{_as_float(verdict['budget']):>12.2e}  {ok}"
        )
    lines.append("")
    lines.append(f"  all within budget    {'yes' if comparison.agree else 'NO'}")
    lines.append("")

    theta = comparison.verdicts.get("theta")
    if theta is not None:
        share = _as_float(theta.get("mc_error_share_of_value", float("nan")))
        lines += wrap(
            f"Theta is the weakest of these and the newest. The two numbers are "
            f"difference quotients of the same derivative over horizons four "
            f"times apart -- the PDE steps by its grid's dt, the simulation by "
            f"one of its own steps -- so an O(h) gap is expected before either "
            f"is wrong. Three sampling errors come to {share:.0%} of the value, "
            f"so read this as agreement rather than as confirmation."
        )
        lines.append("")
    lines += wrap(
        "The simulated theta comes from advancing the valuation date, not from "
        "shortening the tenor: shortening it deletes a knock-out observation -- "
        "nine become eight on the reference trade -- and would measure a "
        "contract change instead of decay. That is why this row read nan until "
        "this phase."
    )
    lines.append("")
    return lines


def snowball_report(
    terms: SnowballTerms,
    market: EquityMarket,
    risk: SnowballRisk,
    *,
    carry: CarryMeasurement | None = None,
    solution: CouponSolution | None = None,
    decomposition: CouponDecomposition | None = None,
    mc: SnowballMCResult | None = None,
    comparison: RiskComparison | None = None,
) -> str:
    """Render the quote report for a snowball note.

    Every section after the header is optional, because each costs solves and
    a caller should be able to pay for what it wants. What is *not* optional is
    the carry section: it prints even with nothing measured, and says so, since
    an absent measurement is the most important thing a reader could know about
    the quote rather than a reason to omit the heading.

    Args:
        terms: Contract terms.
        market: Market state the risk was taken around.
        risk: Greeks, with their provenance.
        carry: A futures measurement, if one was taken.
        solution: A solved par coupon, if one was solved.
        decomposition: The quote's volatility/carry split.
        mc: A simulation, for the outcome probabilities.
        comparison: A two-engine reconciliation.

    Returns:
        The report as plain text.
    """
    sections: list[str] = []
    sections += _header(terms, market)
    sections += _carry_section(carry, market, terms)
    sections += _quote_section(solution, decomposition)
    sections += _value_section(mc, risk.price)
    sections += _risk_section(risk)
    sections += _reconciliation_section(comparison)
    sections.append(rule("="))
    return "\n".join(sections)
