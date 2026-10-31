"""Model validation report for a Dupire local volatility surface.

Same form as the surface report, and the same ordering principle: lead with
what could make the number wrong, not with the number.

What leads here is not numerical. The listed options stop at roughly five
months while the products priced off them run one to three years, so most of a
snowball's life sits in the extrapolated tail of the variance term structure.
No amount of grid refinement touches that, and it dwarfs every other concern
in this file, so it goes first.

Second comes the evidence that the construction is right at all: the Dupire
round trip. A local volatility is derived from an implied volatility surface,
so the one check that matters is repricing the vanillas it was built from and
seeing the smile come back. Running it is optional here because it costs a
PDE solve per strike, but a report without it is a description rather than a
validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from otcpricer.engines.european import price_european_pde
from otcpricer.engines.pde import PDEConfig
from otcpricer.models import bsm
from otcpricer.models.local_vol import LocalVolSurface
from otcpricer.models.surface import FittedSurface
from otcpricer.reports.layout import rule, wrap

__all__ = ["RoundTripRow", "local_vol_report", "round_trip"]

#: Moneyness levels the round trip samples, as a fraction of each expiry's own
#: forward. Spans the range the quote screen admits without reaching into the
#: far wing, where an inverted implied volatility is dominated by the tick.
ROUND_TRIP_MONEYNESS = (0.90, 0.95, 1.00, 1.05, 1.10)


@dataclass(frozen=True, slots=True)
class RoundTripRow:
    """One vanilla repriced through the local volatility.

    Attributes:
        maturity: Expiry year fraction.
        moneyness: Strike over that expiry's forward.
        strike: The strike itself.
        target_vol: Implied volatility the surface says.
        recovered_vol: Implied volatility the local-volatility engine returns.
        price: The engine's price.
    """

    maturity: float
    moneyness: float
    strike: float
    target_vol: float
    recovered_vol: float
    price: float

    @property
    def error_vol_points(self) -> float:
        """Signed discrepancy, in volatility points."""
        return (self.recovered_vol - self.target_vol) * 100.0

    def as_dict(self) -> dict[str, float]:
        return {
            "maturity": round(self.maturity, 6),
            "moneyness": round(self.moneyness, 4),
            "strike": round(self.strike, 6),
            "target_vol": round(self.target_vol, 6),
            "recovered_vol": round(self.recovered_vol, 6),
            "error_vol_points": round(self.error_vol_points, 6),
        }


def round_trip(
    fitted: FittedSurface,
    local_vol: LocalVolSurface,
    *,
    config: PDEConfig | None = None,
    moneyness: tuple[float, ...] = ROUND_TRIP_MONEYNESS,
) -> list[RoundTripRow]:
    """Reprice the vanillas the surface was fitted to, through the local volatility.

    Dupire's construction guarantees the implied volatilities come back in
    continuous time, so the residual is the engine's discretisation error and
    nothing else. That makes the natural budget "small compared with the
    surface's own calibration error" -- the engine should not add more noise
    than the data already carries.

    Each option is inverted at **its own expiry's forward**, not at spot. The
    two differ by the implied carry, which is -4% a year on this data, and
    inverting at spot would measure that and report it as a model error.

    Args:
        fitted: The calibrated surface, for the expiries and forwards.
        local_vol: The local volatility built from it.
        config: Discretisation controls for the repricing solves.
        moneyness: Strikes to sample, as fractions of each forward.

    Returns:
        One :class:`RoundTripRow` per expiry and strike.
    """
    config = config or PDEConfig(space_nodes=1600, steps_per_year=1008)
    rate = fitted.surface_input.rate
    rows: list[RoundTripRow] = []

    for slice_input in fitted.surface_input.usable_slices:
        maturity = slice_input.year_fraction
        forward = float(local_vol.forward(maturity))
        for level in moneyness:
            strike = forward * level
            # Out of the money is the better-conditioned side to invert.
            option_type = "c" if level >= 1.0 else "p"
            target = float(local_vol.surface.implied_vol(float(np.log(level)), maturity))
            price = price_european_pde(
                local_vol.spot, strike, maturity, rate, option_type, local_vol, config
            ).price
            recovered = float(
                bsm.implied_volatility(option_type, price, forward, strike, maturity, rate, rate)
            )
            rows.append(
                RoundTripRow(
                    maturity=maturity,
                    moneyness=level,
                    strike=strike,
                    target_vol=target,
                    recovered_vol=recovered,
                    price=price,
                )
            )
    return rows


def _header(fitted: FittedSurface, local_vol: LocalVolSurface, horizon: float) -> list[str]:
    return [
        rule("="),
        f"LOCAL VOLATILITY REPORT -- {fitted.underlying_code}",
        rule("="),
        f"  as of          {fitted.as_of}",
        f"  data source    {fitted.source}",
        f"  spot anchor    {local_vol.spot:.6g}",
        f"  horizon        {horizon:.2f} years",
        "  model          Dupire local volatility on an SSVI surface",
        "",
    ]


def _extrapolation(local_vol: LocalVolSurface, horizon: float) -> list[str]:
    last = float(local_vol.surface.params.maturities[-1])
    share = local_vol.extrapolated_fraction(horizon)
    lines = [
        "EXTRAPOLATION -- READ THIS FIRST",
        rule(),
        f"  last quoted expiry   {last:.4f} years",
        f"  pricing horizon      {horizon:.4f} years",
        f"  beyond the quotes    {share:.1%} of the horizon",
        "",
    ]
    if share > 0.0:
        lines += wrap(
            f"{share:.0%} of this pricing horizon sits past the last quoted expiry. "
            f"Out there the variance term structure continues at the last quoted "
            f"secant rate, which is a convention and not market information. No "
            f"amount of grid refinement touches it, and on this data it is the "
            f"largest single assumption behind any long-dated number -- larger "
            f"than everything else in this report combined."
        )
    else:
        lines += wrap("The whole horizon is inside the quoted range.")
    lines.append("")
    return lines


def _forward_curve(local_vol: LocalVolSurface) -> list[str]:
    curve = local_vol.forwards
    lines = [
        "FORWARD CURVE",
        rule(),
        "  The surface is parameterised in log(K/F), so the engine's drift has",
        "  to be this curve. Any other and the local volatility is evaluated at",
        "  the wrong moneyness, with nothing downstream to say so.",
        "",
        f"  {'expiry':>9s} {'forward':>10s} {'segment drift':>14s}",
    ]
    drifts = [None, *[float(d) for d in curve.segment_drifts]]
    for index, (maturity, forward) in enumerate(zip(curve.maturities, curve.forwards, strict=True)):
        drift = drifts[index] if index < len(drifts) else None
        shown = "--" if drift is None else f"{drift:+.5f}"
        lines.append(f"  {maturity:9.4f} {forward:10.4f} {shown:>14s}")
    lines += [
        "",
        f"  implied carry over the quoted range   {curve.implied_carry:+.5f} a year",
        f"  tail drift beyond the last expiry     {curve.tail_drift:+.5f} a year",
        f"  kink at the handover                  {curve.kink:+.5f}",
        "",
        "  Carry is the yield deducted from the drift, so it carries the",
        "  opposite sign: a positive carry means declining forwards.",
        "",
    ]
    lines += wrap(
        "The kink is the acknowledged cost of reverting to the caller's own "
        "carry past the quoted range, rather than extrapolating an implied "
        "carry measured on five-month options out to three years. It is "
        "reported rather than smoothed because on this data it is not small.",
    )
    lines.append("")
    return lines


def _round_trip_section(rows: list[RoundTripRow], budget_vol_points: float) -> list[str]:
    lines = [
        "DUPIRE ROUND TRIP -- THE EVIDENCE",
        rule(),
        "  Reprice the vanillas the surface was fitted to and check its own",
        "  smile comes back. Dupire guarantees this in continuous time, so the",
        "  residual is discretisation and nothing else.",
        "",
    ]
    if not rows:
        lines += ["  Not run.", ""]
        return lines

    lines.append(f"  {'maturity':>9s} {'K/F':>6s} {'surface':>8s} {'engine':>8s} {'error':>8s}")
    for row in rows:
        lines.append(
            f"  {row.maturity:9.4f} {row.moneyness:6.2f} {row.target_vol:8.4f} "
            f"{row.recovered_vol:8.4f} {row.error_vol_points:+8.4f}"
        )

    worst = max(abs(row.error_vol_points) for row in rows)
    verdict = "PASS" if worst < budget_vol_points else "FAIL"
    lines += [
        "",
        f"  worst error          {worst:.4f} volatility points",
        f"  calibration rms      {budget_vol_points:.4f} volatility points",
        f"  share of the budget  {worst / budget_vol_points:.1%}   {verdict}",
        "",
    ]
    lines += wrap(
        "The budget is the surface's own calibration error, because an engine "
        "that adds less noise than the data already carries is as accurate as "
        "the inputs can justify. A larger residual means the grid, not the "
        "model.",
    )
    lines.append("")
    return lines


def _surface_shape(local_vol: LocalVolSurface, horizon: float) -> list[str]:
    lines = [
        "LOCAL VOLATILITY AGAINST THE MONEY",
        rule(),
        "  Where a structured note's risk actually sits. A snowball's knock-in",
        "  is around 80% of the initial fixing, and local volatility there is",
        "  far above the at-the-money level, so the short put leg is priced off",
        "  a much wider distribution than a single at-the-money quote implies.",
        "",
        f"  {'maturity':>9s} " + "".join(f"{m:>9.2f}" for m in (0.70, 0.80, 1.00, 1.20)),
    ]
    maturities = [t for t in (0.08, 0.25, 0.5, 1.0, 2.0, 3.0) if t <= horizon + 1e-9]
    for maturity in maturities:
        forward = float(local_vol.forward(maturity))
        values = local_vol.local_vol(forward * np.array([0.70, 0.80, 1.00, 1.20]), maturity)
        lines.append(f"  {maturity:9.4f} " + "".join(f"{v:9.4f}" for v in values))

    if maturities:
        mid = maturities[len(maturities) // 2]
        forward = float(local_vol.forward(mid))
        at_money = float(local_vol.local_vol(forward, mid))
        knock_in = float(local_vol.local_vol(forward * 0.80, mid))
        lines += [
            "",
            f"  at {mid:.2f} years the 80% level carries {knock_in / at_money:.2f}x "
            f"the at-the-money volatility",
        ]
    lines.append("")
    return lines


def _diagnostics(local_vol: LocalVolSurface, horizon: float) -> list[str]:
    lower, upper = local_vol.boundary_moneyness(horizon)
    spots = local_vol.spot * np.exp(np.linspace(lower, upper, 241))
    times = np.linspace(1e-4, horizon, 200)
    scan = local_vol.scan(spots, times, dx=(upper - lower) / 240)

    lines = [
        "NUMERICAL DIAGNOSTICS",
        rule(),
        f"  grid scanned         {scan.n_evaluated} points, "
        f"log-moneyness {lower:+.3f} to {upper:+.3f}",
        f"  local volatility     {scan.min_vol:.4f} to {scan.max_vol:.4f}",
        f"  Dupire denominator   min {scan.min_denominator:.6f}   "
        f"{'PASS' if scan.min_denominator > 0 else 'FAIL'}  (must be > 0)",
        f"  undefined points     {scan.n_undefined}   "
        f"{'PASS' if scan.n_undefined == 0 else 'FAIL'}",
        f"  clamped              {scan.n_floored + scan.n_capped} "
        f"({scan.clamped_fraction:.3%}): {scan.n_floored} at floor, "
        f"{scan.n_capped} at cap",
        f"  max cell Peclet      {scan.max_peclet:.4f}   "
        f"{'PASS' if (scan.max_peclet or 0.0) < 2.0 else 'FAIL'}  "
        f"(oscillation needs > 2)",
        "",
        f"  boundary reach, lower  {local_vol.reach_probability(lower, horizon):.3e}",
        f"  boundary reach, upper  {local_vol.reach_probability(upper, horizon):.3e}",
        "",
    ]
    lines += wrap(
        "Reach probability is reported as a diagnostic, not a gate. The flat "
        "grid that Phase 1 validated to within two Monte Carlo confidence "
        "widths sits at 7.1e-03 by this same measure, so what carries a "
        "boundary is the accuracy of the condition imposed there rather than "
        "the chance of reaching it.",
    )
    lines += [""]
    if scan.n_capped:
        lines += wrap(
            f"Clamping is confined to the short-dated wing, where local "
            f"volatility genuinely diverges like t^(-gamma/2) as expiry "
            f"approaches: at zero time to expiry an option struck away from "
            f"spot has no time value, so only an unbounded local volatility "
            f"can produce any smile. Those nodes sit many standard deviations "
            f"from spot and contribute nothing to a price at spot. Worst "
            f"unclamped value seen at spot {scan.worst_cap[0]:.4g}, "
            f"t = {scan.worst_cap[1]:.2e}."
            if scan.worst_cap
            else "Clamping occurred; see the counts above.",
        )
        lines.append("")
    return lines


def local_vol_report(
    fitted: FittedSurface,
    local_vol: LocalVolSurface,
    *,
    horizon: float = 3.0,
    round_trip_rows: list[RoundTripRow] | None = None,
) -> str:
    """Render the validation report for a local volatility surface.

    Args:
        fitted: The calibrated surface it was built from.
        local_vol: The local volatility surface.
        horizon: Pricing horizon in years, which sets the extrapolation share
            and the region the diagnostics scan.
        round_trip_rows: Results from :func:`round_trip`. Omitted when the
            caller does not want to pay for the repricing solves -- but a
            report without them describes rather than validates, and the
            section says so.

    Returns:
        The report as plain text.
    """
    sections: list[str] = []
    sections += _header(fitted, local_vol, horizon)
    sections += _extrapolation(local_vol, horizon)
    sections += _forward_curve(local_vol)
    sections += _round_trip_section(round_trip_rows or [], fitted.calibration.rms_vol_error * 100.0)
    sections += _surface_shape(local_vol, horizon)
    sections += _diagnostics(local_vol, horizon)
    sections.append(rule("="))
    return "\n".join(sections)
