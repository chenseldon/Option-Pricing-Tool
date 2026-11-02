"""Plain-text validation report for a Heston fit.

Ordered by what a reader most needs to be told, which here is not the fit
quality. Three facts lead, because each one is a reason to trust the number
less than its decimal places suggest:

1. **Heston fits these quotes worse than SSVI does**, by 10% on three expiries
   and 38-41% on six. Anyone reaching for Heston because it is the more
   sophisticated model should see that first.
2. **The Feller condition is violated**, so the variance reaches zero -- on the
   calibrated parameters, in 32-46% of simulated cells.
3. **Whether a second optimiser start corroborated the answer**, which is a
   different question from whether the optimiser converged, and the one that
   actually fails.
"""

from __future__ import annotations

import numpy as np

from otcpricer.models.heston import HestonParams, cumulants
from otcpricer.models.heston_calibration import HestonCalibration
from otcpricer.reports.layout import WIDTH, rule, wrap

__all__ = ["heston_report"]


def _parameters_block(params: HestonParams) -> list[str]:
    return [
        "FITTED PARAMETERS",
        rule(),
        "",
        f"  v0      {params.v0:12.6f}   initial variance   "
        f"({params.initial_vol:6.2%} as a volatility)",
        f"  theta   {params.theta:12.6f}   long-run variance  "
        f"({params.long_run_vol:6.2%} as a volatility)",
        f"  kappa   {params.kappa:12.6f}   mean reversion speed",
        f"  sigma   {params.sigma:12.6f}   volatility of variance",
        f"  rho     {params.rho:+12.6f}   correlation",
        "",
    ]


def _feller_block(params: HestonParams) -> list[str]:
    lines = ["FELLER CONDITION", rule(), ""]
    lines.append(
        f"  2*kappa*theta - sigma^2 = {params.feller_margin:+.6f}"
        f"   ({'satisfied' if params.satisfies_feller else 'VIOLATED'})"
    )
    lines.append("")
    if params.satisfies_feller:
        lines += wrap(
            "The variance stays strictly positive, so any reasonable "
            "discretisation would do. The QE scheme is still used, for one "
            "code path rather than two."
        )
    else:
        lines += wrap(
            "The variance process can reach zero, and on parameters like these "
            "it does so constantly -- measured at 45.8% of all simulated cells "
            "on the 510050 fit and 31.6% on 510300, with essentially every "
            "path touching zero at some point. That is not an edge case to be "
            "guarded against; it is the typical state of the process."
        )
        lines.append("")
        lines += wrap(
            "Consequences, in order of importance. A Euler discretisation "
            "would step the variance negative and sqrt(v) would fail, so "
            "Andersen's QE scheme is a requirement here rather than a "
            "refinement. And the condition is not imposed during calibration: "
            "constraining it would fit the quotes worse without making the "
            "model more true, so it is reported instead."
        )
    lines.append("")
    return lines


def _fit_block(calibration: HestonCalibration, ssvi_rms_vol_points: float | None) -> list[str]:
    lines = ["CALIBRATION ERROR", rule(), ""]
    lines.append(f"  observations           {calibration.n_points:8d}")
    lines.append(f"  rms error             {calibration.rms_vol_points:9.4f} vol points")
    lines.append(f"  worst error           {calibration.max_vol_error * 100:9.4f} vol points")
    if calibration.n_unidentifiable:
        lines.append(
            f"  unscorable points      {calibration.n_unidentifiable:8d}   "
            f"(model price could not be inverted)"
        )
    lines.append("")

    if ssvi_rms_vol_points is not None and ssvi_rms_vol_points > 0.0:
        ratio = calibration.rms_vol_points / ssvi_rms_vol_points
        lines.append(f"  SSVI on the same quotes {ssvi_rms_vol_points:7.4f} vol points")
        lines.append(f"  Heston / SSVI           {ratio:7.2f}x")
        lines.append("")
        if ratio > 1.0:
            lines += wrap(
                f"Heston fits these quotes {(ratio - 1.0) * 100:.0f}% worse than SSVI, "
                "and that is expected rather than wrong: SSVI carries three "
                "global parameters plus one per expiry, so it has more freedom "
                "in the term structure as the number of expiries grows. "
                "Measured across both data sources, Heston is 10-12% worse on "
                "three expiries and 38-41% worse on six."
            )
            lines.append("")
            lines += wrap(
                "So Heston is not bought here for accuracy. It is bought for a "
                "forward smile that does not flatten -- the one thing a Dupire "
                "local volatility cannot provide, since that reprices today's "
                "vanillas by construction and then implies a forward smile "
                "known to be wrong. For a path-dependent payoff that is the "
                "trade worth making; for a vanilla it is not."
            )
            lines.append("")
    return lines


def _identification_block(calibration: HestonCalibration) -> list[str]:
    lines = ["IDENTIFICATION", rule(), ""]
    lines.append(f"  optimiser converged        {str(calibration.converged):>5}")
    lines.append(
        f"  starts reaching the best   {calibration.n_comparable_starts:5d}"
        f" of {calibration.n_starts}"
    )
    lines.append(f"  parameter disagreement     {calibration.parameter_spread:8.4f}")
    lines.append(f"  corroborated by a second   {str(calibration.confirmed_by_multiple_starts):>5}")
    lines.append(f"  parameters ambiguous       {str(calibration.ambiguous):>5}")
    lines.append("")

    lines += wrap(
        "These are three separate facts and collapsing them would hide the "
        "failure that actually happens. On a synthetic surface generated from "
        "known parameters, one start of four recovered every one of them to "
        "four decimals while the next best stopped a thousand times worse, "
        "with kappa at 8.09 against a true 2.00 -- and both reported "
        "successful convergence."
    )
    lines.append("")

    if calibration.ambiguous:
        lines += wrap(
            "Comparable starts disagreed on the parameters. This is the "
            "kappa/sigma trade-off: a fast mean reversion with a large "
            "vol-of-vol looks much like a slow one with a small one over a "
            "year of quotes, so the fit is good and the parameters are not "
            "pinned down. Prices will be more stable than parameters."
        )
        lines.append("")
    elif not calibration.confirmed_by_multiple_starts:
        lines += wrap(
            "Only one start reached this minimum, so nothing corroborates it. "
            "That is not evidence the fit is wrong -- it may simply be the "
            "only start that escaped a local valley -- but it is not evidence "
            "that it is right either."
        )
        lines.append("")
    return lines


def _moments_block(params: HestonParams, maturities: tuple[float, ...]) -> list[str]:
    lines = ["LOG-RETURN MOMENTS", rule(), ""]
    lines.append(f"  {'T':>6}  {'mean':>12}  {'variance':>12}  {'implied vol':>12}")
    for maturity in maturities:
        c1, c2 = cumulants(params, maturity)
        vol = float(np.sqrt(c2 / maturity)) if maturity > 0 else float("nan")
        lines.append(f"  {maturity:6.3f}  {c1:12.6f}  {c2:12.6f}  {vol:11.4%}")
    lines.append("")
    lines += wrap(
        "Computed by differentiating the characteristic function rather than "
        "from a closed form. A transcribed closed form was wrong by a factor "
        "of five in a high-volatility regime, and since the variance sizes the "
        "COS truncation window the result was finite, plausible and wrong "
        "prices. Differentiating the function the engine actually integrates "
        "makes that class of error impossible."
    )
    lines.append("")
    return lines


def heston_report(
    calibration: HestonCalibration,
    *,
    underlying: str = "",
    as_of: str = "",
    ssvi_rms_vol_points: float | None = None,
    maturities: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 3.0),
) -> str:
    """Render a Heston calibration as plain text.

    Args:
        calibration: The fit to report.
        underlying: Underlying code, for the header.
        as_of: Observation date, for the header.
        ssvi_rms_vol_points: SSVI's error on the same quotes, which turns the
            fit quality into a comparison instead of a bare number. Omitted
            when there is nothing to compare against.
        maturities: Tenors for the moment table.
    """
    params = calibration.params

    header = "HESTON CALIBRATION"
    if underlying:
        header += f" -- {underlying}"
    if as_of:
        header += f" @ {as_of}"

    # Double rule above and below the title, matching surface_report.
    lines: list[str] = [rule("="), header, rule("="), ""]

    # The caveats lead, because each is a reason to trust the number less than
    # its decimal places suggest.
    if ssvi_rms_vol_points is not None and calibration.rms_vol_points > ssvi_rms_vol_points:
        lines += wrap(
            "READ THIS FIRST: this model fits these quotes worse than the SSVI "
            "surface does. It is here for its dynamics, not its accuracy."
        )
        lines.append("")
    if not params.satisfies_feller:
        lines += wrap(
            "AND THIS: the Feller condition is violated, so the variance "
            "reaches zero routinely. Simulation requires the QE scheme."
        )
        lines.append("")

    lines += _parameters_block(params)
    lines += _fit_block(calibration, ssvi_rms_vol_points)
    lines += _feller_block(params)
    lines += _identification_block(calibration)
    lines += _moments_block(params, maturities)

    lines.append("ONE ENGINE")
    lines.append(rule())
    lines.append("")
    lines += wrap(
        "European prices under this model are cross-checked two ways: the COS "
        "engine against QuantLib's analytic Heston engine, to 5e-09, and the "
        "QE simulation against COS, within three standard errors. A "
        "path-dependent payoff is not. The second engine would be a "
        "two-dimensional PDE with a cross-derivative term and an ADI solver, "
        "which is not built here, so a Heston snowball rests on one engine "
        "where every other number in this project rests on two. That is a "
        "deliberate departure and it is recorded rather than glossed."
    )
    lines.append("")
    lines.append("=" * WIDTH)
    return "\n".join(lines)
