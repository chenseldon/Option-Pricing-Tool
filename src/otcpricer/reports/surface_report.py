"""Model validation report for a fitted volatility surface.

The form a sell-side model validation note takes, because the question a
reviewer asks is never "what is the surface" but "why should I believe it". So
the report leads with the things that could make it wrong:

* per-expiry calibration error, in volatility points, against the quotes used;
* the put-call parity residual for each expiry, which is the most direct
  measure of how non-synchronous the input quotes were;
* every discarded quote, counted by reason, reconciled against the raw count;
* the independent arbitrage check, and whether a local volatility follows.

Plain text rather than HTML or a notebook: it has to be readable in a terminal,
diffable in git, and quotable in a commit message. Numbers are rendered at the
precision they are actually known to -- volatility to a hundredth of a point,
not to six decimals that imply a confidence the inputs do not support.
"""

from __future__ import annotations

from otcpricer.models.arbitrage import DEFAULT_DEGENERACY_FLOOR
from otcpricer.models.surface import FittedSurface
from otcpricer.reports.layout import rule as _rule
from otcpricer.reports.layout import wrap as _wrap

__all__ = ["surface_report"]


def _header(fitted: FittedSurface) -> list[str]:
    underlying = fitted.underlying_code
    return [
        _rule("="),
        f"VOLATILITY SURFACE CALIBRATION REPORT -- {underlying}",
        _rule("="),
        f"  as of          {fitted.as_of}",
        f"  data source    {fitted.source}",
        f"  rate assumed   {fitted.surface_input.rate:.3%} continuously compounded",
        "  model          SSVI (Gatheral-Jacquier), power-law phi",
        "",
    ]


def _parameters(fitted: FittedSurface) -> list[str]:
    params = fitted.calibration.params
    lines = [
        "FITTED PARAMETERS",
        _rule(),
        f"  rho    {params.rho:+.4f}   slope asymmetry; negative is the equity skew",
        f"  eta    {params.eta:.4f}   curvature scale",
        f"  gamma  {params.gamma:.4f}   curvature decay with maturity",
        "",
        "  at-the-money total variance term structure",
        f"    {'maturity':>10s}  {'theta':>10s}  {'atm vol':>9s}",
    ]
    for maturity, theta in zip(params.maturities, params.thetas, strict=True):
        atm = (theta / maturity) ** 0.5
        lines.append(f"    {maturity:10.4f}  {theta:10.6f}  {atm:9.2%}")
    lines.append("")
    return lines


def _calibration_table(fitted: FittedSurface) -> list[str]:
    calibration = fitted.calibration
    lines = [
        "CALIBRATION ERROR, IN VOLATILITY POINTS",
        _rule(),
        f"  {'expiry':>8s} {'maturity':>9s} {'pts':>4s} {'rms':>7s} {'max':>7s} "
        f"{'atm fit':>8s} {'atm obs':>8s} {'diff':>7s}",
    ]
    usable = fitted.surface_input.usable_slices
    for slice_input, quality in zip(usable, calibration.slices, strict=True):
        diff = (quality.atm_vol_fitted - quality.atm_vol_observed) * 100.0
        lines.append(
            f"  {slice_input.expiry_tag:>8s} {quality.maturity:9.4f} "
            f"{quality.n_points:4d} {quality.rms_vol_error * 100:7.3f} "
            f"{quality.max_vol_error * 100:7.3f} {quality.atm_vol_fitted:8.2%} "
            f"{quality.atm_vol_observed:8.2%} {diff:+7.3f}"
        )
    lines += [
        f"  {'pooled':>8s} {'':>9s} {calibration.n_points:4d} "
        f"{calibration.rms_vol_error * 100:7.3f} {calibration.max_vol_error * 100:7.3f}",
        "",
        f"  optimiser: {'converged' if calibration.converged else 'DID NOT CONVERGE'}"
        f" -- {calibration.message}",
        "",
    ]
    return lines


def _forward_table(fitted: FittedSurface) -> list[str]:
    lines = [
        "FORWARDS AND PUT-CALL PARITY RESIDUALS",
        _rule(),
        "  The residual measures how far the quotes across an expiry are out of",
        "  line with each other. The board carries last traded prices with no",
        "  bid/ask, so those trades happened at different times; this is the most",
        "  direct evidence of how much that matters.",
        "",
        f"  {'expiry':>8s} {'forward':>9s} {'strikes':>8s} {'rms':>9s} "
        f"{'max':>9s} {'rel':>8s}  status",
    ]
    for slice_input in fitted.surface_input.slices:
        forward = slice_input.forward
        if slice_input.forward.n_strikes == 0:
            lines.append(
                f"  {slice_input.expiry_tag:>8s} {'--':>9s} {0:8d} {'--':>9s} "
                f"{'--':>9s} {'--':>8s}  rejected"
            )
            continue
        status = "used" if forward.usable else "rejected"
        lines.append(
            f"  {slice_input.expiry_tag:>8s} {forward.forward:9.4f} "
            f"{forward.n_strikes:8d} {forward.parity_residual_rms:9.6f} "
            f"{forward.parity_residual_max:9.6f} {forward.residual_relative:8.4%}  {status}"
        )
    lines.append("")
    return lines


def _rejection_table(fitted: FittedSurface) -> list[str]:
    surface_input = fitted.surface_input
    lines = [
        "DISCARDED QUOTES",
        _rule(),
    ]
    if not surface_input.rejections:
        lines += ["  None. Every quote in the chain was used.", ""]
        return lines

    lines.append("  Pooled across expiries, by reason:")
    for reason, count in surface_input.rejections.items():
        lines.append(f"    {count:5d}  {reason}")
    lines += [
        f"    {surface_input.n_rejected:5d}  total discarded",
        f"    {surface_input.n_points:5d}  retained and fitted",
        "",
        "  Every quote is either retained or counted above exactly once, so these",
        "  reconcile against the raw chain. Per expiry:",
        "",
        f"    {'expiry':>8s} {'kept':>5s}  rejections",
    ]
    for slice_input in surface_input.slices:
        detail = (
            ", ".join(f"{reason} x{count}" for reason, count in slice_input.rejections.items())
            or "none"
        )
        lines.append(f"    {slice_input.expiry_tag:>8s} {len(slice_input.points):5d}  {detail}")
    lines.append("")
    return lines


def _excluded_expiries(fitted: FittedSurface) -> list[str]:
    unusable = [s for s in fitted.surface_input.slices if not s.usable]
    if not unusable:
        return []
    lines = [
        "EXPIRIES EXCLUDED FROM THE FIT",
        _rule(),
    ]
    for slice_input in unusable:
        lines.append(f"  {slice_input.expiry_tag}  ({slice_input.year_fraction:.4f}y)")
        lines += _wrap(slice_input.reason, indent="      ")
    lines.append("")
    return lines


def _arbitrage_section(fitted: FittedSurface) -> list[str]:
    report = fitted.arbitrage
    lines = [
        "STATIC ARBITRAGE CHECK",
        _rule(),
        "  An independent check, not a restatement of the calibration",
        "  constraints: the conditions are evaluated pointwise on a dense grid,",
        "  spanning maturities inside the first quoted expiry and past the last,",
        "  because that is where the interpolated term structure governs rather",
        "  than the fitted constraints.",
        "",
        f"  grid               {report.n_checked} points, log-moneyness "
        f"+/-{report.moneyness_range:.2f}, "
        f"maturities {report.maturity_range[0]:.4f} to {report.maturity_range[1]:.4f}",
        f"  butterfly  g(k)    min {report.min_butterfly:+.6f}   "
        f"{'PASS' if report.butterfly_free else 'FAIL'}  (must be > 0)",
        f"  calendar   dw/dT   min {report.min_calendar:+.6e}   "
        f"{'PASS' if report.calendar_free else 'FAIL'}  (must be >= 0)",
        f"  calendar, relative  min {report.min_calendar_relative:+.6f}   "
        f"{'FAIL' if report.calendar_degenerate else 'PASS'}  "
        f"(must be > {DEFAULT_DEGENERACY_FLOOR:.2f})",
        "",
        *_wrap(
            "The relative measure is dw/dT against the surface's own average "
            "variance rate theta/T. A flat term structure satisfies dw/dT >= 0 "
            "and still leaves local variance at zero, so the absolute test "
            "above is not sufficient on its own.",
            indent="    ",
        ),
        "",
        "  constraint margins reported by the calibration:",
        f"    butterfly  {fitted.calibration.butterfly_margin:+.6f}  "
        f"(headroom below the Gatheral-Jacquier bound of 4)",
        f"    calendar   {fitted.calibration.calendar_margin:+.6e}  "
        f"(smallest increment in the variance term structure)",
        "",
        f"  conclusion: {report.summary()}",
    ]
    if report.violations:
        lines += [
            "",
            f"  worst violations ({report.n_violations} in total, "
            f"{len(report.violations)} retained):",
            f"    {'condition':>10s} {'k':>8s} {'maturity':>9s} {'value':>14s}",
        ]
        for violation in report.violations[:10]:
            lines.append(
                f"    {violation.condition:>10s} {violation.log_moneyness:8.4f} "
                f"{violation.maturity:9.4f} {violation.value:14.6e}"
            )
    lines.append("")
    return lines


def _local_volatility_section(fitted: FittedSurface) -> list[str]:
    ok, reason = fitted.usable_for_local_volatility
    return [
        "LOCAL VOLATILITY PRECONDITION",
        _rule(),
        "  Dupire's denominator is identically Gatheral's g(k) and its numerator",
        "  is dw/dT, so the two checks above settle whether a local volatility",
        "  can be derived from this surface at all.",
        "",
        f"  verdict: {'USABLE' if ok else 'NOT USABLE'}",
        *_wrap(reason),
        "",
        _rule("="),
    ]


def surface_report(fitted: FittedSurface) -> str:
    """Render the full validation report for a fitted surface.

    Args:
        fitted: The surface and its evidence, from
            :func:`~otcpricer.models.surface.fit_surface`.

    Returns:
        The report as plain text.
    """
    sections: list[str] = []
    sections += _header(fitted)
    sections += _parameters(fitted)
    sections += _calibration_table(fitted)
    sections += _forward_table(fitted)
    sections += _rejection_table(fitted)
    sections += _excluded_expiries(fitted)
    sections += _arbitrage_section(fitted)
    sections += _local_volatility_section(fitted)
    return "\n".join(sections)
