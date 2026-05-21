"""
app.py - Flask Web Application Backend (Enhanced)

Complete OTC option pricing tool backend with full risk analytics.

Endpoints:
    GET  /                      : Serve the web interface (index.html)
    POST /api/calculate         : BSM pricing + Monte Carlo + Greeks + chart data
    POST /api/iv                : Implied volatility solver
    POST /api/portfolio         : Portfolio exposure aggregation (MTM + margin)
    POST /api/export            : Export results to Excel (.xlsx)
    POST /api/vol_smile         : Volatility smile from market prices
    POST /api/mc_distribution   : Monte Carlo payoff distribution histogram
    POST /api/sensitivity       : Sensitivity / stress test matrix

Usage:
    python app.py
    Then open: http://127.0.0.1:5000

Dependencies:
    pip install flask py_vollib numpy pandas matplotlib openpyxl

Author: GitHub Portfolio Project
"""

import os
import sys
import numpy as np
from io import BytesIO
from flask import Flask, render_template, request, jsonify, send_file
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

# Ensure sibling modules (bs_pricing, greeks_calc) are importable
sys.path.insert(0, os.path.dirname(__file__))

from bs_pricing import bs_price, monte_carlo_price, calc_implied_volatility, validate_params
from greeks_calc import calc_delta, calc_gamma, calc_vega, calc_theta, calc_rho

app = Flask(__name__)


# ── Shared helper: compute all 5 Greeks at one point ──────────────────────────

def _greeks_at(flag, S, K, T, r, sigma, q):
    return {
        "Delta": round(float(calc_delta(flag, S, K, T, r, sigma, q)), 6),
        "Gamma": round(float(calc_gamma(flag, S, K, T, r, sigma, q)), 6),
        "Vega":  round(float(calc_vega (flag, S, K, T, r, sigma, q)), 6),
        "Theta": round(float(calc_theta(flag, S, K, T, r, sigma, q)), 6),
        "Rho":   round(float(calc_rho  (flag, S, K, T, r, sigma, q)), 6),
    }


# ─────────────────────────────────────────────────────────
# Frontend Route
# ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main web interface page."""
    return render_template("index.html")


# ─────────────────────────────────────────────────────────
# API: Full Calculation (Price + Greeks + Chart Data)
# ─────────────────────────────────────────────────────────

@app.route("/api/calculate", methods=["POST"])
def calculate():
    """BSM + Monte Carlo pricing, all Greeks, and chart curve data."""
    try:
        data  = request.get_json(force=True)
        flag  = str(data.get("flag", "c")).lower()
        S     = float(data["S"])
        K     = float(data["K"])
        T     = float(data["T"])
        r     = float(data["r"])
        sigma = float(data["sigma"])
        q     = float(data.get("q", 0.0))
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid input parameters: {e}"}), 400

    if flag not in ("c", "p"):
        return jsonify({"error": "flag must be 'c' (Call) or 'p' (Put)"}), 400

    valid, msg = validate_params(S, K, T, r, sigma, q)
    if not valid:
        return jsonify({"error": msg}), 400

    bsm_price_val = bs_price(flag, S, K, T, r, sigma, q)
    mc_price_val  = monte_carlo_price(flag, S, K, T, r, sigma, q, n_simulations=100000)
    greeks        = _greeks_at(flag, S, K, T, r, sigma, q)

    # Chart: 100 S-values spanning 0.6K – 1.4K
    S_arr = np.linspace(0.6 * K, 1.4 * K, 100)

    prices_arr    = [round(float(bs_price(flag, s, K, T, r, sigma, q)), 4) for s in S_arr]
    intrinsic_arr = [round(float(max(s - K, 0) if flag == "c" else max(K - s, 0)), 4)
                     for s in S_arr]
    greek_curves  = {
        "Delta": [round(float(calc_delta(flag, s, K, T, r, sigma, q)), 6) for s in S_arr],
        "Gamma": [round(float(calc_gamma(flag, s, K, T, r, sigma, q)), 6) for s in S_arr],
        "Vega":  [round(float(calc_vega (flag, s, K, T, r, sigma, q)), 6) for s in S_arr],
        "Theta": [round(float(calc_theta(flag, s, K, T, r, sigma, q)), 6) for s in S_arr],
        "Rho":   [round(float(calc_rho  (flag, s, K, T, r, sigma, q)), 6) for s in S_arr],
    }

    return jsonify({
        "bsm_price": round(float(bsm_price_val), 4),
        "mc_price":  round(float(mc_price_val),  4),
        "greeks": {k.lower(): v for k, v in greeks.items()},
        "price_curve": {
            "spots":    [round(float(s), 2) for s in S_arr],
            "prices":   prices_arr,
            "intrinsic": intrinsic_arr,
        },
        "greeks_curve": {k.lower(): v for k, v in greek_curves.items()},
        "K": K,
    })


# ─────────────────────────────────────────────────────────
# API: Implied Volatility Solver
# ─────────────────────────────────────────────────────────

@app.route("/api/iv", methods=["POST"])
def implied_vol():
    """Solve for implied volatility from a market price."""
    try:
        data         = request.get_json(force=True)
        flag         = str(data.get("flag", "c")).lower()
        market_price = float(data["market_price"])
        S            = float(data["S"])
        K            = float(data["K"])
        T            = float(data["T"])
        r            = float(data["r"])
        q            = float(data.get("q", 0.0))
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid input parameters: {e}"}), 400

    try:
        iv = calc_implied_volatility(flag, market_price, S, K, T, r, q)
        return jsonify({"implied_volatility": round(float(iv), 6)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


# ─────────────────────────────────────────────────────────
# API: Portfolio Exposure Aggregation
# ─────────────────────────────────────────────────────────

@app.route("/api/portfolio", methods=["POST"])
def portfolio():
    """
    Compute per-position Greeks, MTM, P&L, and margin for a portfolio,
    plus aggregate net Greeks across all positions.

    Each position in request JSON:
        flag (str), S/K/T/r/sigma/q (float), qty (float, signed),
        entry_price (float), label (str), id (str)

    Margin formula (simplified):
        Long  : |qty| * entry_price   (premium paid)
        Short : |qty| * S * 0.15      (15% of underlying, simplified)
    """
    try:
        data      = request.get_json(force=True)
        positions = data.get("positions", [])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    results = []
    totals  = {"net_delta": 0.0, "net_gamma": 0.0, "net_vega": 0.0,
                "net_theta": 0.0, "net_rho": 0.0,
                "total_mtm": 0.0, "total_pnl": 0.0, "total_margin": 0.0}

    for pos in positions:
        try:
            flag  = str(pos["flag"]).lower()
            S     = float(pos["S"])
            K     = float(pos["K"])
            T     = float(pos["T"])
            r     = float(pos["r"])
            sigma = float(pos["sigma"])
            q     = float(pos.get("q", 0.0))
            qty   = float(pos.get("qty", 1.0))
            entry = float(pos.get("entry", pos.get("entry_price", 0.0)))

            price          = float(bs_price(flag, S, K, T, r, sigma, q))
            greeks_single  = _greeks_at(flag, S, K, T, r, sigma, q)

            mtm    = round(price * qty, 4)
            pnl    = round((price - entry) * qty, 4)
            margin = round(abs(qty) * entry, 4) if qty > 0 \
                     else round(abs(qty) * S * 0.15, 4)

            result_pos = {
                "id":        pos.get("id", ""),
                "label":     pos.get("label", ""),
                "flag":      flag, "S": S, "K": K,
                "T":         round(T, 6),
                "T_days":    round(T * 365),
                "r":         r, "sigma": sigma, "q": q,
                "qty":       qty,
                "entry":     round(entry, 4),
                "mtm":       mtm, "pnl": pnl, "margin": margin,
                "delta_net": round(greeks_single["Delta"] * qty, 6),
                "gamma_net": round(greeks_single["Gamma"] * qty, 6),
                "vega_net":  round(greeks_single["Vega"]  * qty, 6),
                "theta_net": round(greeks_single["Theta"] * qty, 6),
                "rho_net":   round(greeks_single["Rho"]   * qty, 6),
                "error":     None,
            }

            totals["net_delta"]    = round(totals["net_delta"]    + result_pos["delta_net"], 6)
            totals["net_gamma"]    = round(totals["net_gamma"]    + result_pos["gamma_net"], 6)
            totals["net_vega"]     = round(totals["net_vega"]     + result_pos["vega_net"],  6)
            totals["net_theta"]    = round(totals["net_theta"]    + result_pos["theta_net"], 6)
            totals["net_rho"]      = round(totals["net_rho"]      + result_pos["rho_net"],   6)
            totals["total_mtm"]    = round(totals["total_mtm"]    + mtm,    4)
            totals["total_pnl"]    = round(totals["total_pnl"]    + pnl,    4)
            totals["total_margin"] = round(totals["total_margin"] + margin, 4)

        except Exception as e:
            result_pos = {**pos, "error": str(e)}

        results.append(result_pos)

    return jsonify({"positions": results, "totals": totals})


# ─────────────────────────────────────────────────────────
# API: Vol Smile Preset Prices
# ─────────────────────────────────────────────────────────

@app.route("/api/smile_presets", methods=["POST"])
def smile_presets():
    """Return BSM prices for a list of strikes — used to pre-populate the Vol Smile input table."""
    try:
        data   = request.get_json(force=True)
        flag   = str(data.get("flag", "c")).lower()
        S      = float(data["S"])
        T      = float(data["T"])
        r      = float(data["r"])
        sigma  = float(data["sigma"])
        q      = float(data.get("q", 0.0))
        strikes = data.get("strikes", [])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    presets = []
    for K_i in strikes:
        try:
            price = round(float(bs_price(flag, S, float(K_i), T, r, sigma, q)), 2)
            presets.append({"K": float(K_i), "price": price})
        except Exception:
            presets.append({"K": float(K_i), "price": 0})
    return jsonify({"presets": presets})

@app.route("/api/vol_smile", methods=["POST"])
def vol_smile():
    """
    Compute implied volatility for a set of (K, market_price) pairs,
    producing a volatility smile / skew curve.

    Request JSON: {flag, S, T, r, q, smile_data: [{K, market_price}]}
    Response JSON: {smile_points: [{K, iv (%), error}]}
    """
    try:
        data       = request.get_json(force=True)
        flag       = str(data.get("flag", "c")).lower()
        S          = float(data["S"])
        T          = float(data["T"])
        r          = float(data["r"])
        q          = float(data.get("q", 0.0))
        smile_data = data.get("pairs", data.get("smile_data", []))
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    points = []
    for item in smile_data:
        try:
            K_i  = float(item["K"])
            mp_i = float(item["market_price"])
            iv   = calc_implied_volatility(flag, mp_i, S, K_i, T, r, q)
            points.append({"K": K_i, "iv": round(float(iv) * 100, 4), "error": None})
        except Exception as e:
            points.append({"K": float(item.get("K", 0)), "iv": None, "error": str(e)})

    return jsonify({"results": points})


# ─────────────────────────────────────────────────────────
# API: Monte Carlo Payoff Distribution
# ─────────────────────────────────────────────────────────

@app.route("/api/mc_distribution", methods=["POST"])
def mc_distribution():
    """
    Run Monte Carlo simulation and return the discounted payoff distribution.
    Visualises the full probability distribution of option value at expiry.

    Request JSON: standard option params + n_simulations (int, max 200000)
    Response JSON: {bins, counts, mean, median, p5, p95, option_price, n_simulations}
    """
    try:
        data  = request.get_json(force=True)
        flag  = str(data.get("flag", "c")).lower()
        S     = float(data["S"])
        K     = float(data["K"])
        T     = float(data["T"])
        r     = float(data["r"])
        sigma = float(data["sigma"])
        q     = float(data.get("q", 0.0))
        n     = min(int(data.get("n_paths", data.get("n_simulations", 50000))), 200000)
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    valid, msg = validate_params(S, K, T, r, sigma, q)
    if not valid:
        return jsonify({"error": msg}), 400

    np.random.seed(42)
    Z   = np.random.standard_normal(n)
    S_T = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    payoffs    = np.maximum(S_T - K, 0) if flag == "c" else np.maximum(K - S_T, 0)
    discounted = np.exp(-r * T) * payoffs

    counts, edges = np.histogram(discounted, bins=50)
    centers       = ((edges[:-1] + edges[1:]) / 2).tolist()

    return jsonify({
        "bins":   [round(x, 4) for x in centers],
        "counts": counts.tolist(),
        "stats": {
            "mean":       round(float(np.mean(discounted)),           4),
            "median":     round(float(np.median(discounted)),         4),
            "ci95_lower": round(float(np.percentile(discounted,  5)), 4),
            "ci95_upper": round(float(np.percentile(discounted, 95)), 4),
            "std":        round(float(np.std(discounted)),            4),
            "n_paths":    n,
        },
    })


# ─────────────────────────────────────────────────────────
# API: Sensitivity / Stress Test Matrix
# ─────────────────────────────────────────────────────────

@app.route("/api/sensitivity", methods=["POST"])
def sensitivity():
    """
    Generate 5x5 sensitivity matrices for option price and Delta under
    combinations of shifted volatility (+/-20%, +/-10%) and time (+/-60d, +/-30d).

    Request JSON: standard option params
    Response JSON: {sigma_labels, T_labels, price_matrix, delta_matrix, base_price}
    """
    try:
        data  = request.get_json(force=True)
        flag  = str(data.get("flag", "c")).lower()
        S     = float(data["S"])
        K     = float(data["K"])
        T     = float(data["T"])
        r     = float(data["r"])
        sigma = float(data["sigma"])
        q     = float(data.get("q", 0.0))
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    valid, msg = validate_params(S, K, T, r, sigma, q)
    if not valid:
        return jsonify({"error": msg}), 400

    sigma_offsets = [-0.20, -0.10, 0.0, +0.10, +0.20]
    T_offsets     = [-60/365, -30/365, 0.0, +30/365, +60/365]
    sigma_labels  = ["σ-20%", "σ-10%", "σ base", "σ+10%", "σ+20%"]
    T_labels      = ["T-60d", "T-30d", "T base", "T+30d", "T+60d"]

    price_matrix, delta_matrix = [], []
    for ds in sigma_offsets:
        price_row, delta_row = [], []
        for dt in T_offsets:
            s_i = max(sigma + ds, 0.005)
            t_i = max(T     + dt, 1 / 365)
            try:
                p = round(float(bs_price(flag, S, K, t_i, r, s_i, q)), 4)
                d = round(float(calc_delta(flag, S, K, t_i, r, s_i, q)), 4)
            except Exception:
                p, d = None, None
            price_row.append(p)
            delta_row.append(d)
        price_matrix.append(price_row)
        delta_matrix.append(delta_row)

    return jsonify({
        "sigma_labels": sigma_labels,
        "t_labels":     T_labels,
        "price_matrix": price_matrix,
        "delta_matrix": delta_matrix,
        "base_price":   round(float(bs_price(flag, S, K, T, r, sigma, q)), 4),
    })


# ─────────────────────────────────────────────────────────
# API: Excel Export
# ─────────────────────────────────────────────────────────

@app.route("/api/export", methods=["POST"])
def export_excel():
    """
    Generate and return an Excel report from the current pricing session.

    Request JSON: {params, result, portfolio}
    Returns: option_pricing_report.xlsx as file download
    """
    try:
        data      = request.get_json(force=True)
        params    = data.get("params",    {})
        result    = data.get("result",    {})
        portfolio = data.get("portfolio", {})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    wb = openpyxl.Workbook()

    HDR_FILL  = PatternFill("solid", fgColor="0D1626")
    HDR_FONT  = Font(color="4E9EFF", bold=True, size=12)
    SUB_FILL  = PatternFill("solid", fgColor="0D3A6B")
    SUB_FONT  = Font(color="FFFFFF", bold=True)
    BOLD      = Font(bold=True)
    CENTER    = Alignment(horizontal="center")

    def hdr(ws, row, ncols, text, fill=HDR_FILL, font=HDR_FONT):
        ws.merge_cells(start_row=row, start_column=1,
                       end_row=row, end_column=ncols)
        c = ws.cell(row=row, column=1, value=text)
        c.fill, c.font, c.alignment = fill, font, CENTER

    def row_vals(ws, r, vals, bold=False, fill=None):
        for ci, v in enumerate(vals, 1):
            c = ws.cell(row=r, column=ci, value=v)
            if bold: c.font = BOLD
            if fill: c.fill = fill

    # ── Sheet 1: Pricing Summary ──────────────────────────
    ws1 = wb.active
    ws1.title = "Pricing Summary"
    ws1.column_dimensions["A"].width = 22
    ws1.column_dimensions["B"].width = 18
    ws1.column_dimensions["C"].width = 36

    r = 1
    hdr(ws1, r, 3, "OTC Option Pricing Summary Report"); r += 1
    hdr(ws1, r, 3, "Input Parameters", SUB_FILL, SUB_FONT); r += 1
    for label, val in [
        ("Option Type",    "Call" if params.get("flag") == "c" else "Put"),
        ("Underlying (S)", params.get("S", "")),
        ("Strike (K)",     params.get("K", "")),
        ("Maturity T (yr)",params.get("T", "")),
        ("Risk-Free r",    params.get("r", "")),
        ("Volatility σ",   params.get("sigma", "")),
        ("Div Yield q",    params.get("q", "")),
    ]:
        row_vals(ws1, r, [label, val]); r += 1

    r += 1
    hdr(ws1, r, 3, "Pricing Results", SUB_FILL, SUB_FONT); r += 1
    row_vals(ws1, r, ["BSM Price",         result.get("bsm_price", "")]); r += 1
    row_vals(ws1, r, ["Monte Carlo Price", result.get("mc_price",  "")]); r += 1

    r += 1
    hdr(ws1, r, 3, "Greeks at Current S", SUB_FILL, SUB_FONT); r += 1
    ws1.cell(row=r, column=1, value="Greek").font = BOLD
    ws1.cell(row=r, column=2, value="Value").font = BOLD
    ws1.cell(row=r, column=3, value="Description").font = BOLD
    r += 1
    greeks_data = result.get("greeks", {})
    for gname, gdesc in [
        ("Delta", "Price change per $1 move in underlying"),
        ("Gamma", "Delta change per $1 move in underlying"),
        ("Vega",  "Price change per 1% move in volatility"),
        ("Theta", "Daily time decay"),
        ("Rho",   "Price change per 1% move in rate"),
    ]:
        row_vals(ws1, r, [gname, greeks_data.get(gname, ""), gdesc]); r += 1

    # ── Sheet 2: Price Curve Data ─────────────────────────
    chart_data = result.get("chart", {})
    if chart_data:
        ws2 = wb.create_sheet("Price Curve Data")
        for ci, (col, w) in enumerate([("A",12),("B",14),("C",14),("D",14)], 0):
            ws2.column_dimensions[col].width = w
        hdr(ws2, 1, 4, "Option Price Curve Data")
        for ci, h in enumerate(["Underlying S","BSM Price","Intrinsic Value","Time Value"], 1):
            ws2.cell(row=2, column=ci, value=h).font = BOLD
        S_vals = chart_data.get("S_values", [])
        prices = chart_data.get("prices",   [])
        intr   = chart_data.get("intrinsic",[])
        for i, sv in enumerate(S_vals):
            p  = prices[i] if i < len(prices) else ""
            iv = intr[i]   if i < len(intr)   else ""
            tv = round(p - iv, 4) if isinstance(p, (int,float)) and isinstance(iv,(int,float)) else ""
            ws2.append([sv, p, iv, tv])

    # ── Sheet 3: Greeks Curves Data ───────────────────────
    gc = chart_data.get("greek_curves", {})
    if gc:
        ws3 = wb.create_sheet("Greeks Curves Data")
        headers = ["Underlying S","Delta","Gamma","Vega","Theta","Rho"]
        for ci, h in enumerate(headers, 1):
            col = openpyxl.utils.get_column_letter(ci)
            ws3.column_dimensions[col].width = 14
            ws3.cell(row=1, column=ci, value=h).font = BOLD
        for i, sv in enumerate(S_vals):
            row_data = [sv] + [gc[g][i] if i < len(gc.get(g, [])) else ""
                               for g in ["Delta","Gamma","Vega","Theta","Rho"]]
            ws3.append(row_data)

    # ── Sheet 4: Portfolio ────────────────────────────────
    positions = portfolio.get("positions", [])
    totals    = portfolio.get("totals",    {})
    if positions:
        ws4 = wb.create_sheet("Portfolio")
        ph  = ["#","Label","Type","S","K","T(yr)","σ","Qty","Entry",
               "BSM Price","MTM","P&L","Δ×Qty","Γ×Qty","V×Qty","Θ×Qty","ρ×Qty","Margin"]
        for ci, h in enumerate(ph, 1):
            col = openpyxl.utils.get_column_letter(ci)
            ws4.column_dimensions[col].width = 12
            ws4.cell(row=1, column=ci, value=h).font = BOLD
        for pi, pos in enumerate(positions, 1):
            wg = pos.get("w_greeks", {})
            ws4.append([
                pi, pos.get("label",""),
                "Call" if pos.get("flag")=="c" else "Put",
                pos.get("S",""), pos.get("K",""), pos.get("T",""),
                pos.get("sigma",""), pos.get("qty",""), pos.get("entry_price",""),
                pos.get("bsm_price",""), pos.get("mtm",""), pos.get("pnl",""),
                wg.get("Delta",""), wg.get("Gamma",""), wg.get("Vega",""),
                wg.get("Theta",""), wg.get("Rho",""), pos.get("margin",""),
            ])
        # Total row
        tr = len(positions) + 2
        for ci, val in enumerate(
            ["","TOTAL","","","","","","","","",
             totals.get("mtm",""),   totals.get("pnl",""),
             totals.get("Delta",""), totals.get("Gamma",""),
             totals.get("Vega",""),  totals.get("Theta",""),
             totals.get("Rho",""),   totals.get("margin","")], 1
        ):
            c = ws4.cell(row=tr, column=ci, value=val)
            c.font = Font(bold=True, color="4E9EFF")
            c.fill = SUB_FILL

    bio = BytesIO()
    wb.save(bio)
    bio.seek(0)
    return send_file(
        bio,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="option_pricing_report.xlsx",
    )


# ─────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Option Pricing Tool -- Web Application")
    print("  Open in browser: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)
