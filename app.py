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
    POST /api/smile_presets     : Auto-fill smile table from BSM prices
    POST /api/mc_distribution   : Monte Carlo payoff distribution histogram
    POST /api/sensitivity       : Sensitivity / stress test matrix
    POST /api/snowball          : Snowball autocallable MC pricing
    POST /api/shark_fin         : Shark fin barrier option (UOC/DOP/UIC/DIC) pricing
    POST /api/forward           : OTC equity forward pricing
    POST /api/irs               : Interest rate swap NPV + cash flow schedule
    POST /api/snowball_risk     : Snowball Greeks from the PDE solution surface
    POST /api/snowball_risk_crosscheck : PDE vs Monte Carlo risk, with a verdict
    POST /api/cn_fdm            : CN-FDM structured product reprice (snowball/shark_fin)
    POST /api/vol_surface/fit   : Fit an SSVI surface to a real option chain
    POST /api/vol_surface/query : Sample the fitted surface on a (K, T) grid
    GET  /api/vol_surface/snapshots : List the offline chains available

Usage:
    python app.py
    Then open: http://127.0.0.1:5000

Dependencies:
    pip install flask py_vollib numpy pandas matplotlib openpyxl scipy

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
from cn_fdm import CN_FDM

# Corrected snowball engines. The legacy cn_fdm.CN_FDM is still imported above
# for the shark-fin path, which has not been migrated yet.
from otcpricer.core.market import EquityMarket
from otcpricer.core.types import KIStrikeMode
from otcpricer.core.exceptions import OtcPricerError
from otcpricer.engines.mc import MonteCarloConfig, price_snowball_mc
from otcpricer.engines.pde import PDEConfig, price_snowball_pde
from otcpricer.engines.risk import (
    SnowballRisk,
    compare_snowball_risk,
    snowball_risk_mc,
    snowball_risk_pde,
)
from otcpricer.products.snowball import SnowballTerms

# Volatility surface (Phase 2a). Offline by default: core.config ships with
# source="snapshot", so these endpoints answer from data/snapshots unless a
# request explicitly asks for the live feed.
from otcpricer.marketdata.contracts import SUPPORTED_UNDERLYINGS
from otcpricer.marketdata.forward import DEFAULT_RATE
from otcpricer.marketdata.snapshots import list_snapshots
from otcpricer.models.surface import fit_surface
from otcpricer.reports.surface_report import surface_report

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

    return jsonify({
        "results": points,
        "note": (
            "This inverts hand-entered prices one expiry at a time and fits "
            "nothing. For a surface built from a real option chain -- forwards "
            "implied from parity, quotes screened, SSVI calibrated under the "
            "no-arbitrage constraints -- use /api/vol_surface/fit."
        ),
    })


# ─────────────────────────────────────────────────────────
# API: Volatility Surface (SSVI, real option chains)
# ─────────────────────────────────────────────────────────

#: Fitting a surface takes a few hundred milliseconds, and the query endpoint
#: gets called repeatedly while a user drags a chart around. Keyed on
#: everything that changes the answer, so a different source or rate cannot be
#: served a stale fit.
_SURFACE_CACHE = {}


def _cached_surface(underlying, source_kind, snapshot, rate):
    key = (underlying, source_kind, snapshot, round(float(rate), 10))
    if key not in _SURFACE_CACHE:
        _SURFACE_CACHE[key] = fit_surface(
            underlying, source_kind=source_kind, snapshot=snapshot, rate=rate
        )
    return _SURFACE_CACHE[key]


def _surface_request(data):
    """Pull the common surface arguments out of a request body."""
    underlying = str(data.get("underlying", "510050")).strip()
    if underlying not in SUPPORTED_UNDERLYINGS:
        raise ValueError(
            f"underlying {underlying!r} is not supported; available: "
            f"{', '.join(sorted(SUPPORTED_UNDERLYINGS))}"
        )
    source_kind = data.get("source")
    snapshot = data.get("snapshot")
    rate = float(data.get("r", data.get("rate", DEFAULT_RATE)))
    return underlying, source_kind, snapshot, rate


@app.route("/api/vol_surface/fit", methods=["POST"])
def vol_surface_fit():
    """
    Fit an SSVI volatility surface to a real listed option chain.

    Runs the whole chain in one call: fetch quotes, imply a forward per expiry
    from put-call parity, solve and screen implied volatilities, calibrate SSVI
    with the no-arbitrage conditions imposed as constraints, then verify the
    result independently on a dense grid.

    The response carries the evidence, not just the parameters. A surface built
    from non-synchronous last-trade prices with no bid/ask is an estimate, so
    the fit error per expiry, the parity residuals, every discarded quote with
    its reason, and the arbitrage margins all travel with it.

    Request JSON: {underlying, source, snapshot, r, report}
      underlying  "510050" or "510300" (default "510050")
      source      "snapshot" (default), "akshare" for the live feed, or "csv"
      snapshot    snapshot name, when replaying a specific date
      r           continuously compounded rate. Only the forward uses it, and
                  barely: 100bp moves the forward by 0.003%
      report      when true, also return the full validation report as text

    Response JSON: the surface, its inputs, its fit quality and its arbitrage
    conclusion, plus whether a Dupire local volatility can be derived from it.
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        underlying, source_kind, snapshot, rate = _surface_request(data)
        want_report = bool(data.get("report", False))
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    try:
        fitted = _cached_surface(underlying, source_kind, snapshot, rate)
    except OtcPricerError as e:
        return jsonify({"error": str(e)}), 400

    payload = fitted.as_dict()
    if want_report:
        payload["report"] = surface_report(fitted)
    return jsonify(payload)


@app.route("/api/vol_surface/query", methods=["POST"])
def vol_surface_query():
    """
    Sample a fitted surface on a (strike, maturity) grid, for a heat map.

    The default grid spans the range the surface is actually used over rather
    than only the range it was fitted on: out to three years, because the
    structured products priced off it run one to three while the listed options
    stop at roughly five months. The arbitrage conditions are returned
    alongside, since the extrapolated region is where they are least
    guaranteed.

    Request JSON: {underlying, source, snapshot, r, moneyness, strikes,
                   maturities, n_moneyness, n_maturities, t_max}
      moneyness    explicit log-moneyness values, or
      strikes      explicit strikes, converted with the nearest expiry's
                   forward, or neither, for an automatic grid
      maturities   explicit maturities in years, or neither for an auto grid
      t_min        start of the auto maturity grid; defaults to one week, for
                   the reason given at the grid construction below

    Response JSON: {moneyness, strikes, maturities, implied_vol, total_variance,
                    butterfly, dw_dT}, where implied_vol[i][j] sits at
                    maturities[i] and moneyness[j].
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        underlying, source_kind, snapshot, rate = _surface_request(data)
        n_moneyness = max(3, min(int(data.get("n_moneyness", 41)), 401))
        n_maturities = max(2, min(int(data.get("n_maturities", 25)), 201))
        t_max = float(data.get("t_max", 3.0))
        if t_max <= 0:
            raise ValueError("t_max must be positive")
        t_min = data.get("t_min")
        t_min = None if t_min is None else float(t_min)
        if t_min is not None and not 0.0 < t_min < t_max:
            raise ValueError("t_min must be positive and below t_max")
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    try:
        fitted = _cached_surface(underlying, source_kind, snapshot, rate)
    except OtcPricerError as e:
        return jsonify({"error": str(e)}), 400

    usable = fitted.surface_input.usable_slices
    reference_forward = float(usable[0].forward.forward)

    try:
        if data.get("moneyness"):
            moneyness = np.asarray([float(k) for k in data["moneyness"]], dtype=float)
        elif data.get("strikes"):
            strikes = np.asarray([float(k) for k in data["strikes"]], dtype=float)
            if np.any(strikes <= 0):
                raise ValueError("strikes must be positive")
            moneyness = np.log(strikes / reference_forward)
        else:
            # Three standard deviations of the longest fitted expiry, the same
            # reach the quote screen admits.
            longest = float(usable[-1].year_fraction)
            atm = float(fitted.surface.implied_vol(0.0, longest))
            reach = max(3.0 * atm * np.sqrt(longest), 0.2)
            moneyness = np.linspace(-reach, reach, n_moneyness)

        if data.get("maturities"):
            maturities = np.asarray([float(t) for t in data["maturities"]], dtype=float)
            if np.any(maturities <= 0):
                raise ValueError("maturities must be positive")
        else:
            # The moneyness reach is set from the longest expiry, so starting
            # the maturity grid at the shortest one puts the near rows many
            # standard deviations out: 510050's one-day expiry reaches 18, and
            # SSVI dutifully returns 84% vol there. That is not a wrong number,
            # it is a meaningless one, and on a heat map it owns the colour
            # scale. So the default floor is a week, which an explicit t_min or
            # an explicit maturities list overrides.
            floor = t_min if t_min is not None else 1.0 / 52.0
            start = max(float(usable[0].year_fraction), min(floor, t_max * 0.5))
            maturities = np.linspace(start, t_max, n_maturities)
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid grid: {e}"}), 400

    surface = fitted.surface
    implied, variance, butterfly, calendar = [], [], [], []
    for maturity in maturities:
        implied.append([round(float(v), 6) for v in surface.implied_vol(moneyness, maturity)])
        variance.append([round(float(v), 8) for v in surface.total_variance(moneyness, maturity)])
        butterfly.append([round(float(v), 6) for v in surface.butterfly(moneyness, maturity)])
        calendar.append([round(float(v), 8) for v in surface.dw_dT(moneyness, maturity)])

    return jsonify({
        "underlying": underlying,
        "as_of": fitted.as_of,
        "source": fitted.source,
        "reference_forward": round(reference_forward, 6),
        "moneyness": [round(float(k), 6) for k in moneyness],
        "strikes": [round(float(reference_forward * np.exp(k)), 6) for k in moneyness],
        "maturities": [round(float(t), 6) for t in maturities],
        "implied_vol": implied,
        "total_variance": variance,
        "butterfly": butterfly,
        "dw_dT": calendar,
        "quoted_maturities": [round(float(sl.year_fraction), 6) for sl in usable],
        "arbitrage": fitted.arbitrage.as_dict(),
        "note": (
            "implied_vol[i][j] is at maturities[i], moneyness[j]. Maturities "
            "past the last quoted expiry are extrapolated: total variance "
            "continues at the last quoted secant rate."
        ),
    })


@app.route("/api/vol_surface/snapshots", methods=["GET", "POST"])
def vol_surface_snapshots():
    """
    List the offline option chains available, per underlying.

    Snapshots are the default data source, so this is what /api/vol_surface/fit
    will read unless a request asks for the live feed.
    """
    from otcpricer.core.config import get_settings

    settings = get_settings()
    directory = settings.data.snapshot_dir
    return jsonify({
        "snapshot_dir": str(directory),
        "default_source": settings.data.source,
        "underlyings": {
            code: {
                "name": underlying.name,
                "feed_symbol": underlying.feed_symbol,
                "snapshots": list_snapshots(directory, code),
            }
            for code, underlying in sorted(SUPPORTED_UNDERLYINGS.items())
        },
    })


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
# API: Snowball (Autocallable) Pricing — Monte Carlo
# ─────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────
# Snowball request parsing (shared by /api/snowball and /api/cn_fdm)
# ─────────────────────────────────────────────────────────

def _parse_snowball_request(d, spot_key="S"):
    """Build (SnowballTerms, EquityMarket) from a request payload.

    Two spot levels are read, not one:

    ``S0`` -- the **initial fixing**, which sets the barrier levels and the
    knock-in put strike for the life of the trade.

    ``S`` -- the **current spot**, where the note is valued.

    Either key alone is accepted and used for both, which is correct for a
    trade at inception and keeps every existing caller working: /api/snowball
    sends ``S``, /api/cn_fdm sends ``S0``. Supplying both values a *seasoned*
    trade: barriers stay where they were struck while the market has moved.
    That separation is also what makes delta a hedge ratio rather than the
    derivative of a re-struck contract.

    Accepts the legacy field names so the existing frontend keeps working, and
    exposes the parameters the corrected model added:

    ``lockup_months``
        Months during which knock-out is not observed. Defaults to 3, the
        A-share standard. The legacy endpoints had no such concept and so
        observed knock-out from month one.
    ``ki_obs_freq``
        Knock-in observations per year. Defaults to 252 (each trading day).
        The legacy /api/snowball checked knock-in only on the *monthly*
        knock-out dates, which materially understated knock-in risk.
    ``ki_strike_mode``
        ``"s0"`` (default, market convention) or ``"ki"``.
    ``basis_pa``
        Annualised index futures discount. Enters the drift, not the discount
        factor.
    """
    if spot_key in d:
        spot = float(d[spot_key])
    elif "S" in d:
        spot = float(d["S"])
    elif "S0" in d:
        spot = float(d["S0"])
    else:
        raise KeyError("S")
    # The fixing defaults to spot, which is the at-inception case.
    fixing = float(d.get("S0", spot))

    terms = SnowballTerms(
        S0=fixing,
        T=float(d["T"]),
        ko_pct=float(d.get("KO_pct", 1.03)),
        ki_pct=float(d.get("KI_pct", 0.75)),
        coupon_pa=float(d.get("coupon_pa", 0.15)),
        obs_freq=int(d.get("obs_freq", 12)),
        ki_obs_freq=int(d.get("ki_obs_freq", 252)),
        lockup_months=int(d.get("lockup_months", 3)),
        ki_strike_mode=KIStrikeMode(str(d.get("ki_strike_mode", "s0")).lower()),
    )
    market = EquityMarket(
        S=spot,
        r=float(d["r"]),
        sigma=float(d["sigma"]),
        q=float(d.get("q", 0.0)),
        basis_pa=float(d.get("basis_pa", 0.0)),
    )
    return terms, market


@app.route("/api/snowball", methods=["POST"])
def snowball():
    """
    Price a Snowball (Autocallable) note by path Monte Carlo.

    Knock-out is observed monthly; on a knock-out the note terminates early and
    pays the coupon accrued *to that date*. Knock-in is observed every trading
    day. If the note knocks in and never knocks out, the holder absorbs the
    fall as a short put struck at the initial level. If neither happens, the
    full coupon is paid.

    This now calls otcpricer.engines.mc rather than an inline simulation. Three
    corrections are visible in the output relative to the previous version:

    * Knock-in is monitored daily rather than only on the monthly knock-out
      dates, which the inline version did -- that understated knock-in risk.
    * A three-month lock-up applies by default (``lockup_months``).
    * The futures basis can be supplied (``basis_pa``).

    Request JSON:
        S, sigma, r, q    — market parameters
        T                 — maturity in years
        KO_pct, KI_pct    — barriers as a fraction of S0
        coupon_pa         — annual coupon rate
        obs_freq          — knock-out observations per year (12 = monthly)
        ki_obs_freq       — knock-in observations per year (252 = daily)
        lockup_months     — knock-out lock-up, default 3
        ki_strike_mode    — "s0" (default) or "ki"
        basis_pa          — annualised futures discount, default 0
        spread            — bid/ask half-spread as a fraction of mid
        n_paths           — Monte Carlo paths
    """
    try:
        d = request.get_json(force=True)
        terms, market = _parse_snowball_request(d)
        spread = float(d.get("spread", 0.02))
        n_paths = min(int(d.get("n_paths", 50000)), 500000)
    except (KeyError, TypeError, ValueError, OtcPricerError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    try:
        result = price_snowball_mc(
            terms, market, MonteCarloConfig(paths=n_paths, seed=42)
        )
    except OtcPricerError as e:
        return jsonify({"error": str(e)}), 400

    mid = round(result.price, 6)
    return jsonify({
        # Legacy keys, unchanged so the existing frontend keeps working.
        "mid": mid,
        "bid": round(mid - abs(mid) * spread, 6),
        "ask": round(mid + abs(mid) * spread, 6),
        "std_err": round(result.std_error, 6),
        "ci95": round(result.ci95, 6),
        "ko_prob": round(result.ko_prob, 4),
        "ki_prob": round(result.ki_prob, 4),
        "n_paths": result.paths,
        # New diagnostics the corrected engine can report.
        "ki_no_ko_prob": round(result.ki_no_ko_prob, 4),
        "expected_ko_time": round(result.expected_ko_time, 4)
                            if result.expected_ko_time == result.expected_ko_time else None,
        "params": {
            "KO": terms.ko_barrier,
            "KI": terms.ki_barrier,
            "coupon_pa": terms.coupon_pa,
            "T": terms.T,
            "freq": terms.obs_freq,
            "ki_obs_freq": terms.ki_obs_freq,
            "lockup_months": terms.lockup_months,
            "ki_strike_mode": str(terms.ki_strike_mode),
            "ki_put_strike": terms.ki_put_strike,
            "n_ko_observations": int(terms.ko_observation_times().size),
            "basis_pa": market.basis_pa,
        },
    })


@app.route("/api/snowball_pde", methods=["POST"])
def snowball_pde():
    """
    Price the same snowball by Crank-Nicolson finite differences.

    An independent valuation: a different discretisation of a different
    formulation, so agreement with /api/snowball is genuine evidence rather
    than a restatement. The two now agree to within roughly two Monte Carlo
    confidence widths across the parameter range; they previously differed by
    18% while being described as cross-validation.

    Accepts the same payload as /api/snowball, plus optional ``space_nodes``
    and ``steps_per_year``.
    """
    try:
        d = request.get_json(force=True)
        terms, market = _parse_snowball_request(d)
        config = PDEConfig(
            space_nodes=int(d.get("space_nodes", 1600)),
            steps_per_year=int(d.get("steps_per_year", 1008)),
        )
    except (KeyError, TypeError, ValueError, OtcPricerError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    try:
        result = price_snowball_pde(terms, market, config)
    except OtcPricerError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({
        "price": round(result.price, 8),
        "method": "Crank-Nicolson FDM (barrier- and observation-aligned grid)",
        "space_nodes": result.space_nodes,
        "time_steps": result.time_steps,
        "params": {
            "KO": terms.ko_barrier,
            "KI": terms.ki_barrier,
            "lockup_months": terms.lockup_months,
            "ki_strike_mode": str(terms.ki_strike_mode),
            "basis_pa": market.basis_pa,
        },
    })


@app.route("/api/snowball_crosscheck", methods=["POST"])
def snowball_crosscheck():
    """
    Run both snowball engines and report whether they agree.

    This is the model-validation view: two independent valuations, their
    difference, and that difference measured against a real error budget
    rather than a guess.

    Both error terms are *measured*, not assumed:

    * The Monte Carlo contributes its own confidence interval.
    * The PDE is first-order accurate on a finite grid, so it carries a genuine
      bias. Running it at two resolutions gives the step-to-step change; with
      an error series that is approximately geometric with ratio 2, the
      remaining error is of the same size, so the change itself bounds it. The
      two solutions also yield a Richardson-extrapolated price with that
      leading term removed.

    Reporting the budget breakdown matters as much as the verdict: a pass
    carried entirely by a wide Monte Carlo interval is not the same evidence as
    a pass with both engines tightly converged, and the caller can see which
    they have.
    """
    try:
        d = request.get_json(force=True)
        terms, market = _parse_snowball_request(d)
        n_paths = min(int(d.get("n_paths", 200000)), 500000)
    except (KeyError, TypeError, ValueError, OtcPricerError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    try:
        mc = price_snowball_mc(terms, market, MonteCarloConfig(paths=n_paths, seed=42))
        coarse = price_snowball_pde(
            terms, market, PDEConfig(space_nodes=800, steps_per_year=504)
        )
        fine = price_snowball_pde(
            terms, market, PDEConfig(space_nodes=1600, steps_per_year=1008)
        )
    except OtcPricerError as e:
        return jsonify({"error": str(e)}), 400

    # First-order Richardson extrapolation, and the step-to-step change as the
    # bound on what remains.
    pde_change = abs(fine.price - coarse.price)
    pde_extrapolated = 2.0 * fine.price - coarse.price

    gap = abs(mc.price - pde_extrapolated)
    budget = pde_change + 2.0 * mc.ci95

    return jsonify({
        "mc_price": round(mc.price, 8),
        "mc_ci95": round(mc.ci95, 8),
        "pde_price": round(fine.price, 8),
        "pde_price_coarse": round(coarse.price, 8),
        "pde_extrapolated": round(pde_extrapolated, 8),
        "gap": round(gap, 8),
        "gap_relative": round(gap / abs(mc.price), 6) if mc.price else None,
        "budget": round(budget, 8),
        "agree": bool(gap < budget),
        "budget_breakdown": {
            "mc_ci95_times_two": round(2.0 * mc.ci95, 8),
            "pde_grid_change": round(pde_change, 8),
        },
        "grids": {
            "coarse": {"space_nodes": coarse.space_nodes, "time_steps": coarse.time_steps},
            "fine": {"space_nodes": fine.space_nodes, "time_steps": fine.time_steps},
        },
    })


# ─────────────────────────────────────────────────────────
# API: Snowball Greeks
# ─────────────────────────────────────────────────────────

@app.route("/api/snowball_risk", methods=["POST"])
def snowball_risk():
    """
    Snowball Greeks: Delta, Gamma, Vega, Theta, Rho.

    Computed by differencing the Crank-Nicolson solution surface. The solver
    already evaluates the value at every grid node on its way to the price, so
    delta and gamma are read straight off it -- no bumping, and none of the
    sampling noise that makes a simulated gamma unusable on a discontinuous
    payoff. Vega and rho are central differences over re-solves; theta comes
    from two adjacent time steps of the same solve, which keeps the observation
    schedule fixed.

    Units follow the vanilla Greeks: delta per index point, gamma per point
    squared, vega per percentage point of volatility, theta per calendar day,
    rho per percentage point of rate. All as a fraction of notional.

    Accepts the /api/snowball payload, plus:
        knocked_in     — true for a seasoned note that has already breached its
                         knock-in barrier; it must be risked off the other
                         surface, and the two differ materially
        space_nodes, steps_per_year — discretisation overrides

    Note that market spot and the initial fixing are separate: ``S`` is the
    current level, while the barriers stay pinned to where the trade was struck.
    That is what makes delta a hedge ratio rather than the derivative of a
    re-struck contract.
    """
    try:
        d = request.get_json(force=True)
        terms, market = _parse_snowball_request(d)
        knocked_in = bool(d.get("knocked_in", False))
        config = PDEConfig(
            space_nodes=int(d.get("space_nodes", 1600)),
            steps_per_year=int(d.get("steps_per_year", 1008)),
        )
    except (KeyError, TypeError, ValueError, OtcPricerError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    try:
        risk = snowball_risk_pde(terms, market, config, knocked_in=knocked_in)
    except OtcPricerError as e:
        return jsonify({"error": str(e)}), 400

    payload = risk.as_dict()
    payload["knocked_in"] = knocked_in
    payload["units"] = {
        "delta": "per index point",
        "gamma": "per index point squared",
        "vega": "per 1 percentage point of volatility",
        "theta": "per calendar day",
        "rho": "per 1 percentage point of rate",
        "basis": "fraction of notional",
    }
    return jsonify(payload)


@app.route("/api/snowball_risk_crosscheck", methods=["POST"])
def snowball_risk_crosscheck():
    """
    Run both risk methods and report whether they agree, Greek by Greek.

    The PDE is the production method; Monte Carlo resimulation with common
    random numbers is an independent check -- a different formulation, sampled
    rather than solved. Each Greek gets a budget built from the simulation's own
    measured standard error rather than a chosen tolerance, and the breakdown is
    returned alongside the verdict: a pass carried by a wide sampling error is
    weaker evidence than one with both methods tight, and the caller should be
    able to tell them apart.

    Gamma is judged against its sampling error rather than on sign. The
    simulated estimator divides a second difference of a discontinuous payoff by
    h^2, so where the true gamma is small its error can exceed the value and the
    sign is undetermined.

    This replaces the previous "three parallel Greeks" output, which reported
    three sets that disagreed in sign and so could not all be right.
    """
    try:
        d = request.get_json(force=True)
        terms, market = _parse_snowball_request(d)
        mc_paths = min(int(d.get("n_paths", 200000)), 500000)
    except (KeyError, TypeError, ValueError, OtcPricerError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    try:
        comparison = compare_snowball_risk(
            terms, market, mc_config=MonteCarloConfig(paths=mc_paths, seed=42)
        )
    except OtcPricerError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(comparison.as_dict())


# ─────────────────────────────────────────────────────────
# API: Shark Fin (Barrier Option) Pricing — Closed Form
# ─────────────────────────────────────────────────────────

@app.route("/api/shark_fin", methods=["POST"])
def shark_fin():
    """
    Price barrier options (Up-and-Out Call / Down-and-Out Put) analytically.
    Uses the Rubinstein-Reiner / Haug (1998) closed-form formula.

    For Up-and-Out Call (UOC): η = -1, φ = 1, requires H > K
    For Down-and-Out Put (DOP): η = +1, φ = -1, requires H < K

    Correct formula: price = A - B - C + D  (with η-signed N() arguments)

    Request JSON:
        flag    — 'c' (Up-and-Out Call) or 'p' (Down-and-Out Put)
        S, K, T, r, sigma, q — standard BSM params
        H       — barrier level
        rebate  — cash rebate paid if barrier is hit (default 0)
        spread  — bid-ask half-spread fraction (default 0.02)
    """
    try:
        d       = request.get_json(force=True)
        flag    = str(d.get("flag", "c")).lower()
        # Normalize frontend flag values ('uo','ui','do','di') to internal 'c'/'p' convention
        _is_in_barrier = flag in ('ui', 'di')
        if flag in ('uo', 'ui'):
            flag = 'c'
        elif flag in ('do', 'di'):
            flag = 'p'
        S       = float(d["S"])
        K       = float(d["K"])
        T       = float(d["T"])
        r       = float(d["r"])
        sigma   = float(d["sigma"])
        q       = float(d.get("q", 0.0))
        H       = float(d["H"])
        rebate  = float(d.get("rebate", 0.0))
        spread  = float(d.get("spread", 0.02))
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    if T <= 0 or sigma <= 0:
        return jsonify({"error": "T and sigma must be positive"}), 400

    from scipy.stats import norm

    # ── Haug (1998) closed-form for continuous barrier options ──────────────
    phi = 1.0 if flag == 'c' else -1.0   # +1 call, -1 put
    eta = -1.0 if flag == 'c' else 1.0   # -1 up barrier, +1 down barrier

    if flag == 'c' and H <= S:
        return jsonify({"error": "For Call barrier option, barrier H must be above current spot S"}), 400
    if flag == 'p' and H >= S:
        return jsonify({"error": "For Put barrier option, barrier H must be below current spot S"}), 400
    if flag == 'c' and H <= K:
        return jsonify({"error": "For Call barrier option, H must be > K (otherwise always knocked out)"}), 400
    if flag == 'p' and H >= K:
        return jsonify({"error": "For Put barrier option, H must be < K (otherwise always knocked out)"}), 400

    try:
        sT  = sigma * np.sqrt(T)
        # λ = (r - q + σ²/2) / σ²
        lam = (r - q + 0.5 * sigma**2) / sigma**2
        # μ = λ - 1 = (r - q - σ²/2) / σ²
        mu  = lam - 1.0

        eqT = np.exp(-q * T)
        erT = np.exp(-r * T)
        hS  = H / S

        # BSM argument definitions (Haug 1998, p.66; validated against CN-FDM PDE)
        x1 = np.log(S / K) / sT + lam * sT
        x2 = np.log(S / H) / sT + lam * sT
        y1 = np.log(H**2 / (S * K)) / sT + lam * sT
        y2 = np.log(H / S) / sT + lam * sT

        # ── Correct Haug (1998) formula using N(y), NOT N(η·y) ────────
        # The A terms use φ·x (call vs put sign flip)
        # The B/C/D reflection terms use y directly (barrier type encoded in formula)
        # Reference: verified against CN-FDM PDE (error < 0.01%)

        def _A_term(x):
            """Standard BSM term: φ·[S·e^(-qT)·N(φx) - K·e^(-rT)·N(φx - φσ√T)]"""
            return (phi * S * eqT * norm.cdf(phi * x)
                    - phi * K * erT * norm.cdf(phi * x - phi * sT))

        def _B_reflection(y):
            """Reflection term using N(y) directly (NOT N(η·y))"""
            return (phi * S * eqT * hS**(2 * lam) * norm.cdf(y)
                    - phi * K * erT * hS**(2 * lam - 2) * norm.cdf(y - sT))

        A = _A_term(x1)   # vanilla option price
        B = _A_term(x2)   # barrier-adjusted BSM term
        C = _B_reflection(y1)
        D = _B_reflection(y2)

        barrier_price_out = float(A - B - C + D)
        barrier_price = float(A - barrier_price_out) if _is_in_barrier else barrier_price_out

        # Rebate present value (correct formula using N(y) convention)
        if rebate > 0:
            z  = np.log(H / S) / sT + mu * sT
            r1 = hS ** (mu + lam)
            r2 = hS ** (mu - lam)
            rebate_pv = float(rebate * (
                r1 * norm.cdf(phi * z) +
                r2 * norm.cdf(phi * z - 2 * phi * lam * sT)
            ))
        else:
            rebate_pv = 0.0

        # Vanilla price for comparison
        vanilla = float(A)

    except Exception as e:
        return jsonify({"error": f"Pricing error: {e}"}), 500

    mid = round(max(0.0, barrier_price + rebate_pv), 6)
    bid = round(mid * (1 - spread), 6)
    ask = round(mid * (1 + spread), 6)

    return jsonify({
        "mid": mid, "bid": bid, "ask": ask,
        "vanilla_price": round(vanilla, 6),
        "barrier_discount": round(max(0.0, vanilla - barrier_price), 6),
        "rebate_pv": round(rebate_pv, 6),
        "params": {"H": H, "rebate": rebate, "flag": flag}
    })


# ─────────────────────────────────────────────────────────
# API: OTC Forward Pricing
# ─────────────────────────────────────────────────────────

@app.route("/api/forward", methods=["POST"])
def otc_forward():
    """
    Price an OTC Forward contract.  F = S * exp((r - q) * T)

    Request JSON:
        S, T, r, q  — standard params
        direction   — 'long' or 'short'
        notional    — contract notional (default 1)
        spread      — bid-ask half-spread as absolute points on F (default 0)
    """
    try:
        d          = request.get_json(force=True)
        S          = float(d["S"])
        T          = float(d["T"])
        r          = float(d["r"])
        q          = float(d.get("q", 0.0))
        direction  = str(d.get("direction", "long")).lower()
        notional   = float(d.get("notional", 1.0))
        spread_pts = float(d.get("spread", 0.0))
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    F             = S * np.exp((r - q) * T)
    cost_of_carry = S * (np.exp((r - q) * T) - 1.0)
    funding_cost  = S * (np.exp(r * T) - 1.0)
    div_income    = S * (np.exp(q * T) - 1.0)

    mid = round(float(F), 6)
    bid = round(mid - spread_pts, 6)
    ask = round(mid + spread_pts, 6)

    sign = 1 if direction == "long" else -1
    mtm_per_unit = sign * (F - S)  # theoretical P&L vs spot

    return jsonify({
        "forward_price": mid,
        "bid": bid, "ask": ask,
        "cost_of_carry":   round(float(cost_of_carry), 4),
        "funding_cost":    round(float(funding_cost), 4),
        "dividend_income": round(float(div_income), 4),
        "mtm_per_unit":    round(float(mtm_per_unit), 4),
        "notional_mtm":    round(float(mtm_per_unit * notional), 4),
        "params": {"S": S, "T": T, "r": r, "q": q,
                   "direction": direction, "notional": notional}
    })


# ─────────────────────────────────────────────────────────
# API: Interest Rate Swap (IRS) — Simplified DCF
# ─────────────────────────────────────────────────────────

@app.route("/api/irs", methods=["POST"])
def irs():
    """
    Price a plain vanilla Interest Rate Swap (IRS) using simplified flat-curve DCF.

    Pay-fixed / receive-float convention (NPV from perspective of fixed payer).
    NPV = PV(Float leg) - PV(Fixed leg)

    Request JSON:
        notional      — notional principal
        fixed_rate    — annual fixed rate (e.g. 0.035 = 3.5%)
        float_rate    — current floating rate / par rate (e.g. 0.025 = SHIBOR)
        T             — maturity in years
        freq          — payment frequency per year (1=annual, 2=semi, 4=quarterly)
        r_disc        — discount rate (defaults to float_rate)
        spread        — bid-ask half-spread as absolute NPV (default 0)
    """
    try:
        d           = request.get_json(force=True)
        notional    = float(d.get("notional", 1_000_000))
        fixed_rate  = float(d["fixed_rate"])
        float_rate  = float(d["float_rate"])
        T           = float(d["T"])
        freq        = int(d.get("freq", 4))
        r_disc      = float(d.get("r_disc", float_rate))
        spread      = float(d.get("spread", 0.0))
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    dt = 1.0 / freq
    periods = int(round(T * freq))
    if periods < 1:
        return jsonify({"error": "Too short: T * freq < 1"}), 400

    fixed_cf  = fixed_rate * dt * notional
    float_cf  = float_rate * dt * notional

    cashflows = []
    pv_fixed = pv_float = 0.0
    for i in range(1, periods + 1):
        t_i    = i * dt
        df     = np.exp(-r_disc * t_i)
        # Add principal repayment to last period
        extra  = notional if i == periods else 0.0
        fix_pmt = fixed_cf + extra
        flt_pmt = float_cf + extra
        pv_f   = fix_pmt * df
        pv_fl  = flt_pmt * df
        pv_fixed  += pv_f
        pv_float  += pv_fl
        cashflows.append({
            "period":     i,
            "t":          round(t_i, 4),
            "df":         round(float(df), 6),
            "fixed_pmt":  round(fix_pmt, 2),
            "float_pmt":  round(flt_pmt, 2),
            "pv_fixed":   round(pv_f, 2),
            "pv_float":   round(pv_fl, 2),
            "net_pmt":    round(flt_pmt - fix_pmt, 2),
        })

    npv  = round(float(pv_float - pv_fixed), 2)
    bid  = round(npv - spread, 2)
    ask  = round(npv + spread, 2)
    dv01 = round(float(notional * dt * np.exp(-r_disc * T) * 0.0001), 2)  # approx DV01

    return jsonify({
        "npv": npv, "bid": bid, "ask": ask,
        "pv_fixed":  round(float(pv_fixed), 2),
        "pv_float":  round(float(pv_float), 2),
        "dv01":      dv01,
        "cashflows": cashflows,
        "params": {
            "notional": notional, "fixed_rate": fixed_rate,
            "float_rate": float_rate, "T": T, "freq": freq
        }
    })






# ─────────────────────────────────────────────────────────
# API: CN-FDM 结构化产品定价（雪球 / 鲨鱼鳍）
# ─────────────────────────────────────────────────────────

@app.route("/api/cn_fdm", methods=["POST"])
def cn_fdm_price_api():
    """
    通用 Crank-Nicolson 有限差分法定价接口 + Glasserman&Broadie 三套 Greeks。
    支持 product_type = 'snowball' | 'shark_fin'

    请求体（JSON）：
        product_type (str)  : 'snowball' 或 'shark_fin'
        S0      (float): 标的当前价格
        T       (float): 期限（年）
        r       (float): 无风险利率（小数）
        sigma   (float): 年化波动率（小数）
        q       (float): 连续股息率（小数），默认 0
        KO_pct  (float): 敲出价 / S0，如 1.05
        KI_pct  (float): 敲入价 / S0，如 0.75（仅雪球）
        coupon_pa (float): 年化票息（小数），如 0.15
        obs_freq  (int) : 敲出观察频率（每年），默认 12
        ki_obs_freq(int): 敲入观察频率（每年），默认 252（日频）
        rebate    (float): 敲出补偿（鲨鱼鳍），默认 0
        K_strike  (float): 行权价（鲨鱼鳍），默认 = S0
        N       (int)  : 空间网格节点数，默认 400
        mc_paths  (int) : MC仿真路径数，默认 50000
        calc_greeks (bool): 是否计算三套希腊字母，默认 True

    返回：
        price, resimulation_fd, pathwise, likelihood, method, params
    """
    try:
        d            = request.get_json(force=True)
        product_type = str(d.get("product_type", "snowball")).lower()
        S0           = float(d["S0"])
        T            = float(d["T"])
        r            = float(d["r"])
        sigma        = float(d["sigma"])
        q            = float(d.get("q", 0.0))
        KO_pct       = float(d.get("KO_pct", 1.05))
        KI_pct       = float(d.get("KI_pct", 0.75))
        coupon_pa    = float(d.get("coupon_pa", 0.15))
        obs_freq     = int(d.get("obs_freq", 12))
        ki_obs_freq  = int(d.get("ki_obs_freq", 252))
        rebate       = float(d.get("rebate", 0.0))
        K_strike     = float(d.get("K_strike", S0))
        N            = min(int(d.get("N", 400)), 600)
        mc_paths     = min(int(d.get("mc_paths", 50000)), 200000)
        calc_g       = bool(d.get("calc_greeks", True))
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    if product_type not in ("snowball", "shark_fin"):
        return jsonify({"error": "product_type must be 'snowball' or 'shark_fin'"}), 400
    if T <= 0 or sigma <= 0 or S0 <= 0:
        return jsonify({"error": "S0, T and sigma must be positive"}), 400

    try:
        cn = CN_FDM(
            product_type = product_type,
            S0           = S0,
            T            = T,
            r            = r,
            sigma        = sigma,
            q            = q,
            KO_pct       = KO_pct,
            KI_pct       = KI_pct,
            coupon_pa    = coupon_pa,
            obs_freq     = obs_freq,
            ki_obs_freq  = ki_obs_freq,
            rebate       = rebate,
            K_strike     = K_strike,
            N            = N,
            M_per_year   = 252,
            mc_paths     = mc_paths,
            seed          = 42,
        )

        if calc_g and product_type == "snowball":
            # Snowball Greeks now come from the corrected engines. The legacy
            # three-method output reported Resimulation FD, Pathwise and
            # Likelihood Ratio deltas of -8.4e-05, +1.0e-05 and +9.0e-06 on the
            # reference trade against a correct +4.9e-05: all three wrong, and
            # disagreeing in sign, so at most one could have been right.
            #
            # The response keys are kept so the existing frontend does not
            # break. "pathwise" is deliberately null: the method requires a
            # Lipschitz payoff and a snowball's jumps at both barriers, so it
            # does not apply here. Reporting null with a reason is more useful
            # than a number that looks like the other two.
            risk_terms, risk_market = _parse_snowball_request(d)
            pde_risk = snowball_risk_pde(
                risk_terms,
                risk_market,
                PDEConfig(space_nodes=1600, steps_per_year=1008),
            )
            mc_risk = snowball_risk_mc(
                risk_terms, risk_market, MonteCarloConfig(paths=min(mc_paths, 200000), seed=42)
            )
            price = pde_risk.price
            greeks_out = {
                "pde_grid": pde_risk.greeks.as_dict(SnowballRisk.DECIMALS),
                "resimulation_fd": mc_risk.greeks.as_dict(SnowballRisk.DECIMALS),
                "pathwise": None,
                "likelihood": None,
                "notes": {
                    "pde_grid": "Production method: differencing the PDE solution surface.",
                    "resimulation_fd": (
                        "Monte Carlo with common random numbers, as an independent "
                        "check. Theta is null: bumping the tenor deletes a knock-out "
                        "observation, so a simulated theta would measure a contract "
                        "change rather than time decay."
                    ),
                    "pathwise": (
                        "Not applicable. Pathwise differentiation needs the payoff to "
                        "be Lipschitz in the parameter so the derivative and the "
                        "expectation can be exchanged; a snowball's jumps to coupon*t "
                        "on a knock-out and switches legs on a knock-in."
                    ),
                    "likelihood": (
                        "Withdrawn. The previous implementation took its delta score "
                        "from the terminal distribution for a path-dependent payoff, "
                        "carried a sign error in the volatility score, and omitted "
                        "rho's discount-factor term."
                    ),
                },
            }
        elif calc_g:
            # Shark fin still runs on the legacy engine; it has not been
            # through the review the snowball has, and its Greeks carry the
            # same caveats they always did.
            full_result = cn.greeks()
            price = full_result["price"]
            greeks_out = {
                "resimulation_fd": full_result["resimulation_fd"],
                "pathwise": full_result["pathwise"],
                "likelihood": full_result["likelihood"],
                "notes": {
                    "engine": "Legacy cn_fdm.py -- not yet migrated or validated.",
                },
            }
        else:
            price = cn.price()
            greeks_out = {}

    except Exception as e:
        return jsonify({"error": f"CN-FDM solver error: {e}"}), 500

    # ── Haug 解析价格（鲨鱼鳍：与 CN-FDM PDE 交叉验证）──
    haug_price = None
    if product_type == "shark_fin":
        try:
            from scipy.stats import norm
            sT_val  = sigma * np.sqrt(T)
            lam_val = (r - q + 0.5 * sigma**2) / sigma**2
            eqT_val = np.exp(-q * T)
            erT_val = np.exp(-r * T)
            hS_val  = KO_pct
            x1v = np.log(S0 / K_strike) / sT_val + lam_val * sT_val
            x2v = np.log(S0 / (S0 * KO_pct)) / sT_val + lam_val * sT_val
            y1v = np.log((S0 * KO_pct)**2 / (S0 * K_strike)) / sT_val + lam_val * sT_val
            y2v = np.log(S0 * KO_pct / S0) / sT_val + lam_val * sT_val
            def _a(x): return S0*eqT_val*norm.cdf(x) - K_strike*erT_val*norm.cdf(x - sT_val)
            def _b(y): return S0*eqT_val*hS_val**(2*lam_val)*norm.cdf(y) - K_strike*erT_val*hS_val**(2*lam_val-2)*norm.cdf(y - sT_val)
            haug_price = round(float(_a(x1v) - _a(x2v) - _b(y1v) + _b(y2v)), 6)
        except Exception:
            haug_price = None

    return jsonify({
        "price":           round(float(price), 6),
        "haug_price":      haug_price,
        "pde_grid":        greeks_out.get("pde_grid"),
        "resimulation_fd": greeks_out.get("resimulation_fd", {}),
        "pathwise":        greeks_out.get("pathwise"),
        "likelihood":      greeks_out.get("likelihood"),
        "greek_notes":     greeks_out.get("notes"),
        "method":          "Crank-Nicolson FDM + Glasserman&Broadie MC Greeks",
        "product_type":    product_type,
        "params": {
            "S0": S0, "T": T, "r": r, "sigma": sigma, "q": q,
            "KO": round(S0 * KO_pct, 4),
            "KI": round(S0 * KI_pct, 4) if product_type == "snowball" else None,
            "KO_pct": KO_pct, "KI_pct": KI_pct,
            "coupon_pa": coupon_pa, "obs_freq": obs_freq,
            "ki_obs_freq": ki_obs_freq,
            "rebate": rebate, "K_strike": K_strike,
            "mc_paths": mc_paths,
        }
    })


if __name__ == "__main__":
    print("=" * 60)
    print("  Option Pricing Tool -- Web Application")
    print("  Open in browser: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)
