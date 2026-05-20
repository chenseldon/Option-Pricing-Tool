"""
app.py - Flask Web Application Backend

Provides REST API endpoints for the option pricing web interface.
Serves the single-page HTML frontend and handles all calculation requests.

Endpoints:
    GET  /              : Serve the web interface (index.html)
    POST /api/calculate : BSM pricing + Monte Carlo + all Greeks + chart data
    POST /api/iv        : Implied volatility solver

Usage:
    python app.py
    Then open: http://127.0.0.1:5000

Dependencies:
    pip install flask py_vollib numpy pandas matplotlib

Author: GitHub Portfolio Project
"""

import os
import sys
import numpy as np
from flask import Flask, render_template, request, jsonify

# Ensure sibling modules (bs_pricing, greeks_calc) are importable
sys.path.insert(0, os.path.dirname(__file__))

from bs_pricing import bs_price, monte_carlo_price, calc_implied_volatility, validate_params
from greeks_calc import calc_delta, calc_gamma, calc_vega, calc_theta, calc_rho

app = Flask(__name__)


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
    """
    Calculate BSM price, Monte Carlo price, all Greeks, and chart data.

    Request JSON:
        flag  (str)  : 'c' for Call, 'p' for Put
        S     (float): Underlying price
        K     (float): Strike price
        T     (float): Time to expiry (years)
        r     (float): Risk-free rate (decimal)
        sigma (float): Volatility (decimal)
        q     (float): Dividend yield (decimal), optional, default 0

    Response JSON:
        bsm_price    (float): BSM theoretical price
        mc_price     (float): Monte Carlo simulated price
        greeks       (dict) : Delta, Gamma, Vega, Theta, Rho at current S
        chart        (dict) : S_values, prices, intrinsic, greek_curves arrays
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
        return jsonify({"error": f"Invalid input parameters: {e}"}), 400

    if flag not in ("c", "p"):
        return jsonify({"error": "flag must be 'c' (Call) or 'p' (Put)"}), 400

    valid, msg = validate_params(S, K, T, r, sigma, q)
    if not valid:
        return jsonify({"error": msg}), 400

    # ── Core Pricing ──
    bsm_price_val = bs_price(flag, S, K, T, r, sigma, q)
    mc_price_val  = monte_carlo_price(flag, S, K, T, r, sigma, q, n_simulations=100000)

    # ── Greeks at current S ──
    greeks = {
        "Delta": round(float(calc_delta(flag, S, K, T, r, sigma, q)), 6),
        "Gamma": round(float(calc_gamma(flag, S, K, T, r, sigma, q)), 6),
        "Vega":  round(float(calc_vega(flag,  S, K, T, r, sigma, q)), 6),
        "Theta": round(float(calc_theta(flag, S, K, T, r, sigma, q)), 6),
        "Rho":   round(float(calc_rho(flag,   S, K, T, r, sigma, q)), 6),
    }

    # ── Chart Data: 100 evenly-spaced S values from 0.6K to 1.4K ──
    S_arr = np.linspace(0.6 * K, 1.4 * K, 100)

    prices_arr    = [round(float(bs_price(flag, s, K, T, r, sigma, q)), 4) for s in S_arr]
    intrinsic_arr = [
        round(float(max(s - K, 0) if flag == "c" else max(K - s, 0)), 4)
        for s in S_arr
    ]

    greek_curves = {
        "Delta": [round(float(calc_delta(flag, s, K, T, r, sigma, q)), 6) for s in S_arr],
        "Gamma": [round(float(calc_gamma(flag, s, K, T, r, sigma, q)), 6) for s in S_arr],
        "Vega":  [round(float(calc_vega(flag,  s, K, T, r, sigma, q)), 6) for s in S_arr],
        "Theta": [round(float(calc_theta(flag, s, K, T, r, sigma, q)), 6) for s in S_arr],
        "Rho":   [round(float(calc_rho(flag,   s, K, T, r, sigma, q)), 6) for s in S_arr],
    }

    return jsonify({
        "bsm_price": round(float(bsm_price_val), 4),
        "mc_price":  round(float(mc_price_val),  4),
        "greeks":    greeks,
        "chart": {
            "S_values":    [round(float(s), 2) for s in S_arr],
            "prices":      prices_arr,
            "intrinsic":   intrinsic_arr,
            "greek_curves": greek_curves,
        },
        "K": K,
    })


# ─────────────────────────────────────────────────────────
# API: Implied Volatility
# ─────────────────────────────────────────────────────────

@app.route("/api/iv", methods=["POST"])
def implied_vol():
    """
    Solve for implied volatility given market price and option parameters.

    Request JSON:
        flag         (str)  : 'c' or 'p'
        market_price (float): Observed market price of the option
        S, K, T, r, q       : Standard option parameters

    Response JSON:
        iv (float): Implied volatility (decimal, e.g. 0.2045 = 20.45%)
    """
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
        return jsonify({"iv": round(float(iv), 6)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


# ─────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Option Pricing Tool — Web Application")
    print("  Open in browser: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)
