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
    POST /api/local_vol/surface : Dupire local volatility on a (S, t) grid
    POST /api/local_vol/validate: Reprice the vanillas; the correctness evidence
    POST /api/european          : European option on the local volatility
    POST /api/heston/calibrate  : Fit Heston to a chain, with the SSVI comparison
    POST /api/heston/european   : European option by COS under Heston
    POST /api/heston/snowball   : Snowball by QE simulation -- one engine only

Usage:
    python app.py
    Then open: http://127.0.0.1:5000

Dependencies:
    pip install flask py_vollib numpy pandas matplotlib openpyxl scipy

Author: GitHub Portfolio Project
"""

import io as io_module
import os
import re
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import openpyxl
from flask import Flask, jsonify, render_template, request, send_file
from openpyxl.styles import Alignment, Font, PatternFill

# Ensure sibling modules (bs_pricing, greeks_calc) are importable
sys.path.insert(0, os.path.dirname(__file__))

# The library lives under src/, so a clone that has not been installed cannot
# import it. Appended rather than inserted: an installed copy wins, so a
# developer running against a `pip install -e .` checkout does not silently
# get this directory instead.
#
# Until now `python app.py` failed from a fresh clone with
# "ModuleNotFoundError: No module named 'otcpricer'" -- the tests worked only
# because pyproject sets pythonpath = ["src"] for pytest. For a tool meant to
# be handed over, the entry point has to run.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.append(_SRC)

from bs_pricing import bs_price, calc_implied_volatility, monte_carlo_price, validate_params
from greeks_calc import calc_delta, calc_gamma, calc_rho, calc_theta, calc_vega

from otcpricer.core.exceptions import MarketDataError, OtcPricerError

# cn_fdm.CN_FDM is gone. Its snowball price was the last legacy number any
# route could still serve -- reachable through /api/cn_fdm with
# calc_greeks=false -- and that branch now runs the reviewed engine. The two
# compatibility layers above stay: bs_pricing and greeks_calc are thin
# forwarders onto otcpricer.models.bsm and agree with it bit for bit.
from otcpricer.core.market import EquityMarket
from otcpricer.core.types import BarrierKind, KIStrikeMode, OptionType
from otcpricer.engines.barrier import (
    price_shark_fin_analytic,
    price_shark_fin_mc,
    price_shark_fin_pde,
)
from otcpricer.engines.european import price_european_pde
from otcpricer.engines.fourier import price_heston_european
from otcpricer.engines.mc import MonteCarloConfig, price_snowball_mc
from otcpricer.engines.pde import LOCAL_VOL_PDE_CONFIG, PDEConfig, price_snowball_pde
from otcpricer.engines.risk import (
    SnowballRisk,
    compare_snowball_risk,
    snowball_risk_mc,
    snowball_risk_pde,
)

# Volatility surface (Phase 2a). Offline by default: core.config ships with
# source="snapshot", so these endpoints answer from data/snapshots unless a
# request explicitly asks for the live feed.
from otcpricer.marketdata.carry import TailCarry, measure_carry
from otcpricer.marketdata.contracts import SUPPORTED_UNDERLYINGS, resolve_underlying
from otcpricer.marketdata.forward import DEFAULT_RATE
from otcpricer.marketdata.marks import evaluate_gates, load_mark, publish_mark
from otcpricer.marketdata.rates import rate_curve_from_panel
from otcpricer.marketdata.snapshots import list_snapshots
from otcpricer.models.heston import HestonParams
from otcpricer.models.heston_calibration import calibrate_heston
from otcpricer.models.local_vol import local_vol_from_fitted
from otcpricer.models.surface import fit_surface
from otcpricer.products.irs import flat_curve, price_swap
from otcpricer.products.shark_fin import CONTINUOUS, SharkFinTerms
from otcpricer.products.snowball import SnowballTerms
from otcpricer.products.snowball_quote import (
    PAR_COUPON_PDE_CONFIG,
    decompose_par_coupon,
    solve_par_coupon,
    verify_par_coupon_mc,
)
from otcpricer.reports.heston_report import heston_report
from otcpricer.reports.local_vol_report import local_vol_report, round_trip
from otcpricer.reports.snowball_report import snowball_report
from otcpricer.reports.surface_report import surface_report

app = Flask(__name__)


# ── Shared helper: compute all 5 Greeks at one point ──────────────────────────


def _greeks_at(flag, S, K, T, r, sigma, q):
    return {
        "Delta": round(float(calc_delta(flag, S, K, T, r, sigma, q)), 6),
        "Gamma": round(float(calc_gamma(flag, S, K, T, r, sigma, q)), 6),
        "Vega": round(float(calc_vega(flag, S, K, T, r, sigma, q)), 6),
        "Theta": round(float(calc_theta(flag, S, K, T, r, sigma, q)), 6),
        "Rho": round(float(calc_rho(flag, S, K, T, r, sigma, q)), 6),
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
        data = request.get_json(force=True)
        flag = str(data.get("flag", "c")).lower()
        S = float(data["S"])
        K = float(data["K"])
        T = float(data["T"])
        r = float(data["r"])
        sigma = float(data["sigma"])
        q = float(data.get("q", 0.0))
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid input parameters: {e}"}), 400

    if flag not in ("c", "p"):
        return jsonify({"error": "flag must be 'c' (Call) or 'p' (Put)"}), 400

    valid, msg = validate_params(S, K, T, r, sigma, q)
    if not valid:
        return jsonify({"error": msg}), 400

    bsm_price_val = bs_price(flag, S, K, T, r, sigma, q)
    mc_price_val = monte_carlo_price(flag, S, K, T, r, sigma, q, n_simulations=100000)
    greeks = _greeks_at(flag, S, K, T, r, sigma, q)

    # Chart: 100 S-values spanning 0.6K – 1.4K
    S_arr = np.linspace(0.6 * K, 1.4 * K, 100)

    prices_arr = [round(float(bs_price(flag, s, K, T, r, sigma, q)), 4) for s in S_arr]
    intrinsic_arr = [
        round(float(max(s - K, 0) if flag == "c" else max(K - s, 0)), 4) for s in S_arr
    ]
    greek_curves = {
        "Delta": [round(float(calc_delta(flag, s, K, T, r, sigma, q)), 6) for s in S_arr],
        "Gamma": [round(float(calc_gamma(flag, s, K, T, r, sigma, q)), 6) for s in S_arr],
        "Vega": [round(float(calc_vega(flag, s, K, T, r, sigma, q)), 6) for s in S_arr],
        "Theta": [round(float(calc_theta(flag, s, K, T, r, sigma, q)), 6) for s in S_arr],
        "Rho": [round(float(calc_rho(flag, s, K, T, r, sigma, q)), 6) for s in S_arr],
    }

    return jsonify(
        {
            "bsm_price": round(float(bsm_price_val), 4),
            "mc_price": round(float(mc_price_val), 4),
            "greeks": {k.lower(): v for k, v in greeks.items()},
            "price_curve": {
                "spots": [round(float(s), 2) for s in S_arr],
                "prices": prices_arr,
                "intrinsic": intrinsic_arr,
            },
            "greeks_curve": {k.lower(): v for k, v in greek_curves.items()},
            "K": K,
        }
    )


# ─────────────────────────────────────────────────────────
# API: Implied Volatility Solver
# ─────────────────────────────────────────────────────────


@app.route("/api/iv", methods=["POST"])
def implied_vol():
    """Solve for implied volatility from a market price."""
    try:
        data = request.get_json(force=True)
        flag = str(data.get("flag", "c")).lower()
        market_price = float(data["market_price"])
        S = float(data["S"])
        K = float(data["K"])
        T = float(data["T"])
        r = float(data["r"])
        q = float(data.get("q", 0.0))
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
        data = request.get_json(force=True)
        positions = data.get("positions", [])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    results = []
    totals = {
        "net_delta": 0.0,
        "net_gamma": 0.0,
        "net_vega": 0.0,
        "net_theta": 0.0,
        "net_rho": 0.0,
        "total_mtm": 0.0,
        "total_pnl": 0.0,
        "total_margin": 0.0,
    }

    for pos in positions:
        try:
            flag = str(pos["flag"]).lower()
            S = float(pos["S"])
            K = float(pos["K"])
            T = float(pos["T"])
            r = float(pos["r"])
            sigma = float(pos["sigma"])
            q = float(pos.get("q", 0.0))
            qty = float(pos.get("qty", 1.0))
            entry = float(pos.get("entry", pos.get("entry_price", 0.0)))

            price = float(bs_price(flag, S, K, T, r, sigma, q))
            greeks_single = _greeks_at(flag, S, K, T, r, sigma, q)

            mtm = round(price * qty, 4)
            pnl = round((price - entry) * qty, 4)
            margin = round(abs(qty) * entry, 4) if qty > 0 else round(abs(qty) * S * 0.15, 4)

            result_pos = {
                "id": pos.get("id", ""),
                "label": pos.get("label", ""),
                "flag": flag,
                "S": S,
                "K": K,
                "T": round(T, 6),
                "T_days": round(T * 365),
                "r": r,
                "sigma": sigma,
                "q": q,
                "qty": qty,
                "entry": round(entry, 4),
                "mtm": mtm,
                "pnl": pnl,
                "margin": margin,
                "delta_net": round(greeks_single["Delta"] * qty, 6),
                "gamma_net": round(greeks_single["Gamma"] * qty, 6),
                "vega_net": round(greeks_single["Vega"] * qty, 6),
                "theta_net": round(greeks_single["Theta"] * qty, 6),
                "rho_net": round(greeks_single["Rho"] * qty, 6),
                "error": None,
            }

            totals["net_delta"] = round(totals["net_delta"] + result_pos["delta_net"], 6)
            totals["net_gamma"] = round(totals["net_gamma"] + result_pos["gamma_net"], 6)
            totals["net_vega"] = round(totals["net_vega"] + result_pos["vega_net"], 6)
            totals["net_theta"] = round(totals["net_theta"] + result_pos["theta_net"], 6)
            totals["net_rho"] = round(totals["net_rho"] + result_pos["rho_net"], 6)
            totals["total_mtm"] = round(totals["total_mtm"] + mtm, 4)
            totals["total_pnl"] = round(totals["total_pnl"] + pnl, 4)
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
        data = request.get_json(force=True)
        flag = str(data.get("flag", "c")).lower()
        S = float(data["S"])
        T = float(data["T"])
        r = float(data["r"])
        sigma = float(data["sigma"])
        q = float(data.get("q", 0.0))
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
        data = request.get_json(force=True)
        flag = str(data.get("flag", "c")).lower()
        S = float(data["S"])
        T = float(data["T"])
        r = float(data["r"])
        q = float(data.get("q", 0.0))
        smile_data = data.get("pairs", data.get("smile_data", []))
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    points = []
    for item in smile_data:
        try:
            K_i = float(item["K"])
            mp_i = float(item["market_price"])
            iv = calc_implied_volatility(flag, mp_i, S, K_i, T, r, q)
            points.append({"K": K_i, "iv": round(float(iv) * 100, 4), "error": None})
        except Exception as e:
            points.append({"K": float(item.get("K", 0)), "iv": None, "error": str(e)})

    return jsonify(
        {
            "results": points,
            "note": (
                "This inverts hand-entered prices one expiry at a time and fits "
                "nothing. For a surface built from a real option chain -- forwards "
                "implied from parity, quotes screened, SSVI calibrated under the "
                "no-arbitrage constraints -- use /api/vol_surface/fit."
            ),
        }
    )


# ─────────────────────────────────────────────────────────
# API: Volatility Surface (SSVI, real option chains)
# ─────────────────────────────────────────────────────────

#: Fitting a surface takes a few hundred milliseconds, and the query endpoint
#: gets called repeatedly while a user drags a chart around. Keyed on
#: everything that changes the answer, so a different source or rate cannot be
#: served a stale fit.
_SURFACE_CACHE = {}

#: Where an uploaded chain is kept.
#:
#: Files are named by a hash of their own contents, never by the name the
#: caller supplied -- a client-chosen filename reaching a path join is the
#: oldest upload defect there is. The hash also makes the same chain uploaded
#: twice one file rather than two.
UPLOAD_DIR = Path("data/uploads")

#: Where published marks are kept. One file per underlying, gitignored:
#: a mark records which surface *this* installation blessed.
MARK_DIR = Path("data/marks")

#: Largest chain this will accept, in bytes. The committed 510300 snapshot is
#: 126 contracts and about 9 KB as CSV; the CFFEX index option boards are a
#: few thousand rows. 8 MB is far above anything real and far below anything
#: that would exhaust memory.
MAX_UPLOAD_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Everything that decides which option chain gets fitted.

    One object rather than six positional arguments threaded through three
    cache layers. This project has already had three defects from argument
    order alone -- ``_panel_rate_curve``, ``/api/carry`` and
    ``AkshareSource.fetch`` each evaluated their arguments in the wrong order,
    and each read correctly -- and six positionals through three layers is
    where the fourth would come from.

    Frozen and hashable, so it is also the cache key instead of a tuple that
    each layer assembles by hand. Adding a source argument is then one field,
    not an edit at every layer and every call site.
    """

    underlying: str
    source_kind: str | None = None
    snapshot: str | None = None
    rate: float = DEFAULT_RATE
    csv_path: Path | None = None
    as_of: object | None = None

    @property
    def key(self):
        """Hashable identity, with the rate rounded as the old key was."""
        return (
            self.underlying,
            self.source_kind,
            self.snapshot,
            round(float(self.rate), 10),
            str(self.csv_path or ""),
            str(self.as_of or ""),
        )

    def describe(self):
        """What a reader needs to see beside a number that came from here."""
        parts = [self.underlying, self.source_kind or "configured default"]
        if self.snapshot:
            parts.append(f"snapshot={self.snapshot}")
        if self.csv_path:
            parts.append(f"csv={self.csv_path.name}")
        if self.as_of:
            parts.append(f"as_of={self.as_of}")
        parts.append(f"r={self.rate:.6g}")
        return " / ".join(parts)


def _cached_surface(spec):
    if spec.key not in _SURFACE_CACHE:
        _SURFACE_CACHE[spec.key] = fit_surface(
            spec.underlying,
            source_kind=spec.source_kind,
            snapshot=spec.snapshot,
            csv_path=spec.csv_path,
            as_of=spec.as_of,
            rate=spec.rate,
        )
    return _SURFACE_CACHE[spec.key]


def _upload_path(token):
    """Resolve an upload token to a file, refusing anything else.

    The token is the content hash this server assigned on upload, never a name
    the caller chose. Validated by shape rather than merely joined to a
    directory: a token is sixteen hex characters, so ``..`` and path
    separators cannot occur in a valid one, and refusing on the shape leaves
    nothing to reason about afterwards.
    """
    if not re.fullmatch(r"[0-9a-f]{16}", str(token or "")):
        raise ValueError(
            "csv_token must be the 16-character token returned by /api/marketdata/upload"
        )
    path = UPLOAD_DIR / f"{token}.csv"
    if not path.exists():
        raise ValueError(f"no uploaded chain {token!r}; upload it again.")
    return path


def _surface_request(data):
    """Pull the source arguments out of a request body into a SourceSpec."""
    underlying = str(data.get("underlying", "510050")).strip()
    if underlying not in SUPPORTED_UNDERLYINGS:
        raise ValueError(
            f"underlying {underlying!r} is not supported; available: "
            f"{', '.join(sorted(SUPPORTED_UNDERLYINGS))}"
        )
    source_kind = data.get("source")
    csv_path = None
    if data.get("csv_token"):
        csv_path = _upload_path(data["csv_token"])
        # An uploaded chain implies the csv kind. Stated rather than required,
        # because a caller who sent a token and forgot `source` meant csv.
        source_kind = source_kind or "csv"

    return SourceSpec(
        underlying=underlying,
        source_kind=source_kind,
        snapshot=data.get("snapshot"),
        rate=float(data.get("r", data.get("rate", DEFAULT_RATE))),
        csv_path=csv_path,
        as_of=_panel_date(data["as_of"]) if data.get("as_of") else None,
    )


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
        spec = _surface_request(data)
        want_report = bool(data.get("report", False))
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    try:
        fitted = _cached_surface(spec)
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
        spec = _surface_request(data)
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
        fitted = _cached_surface(spec)
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

    return jsonify(
        {
            "underlying": spec.underlying,
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
        }
    )


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
    return jsonify(
        {
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
        }
    )


# ─────────────────────────────────────────────────────────
# API: Market data -- upload, mark, publish
# ─────────────────────────────────────────────────────────


@app.route("/api/marketdata/upload", methods=["POST"])
def marketdata_upload():
    """
    Accept an option chain as CSV and return a token for it.

    Columns: ``code``, ``last``, ``previous_settlement``, ``strike``,
    ``expiry``, ``days_to_expiry``, and optionally ``as_of``. One row per
    contract. ``as_of`` is the date the quotes were taken; without it the
    chain dates itself today, which fits identically and labels a published
    mark with the wrong day.

    The file is stored under a hash of its own contents. The name the caller
    sent is never used for anything but the echo in this response: a
    client-chosen filename reaching a path join is the oldest defect in the
    category, and hashing also makes the same chain uploaded twice one file.

    Request: multipart form with a ``file`` field, or a raw CSV body.
    Response JSON: {token, bytes, rows, as_of, columns, note}
    """
    import hashlib

    upload = request.files.get("file")
    raw = upload.read() if upload is not None else request.get_data()
    if not raw:
        return jsonify({"error": "no CSV content; send a 'file' field or a raw body"}), 400
    if len(raw) > MAX_UPLOAD_BYTES:
        return jsonify(
            {
                "error": (
                    f"{len(raw)} bytes exceeds the {MAX_UPLOAD_BYTES}-byte limit. "
                    f"A full CFFEX board is a few thousand rows; anything this "
                    f"large is not an option chain."
                )
            }
        ), 400

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return jsonify(
            {"error": "the file is not UTF-8. Save it as UTF-8 (or UTF-8 with BOM)."}
        ), 400

    import csv as _csv

    reader = _csv.DictReader(io_module.StringIO(text))
    columns = list(reader.fieldnames or [])
    required = {"code", "last", "previous_settlement", "strike", "expiry", "days_to_expiry"}
    missing = sorted(required - set(columns))
    if missing:
        return jsonify(
            {
                "error": (
                    f"missing required column(s): {', '.join(missing)}. "
                    f"Expected {', '.join(sorted(required))} plus an optional as_of."
                ),
                "columns_found": columns,
            }
        ), 400

    rows = list(reader)
    if not rows:
        return jsonify({"error": "the file has a header and no rows"}), 400

    token = hashlib.sha256(raw).hexdigest()[:16]
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    (UPLOAD_DIR / f"{token}.csv").write_bytes(raw)

    stated = sorted({(r.get("as_of") or "").strip() for r in rows} - {""})
    return jsonify(
        {
            "token": token,
            "bytes": len(raw),
            "rows": len(rows),
            "columns": columns,
            "as_of": stated[0] if len(stated) == 1 else None,
            "filename_sent": None if upload is None else upload.filename,
            "note": (
                "Pass this token as csv_token to any surface endpoint. Stored "
                "under a hash of the contents, so the same chain uploaded "
                "again returns the same token rather than a second file."
                + ("" if stated else " No as_of column, so this chain will date itself today.")
                + (
                    f" Two different as_of values present ({', '.join(stated)}); "
                    f"fitting will refuse it -- one file is one valuation date."
                    if len(stated) > 1
                    else ""
                )
            ),
        }
    )


@app.route("/api/marketdata/mark", methods=["GET", "POST"])
def marketdata_mark():
    """
    The published surface for an underlying, or nothing if it has not been marked.

    Reading re-fits the mark's own inputs and checks the fingerprint, so a
    surface that no longer reproduces is reported rather than served. An
    unmarked underlying is an ordinary state -- it is every underlying before
    the first publish of the day -- so it answers 200 with ``marked: false``
    rather than 404.

    Request JSON: {underlying}
    """
    data = request.get_json(force=True, silent=True) or {}
    underlying = str(data.get("underlying") or request.args.get("underlying") or "510300").strip()
    if underlying not in SUPPORTED_UNDERLYINGS:
        return jsonify({"error": f"underlying {underlying!r} is not supported"}), 400

    try:
        stored = load_mark(underlying, directory=MARK_DIR, verify=False)
    except OtcPricerError as e:
        return jsonify({"error": str(e)}), 400
    if stored is None:
        return jsonify(
            {
                "underlying": underlying,
                "marked": False,
                "note": (
                    "No published mark. Fit a surface, check the gates, and "
                    "publish it -- until then every pricing call fits its own."
                ),
            }
        )

    # Rebuilt from what the mark recorded, which is the point of recording the
    # inputs rather than the surface. A mark published off an uploaded chain
    # carries its token, so it re-fits from the same file.
    try:
        spec = SourceSpec(
            underlying=underlying,
            source_kind=stored.source_kind,
            snapshot=stored.snapshot,
            rate=stored.rate,
            csv_path=_upload_path(stored.csv_token) if stored.csv_token else None,
        )
    except ValueError as e:
        return jsonify(
            {
                "underlying": underlying,
                "marked": True,
                "reproduces": False,
                "error": f"the mark's own inputs are no longer available: {e}",
                "mark": stored.as_dict(),
            }
        ), 409

    try:
        fitted = _cached_surface(spec)
        verified = load_mark(underlying, directory=MARK_DIR, fitted=fitted)
    except OtcPricerError as e:
        return jsonify(
            {
                "underlying": underlying,
                "marked": True,
                "reproduces": False,
                "error": str(e),
                "mark": stored.as_dict(),
            }
        ), 409

    return jsonify(
        {
            "underlying": underlying,
            "marked": True,
            "reproduces": True,
            "mark": (verified or stored).as_dict(),
            "spec": spec.describe(),
            "note": (
                "Re-fitted from the mark's own inputs and the fingerprint "
                "matched, so this is the surface that was published."
            ),
        }
    )


@app.route("/api/marketdata/publish", methods=["POST"])
def marketdata_publish():
    """
    Run the gates on a fitted surface and publish it as today's mark.

    This is the step a desk does once, before pricing anything: fit, check,
    publish. Five gates run and all five are reported -- fit quality,
    butterfly, calendar, an independent arbitrage scan, and the move against
    the previous mark. A failure refuses the publication and says which gate
    and by how much.

    ``force`` publishes anyway and records that it did. It exists because the
    alternative to an override is editing the thresholds, which would remove
    the check for every later day.

    ``dry_run`` runs the gates and writes nothing. The interface shows the gate
    results as soon as a surface is fitted, well before anyone decides to
    publish; without this that preview would publish a mark on every fit -- a
    side effect nobody asked for, arriving silently, which is the shape of
    defect the rest of this project exists to remove.

    Request JSON: {underlying, source, snapshot, csv_token, as_of, r, force,
                   dry_run}
    Response JSON: {published, mark, gates, ...} or 409 with the gate results.
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        spec = _surface_request(data)
        force = bool(data.get("force", False))
        dry_run = bool(data.get("dry_run", False))
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    try:
        fitted = _cached_surface(spec)
    except OtcPricerError as e:
        return jsonify({"error": str(e)}), 400

    gates = evaluate_gates(fitted, load_mark(spec.underlying, directory=MARK_DIR, verify=False))
    if dry_run:
        failed = [g for g in gates if not g.passed]
        return jsonify(
            {
                "published": False,
                "dry_run": True,
                "would_publish": not failed,
                "gates": [g.as_dict() for g in gates],
                "spec": spec.describe(),
                "note": (
                    "Nothing was written. "
                    + (
                        f"{len(failed)} of {len(gates)} gates would refuse this: "
                        + ", ".join(g.name for g in failed)
                        + ". Publishing needs force=True."
                        if failed
                        else "All gates pass, so a publish would be accepted."
                    )
                ),
            }
        )
    try:
        mark = publish_mark(
            fitted,
            source_kind=spec.source_kind,
            snapshot=spec.snapshot,
            rate=spec.rate,
            csv_token=None if spec.csv_path is None else spec.csv_path.stem,
            directory=MARK_DIR,
            force=force,
        )
    except OtcPricerError as e:
        return jsonify(
            {
                "published": False,
                "error": str(e),
                "gates": [g.as_dict() for g in gates],
                "spec": spec.describe(),
            }
        ), 409

    return jsonify(
        {
            "published": True,
            "forced": any(not g.passed for g in gates),
            "mark": mark.as_dict(),
            "gates": [g.as_dict() for g in gates],
            "spec": spec.describe(),
            "surface": fitted.as_dict(),
        }
    )


# ─────────────────────────────────────────────────────────
# API: Dupire Local Volatility
# ─────────────────────────────────────────────────────────

#: Local volatility surfaces are derived from a fitted surface, so they inherit
#: that cache's reasoning and add the two inputs of their own that change the
#: answer: the spot the forward curve is anchored at, and the drift used beyond
#: the last quoted expiry.
_LOCAL_VOL_CACHE = {}


def _cached_local_vol(spec, spot, tail_drift):
    key = (
        spec.key,
        None if spot is None else round(float(spot), 10),
        round(float(tail_drift), 10),
    )
    if key not in _LOCAL_VOL_CACHE:
        fitted = _cached_surface(spec)
        _LOCAL_VOL_CACHE[key] = (
            fitted,
            local_vol_from_fitted(fitted, spot=spot, tail_drift=tail_drift),
        )
    return _LOCAL_VOL_CACHE[key]


def _local_vol_request(data):
    """Pull the local-volatility arguments out of a request body."""
    spec = _surface_request(data)
    spot = data.get("S", data.get("spot"))
    spot = None if spot is None else float(spot)
    if spot is not None and spot <= 0.0:
        raise ValueError("S must be positive")
    # The drift past the last quoted expiry is the caller's own carry. Not
    # extrapolated from the implied forwards: those come from options stopping
    # at five months, and carrying their -4% carry out to three years would
    # drop the forward 14% on that evidence alone.
    q = float(data.get("q", 0.0))
    basis = float(data.get("basis_pa", 0.0))
    return spec, spot, spec.rate - q - basis


@app.route("/api/local_vol/surface", methods=["POST"])
def local_vol_surface():
    """
    Dupire local volatility on a (spot, time) grid.

    Local volatility is a different object from implied volatility and reading
    one as the other is a standing trap, so the response also returns the
    implied volatility at the same points. On the captured chains the 80%
    level carries over twice the at-the-money local volatility, which is where
    a snowball's knock-in risk actually lives.

    Request JSON: {underlying, source, snapshot, r, q, basis_pa, S,
                   horizon, moneyness, n_spots, n_times}
      moneyness    half-width of the display range in log-moneyness; defaults
                   to three at-the-money standard deviations. Deliberately
                   narrower than the solver's boundary, which is reported
                   separately as solver_moneyness_range.

    Response JSON: {spots, times, local_vol, implied_vol, diagnostics,
                    forward_curve} with local_vol[i][j] at times[i], spots[j].
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        spec, spot, tail = _local_vol_request(data)
        horizon = float(data.get("horizon", 3.0))
        if horizon <= 0.0:
            raise ValueError("horizon must be positive")
        n_spots = max(3, min(int(data.get("n_spots", 41)), 401))
        n_times = max(2, min(int(data.get("n_times", 25)), 201))
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    try:
        fitted, lv = _cached_local_vol(spec, spot, tail)
    except OtcPricerError as e:
        return jsonify({"error": str(e)}), 400

    # Display range, not the solver's boundary. boundary_moneyness is sized so
    # a Dirichlet condition is harmless, which at a three-year horizon reaches
    # +/-2.65 in log-moneyness -- spot from 7% to 1412% of forward. Correct for
    # a PDE and useless for a heat map, so this matches the surface query
    # endpoint's three at-the-money standard deviations instead.
    reach = max(float(data.get("moneyness", 3.0 * lv.atm_total_vol(horizon))), 1e-3)
    moneyness = np.linspace(-reach, reach, n_spots)
    spots = lv.spot * np.exp(moneyness)
    times = np.linspace(max(horizon / n_times, 1e-4), horizon, n_times)
    # The diagnostics still scan the solver's region, which is what they are
    # about -- clamping and the denominator have to be checked where the engine
    # actually looks, not where a chart does.
    scan_lower, scan_upper = lv.boundary_moneyness(horizon)
    scan_spots = lv.spot * np.exp(np.linspace(scan_lower, scan_upper, 241))

    local, implied = [], []
    for t in times:
        local.append([round(float(v), 6) for v in lv.local_vol(spots, float(t))])
        y = lv.log_moneyness(spots, float(t))
        implied.append([round(float(v), 6) for v in lv.surface.implied_vol(y, float(t))])

    return jsonify(
        {
            "underlying": spec.underlying,
            "as_of": fitted.as_of,
            "spot": round(lv.spot, 6),
            "spots": [round(float(x), 6) for x in spots],
            "moneyness": [round(float(k), 6) for k in moneyness],
            "times": [round(float(t), 6) for t in times],
            "local_vol": local,
            "implied_vol": implied,
            "forward_curve": lv.forwards.as_dict(),
            "diagnostics": lv.scan(scan_spots, times, dx=(scan_upper - scan_lower) / 240).as_dict(),
            "solver_moneyness_range": [round(scan_lower, 6), round(scan_upper, 6)],
            "extrapolated_fraction": round(lv.extrapolated_fraction(horizon), 6),
            "note": (
                "local_vol[i][j] is at times[i], spots[j]. Local volatility is not "
                "implied volatility; implied_vol is returned at the same points so "
                "the two are not confused. Times past the last quoted expiry are "
                "extrapolated -- see extrapolated_fraction."
            ),
        }
    )


@app.route("/api/local_vol/validate", methods=["POST"])
def local_vol_validate():
    """
    Reprice the vanillas the surface was built from. The correctness evidence.

    A local volatility is derived from an implied volatility surface, so there
    is exactly one way to know it is right: price the options it came from and
    check the smile comes back. Dupire guarantees that in continuous time, so
    any residual is the engine's discretisation and nothing else -- which makes
    the natural budget the surface's own calibration error.

    Measured on the captured chains at a production grid: 0.0064 volatility
    points on 510300 against a 0.654 fit error, and 0.426 against 0.587 on
    510050.

    Request JSON: {underlying, source, snapshot, r, q, basis_pa, S, report}
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        spec, spot, tail = _local_vol_request(data)
        want_report = bool(data.get("report", False))
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    try:
        fitted, lv = _cached_local_vol(spec, spot, tail)
        rows = round_trip(fitted, lv)
    except OtcPricerError as e:
        return jsonify({"error": str(e)}), 400

    worst = max((abs(r.error_vol_points) for r in rows), default=0.0)
    budget = fitted.calibration.rms_vol_error * 100.0
    payload = {
        "underlying": spec.underlying,
        "as_of": fitted.as_of,
        "rows": [r.as_dict() for r in rows],
        "worst_error_vol_points": round(worst, 6),
        "calibration_rms_vol_points": round(budget, 6),
        "share_of_budget": round(worst / budget, 6) if budget else None,
        "passes": bool(worst < budget),
        "note": (
            "Errors are in volatility points. The budget is the surface's own "
            "calibration error: an engine adding less noise than the data "
            "already carries is as accurate as the inputs can justify."
        ),
    }
    if want_report:
        payload["report"] = local_vol_report(fitted, lv, round_trip_rows=rows)
    return jsonify(payload)


@app.route("/api/european", methods=["POST"])
def european_local_vol():
    """
    Price a European option on the Dupire local volatility.

    A skew-consistent vanilla quote, which a single flat volatility cannot
    give. The drift comes from the surface's own forward curve rather than from
    r - q, because the surface is parameterised against that curve and any
    other would evaluate the volatility at the wrong moneyness.

    Request JSON: {underlying, source, snapshot, r, q, basis_pa, S, K, T, flag}
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        spec, spot, tail = _local_vol_request(data)
        strike = float(data["K"])
        T = float(data["T"])
        flag = str(data.get("flag", "c")).lower()
        if strike <= 0.0 or T <= 0.0:
            raise ValueError("K and T must be positive")
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    try:
        fitted, lv = _cached_local_vol(spec, spot, tail)
        result = price_european_pde(lv.spot, strike, T, spec.rate, flag, lv)
        forward = float(lv.forward(T))
        implied = float(
            calc_implied_volatility(flag, result.price, forward, strike, T, spec.rate, spec.rate)
        )
    except (OtcPricerError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(
        {
            "price": round(result.price, 8),
            "spot": round(lv.spot, 6),
            "strike": strike,
            "T": T,
            "forward": round(forward, 6),
            "implied_vol": round(implied, 6),
            "surface_implied_vol": round(
                float(lv.surface.implied_vol(np.log(strike / forward), T)), 6
            ),
            "local_vol_at_spot": round(float(lv.local_vol(lv.spot, T / 2.0)), 6),
            "space_nodes": result.space_nodes,
            "time_steps": result.time_steps,
            "vol_model": result.vol_model,
            "extrapolated_fraction": round(lv.extrapolated_fraction(T), 6),
            "note": (
                "implied_vol inverts the price at this expiry's own forward, which "
                "is what makes it comparable with surface_implied_vol. Inverting at "
                "spot would measure the carry instead -- about -4% a year here."
            ),
        }
    )


# ─────────────────────────────────────────────────────────
# API: Heston Stochastic Volatility
# ─────────────────────────────────────────────────────────

#: Grid for the shark fin.
#:
#: Finer than the snowball default because a discretely monitored barrier
#: converges only first order in dx -- every observation re-creates a
#: discontinuity at the barrier, so the solution is never smooth for long.
#: Measured on a daily down-and-out put: 1.3381, 1.3529, 1.3609, 1.3649,
#: 1.3669 as the node count doubled, against a Monte Carlo mean of 1.3711.
SHARK_FIN_PDE_CONFIG = PDEConfig(space_nodes=3200, steps_per_year=2016)


#: Heston fits cost about a minute, so they are cached like the surfaces.
_HESTON_CACHE = {}


def _cached_heston(spec):
    key = spec.key
    if key not in _HESTON_CACHE:
        fitted = _cached_surface(spec)
        usable = fitted.surface_input.usable_slices
        calibration = calibrate_heston(
            [s.year_fraction for s in usable],
            [np.array([q.log_moneyness for q in s.points]) for s in usable],
            [np.array([q.implied_vol for q in s.points]) for s in usable],
            [np.array([q.vega for q in s.points]) for s in usable],
            forwards=[s.forward.forward for s in usable],
            rate=spec.rate,
            terms=192,
        )
        _HESTON_CACHE[key] = (fitted, calibration)
    return _HESTON_CACHE[key]


def _heston_params_from(data):
    """Explicit parameters from a request body, or None to calibrate."""
    names = ("v0", "kappa", "theta", "sigma", "rho")
    if not all(name in data for name in names):
        return None
    return HestonParams(**{name: float(data[name]) for name in names})


@app.route("/api/heston/calibrate", methods=["POST"])
def heston_calibrate():
    """
    Fit Heston to a real option chain, alongside the SSVI fit of the same quotes.

    The comparison is the point, and it does not flatter Heston: measured on
    the same quotes with the same vega-weighted objective, Heston is 10-12%
    worse on three expiries and 38-41% worse on six, because SSVI carries one
    parameter per expiry against Heston's five in total. Heston is here for a
    forward smile that does not flatten, not for accuracy.

    Request JSON: {underlying, source, snapshot, r, report}
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        spec = _surface_request(data)
        want_report = bool(data.get("report", False))
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    try:
        fitted, calibration = _cached_heston(spec)
    except OtcPricerError as e:
        return jsonify({"error": str(e)}), 400

    ssvi_rms = fitted.calibration.rms_vol_error * 100.0
    payload = calibration.as_dict()
    payload["underlying"] = spec.underlying
    payload["as_of"] = fitted.as_of
    payload["ssvi_rms_vol_points"] = round(ssvi_rms, 5)
    payload["heston_over_ssvi"] = (
        round(calibration.rms_vol_points / ssvi_rms, 4) if ssvi_rms else None
    )
    payload["fits_worse_than_ssvi"] = bool(calibration.rms_vol_points > ssvi_rms)
    if want_report:
        payload["report"] = heston_report(
            calibration,
            underlying=spec.underlying,
            as_of=fitted.as_of,
            ssvi_rms_vol_points=ssvi_rms,
        )
    return jsonify(payload)


@app.route("/api/heston/european", methods=["POST"])
def heston_european():
    """
    Price a European option under Heston by the COS method.

    Cross-checked against QuantLib's analytic Heston engine to 5e-09 across
    three parameter sets, four maturities and three strikes.

    Parameters come either explicitly (v0, kappa, theta, sigma, rho) or from a
    calibration to the named underlying.

    Request JSON: {K, T, flag, S, r, q, terms, width} plus either
                  {v0, kappa, theta, sigma, rho} or {underlying, source, snapshot}
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        strike = float(data["K"])
        T = float(data["T"])
        flag = str(data.get("flag", "c")).lower()
        rate = float(data.get("r", 0.02))
        dividend = float(data.get("q", 0.0))
        terms = int(data.get("terms", 1024))
        width = float(data.get("width", 30.0))
        if strike <= 0.0 or T <= 0.0:
            raise ValueError("K and T must be positive")
        params = _heston_params_from(data)
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    try:
        if params is None:
            spec = _surface_request(data)
            fitted, calibration = _cached_heston(spec)
            params = calibration.params
            # The surface carries no spot -- it is parameterised in forward
            # terms on purpose -- so the default is the nearest expiry's
            # forward discounted back. A caller who has a spot should pass it.
            nearest = min(fitted.surface_input.usable_slices, key=lambda sl: sl.year_fraction)
            implied_spot = nearest.forward.forward * float(
                np.exp(-spec.rate * nearest.year_fraction)
            )
            spot = float(data.get("S", implied_spot))
            source = f"calibrated:{spec.underlying}"
        else:
            spot = float(data.get("S", 100.0))
            source = "explicit"
        result = price_heston_european(
            params,
            spot=spot,
            strike=strike,
            maturity=T,
            rate=rate,
            dividend=dividend,
            flag=flag,
            terms=terms,
            width=width,
        )
    except (OtcPricerError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    payload = result.as_dict()
    payload.update(
        {
            "spot": round(spot, 6),
            "strike": strike,
            "T": T,
            "flag": flag,
            "params": params.as_dict(),
            "params_source": source,
            "note": (
                "Puts are integrated and calls come from put-call parity: the "
                "call's cosine coefficients contain exp(upper bound), which "
                "overflows on a wide density. The truncation range is set from the "
                "model's own cumulants."
            ),
        }
    )
    return jsonify(payload)


@app.route("/api/heston/snowball", methods=["POST"])
def heston_snowball():
    """
    Price a snowball under Heston by Andersen's QE scheme.

    **This price rests on one engine.** Every other number this service returns
    is cross-checked by a second independent engine; a Heston snowball's
    partner would be a two-dimensional PDE with a cross-derivative and an ADI
    solver, which is not built. What is verified is the layer below: QE
    reproduces the COS European prices within three standard errors, so the
    simulation is consistent with the model. The step to a path-dependent
    payoff is not independently confirmed, and the response says so.

    QE rather than Euler is a requirement, not a refinement: the calibrated
    parameters violate Feller, so the variance reaches zero in 32-46% of
    simulated cells and a Euler step would go negative.

    Request JSON: {S0, T, ko_pct, ki_pct, coupon_pa, obs_freq, lockup_months,
                   r, q, paths, steps_per_year, seed} plus either
                  {v0, kappa, theta, sigma, rho} or {underlying, ...}
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        S0 = float(data.get("S0", 100.0))
        T = float(data.get("T", 1.0))
        terms_kwargs = {
            "S0": S0,
            "T": T,
            "ko_pct": float(data.get("ko_pct", 1.03)),
            "ki_pct": float(data.get("ki_pct", 0.75)),
            "coupon_pa": float(data.get("coupon_pa", 0.20)),
        }
        for optional in ("obs_freq", "ki_obs_freq", "lockup_months"):
            if optional in data:
                terms_kwargs[optional] = int(data[optional])
        rate = float(data.get("r", 0.02))
        dividend = float(data.get("q", 0.0))
        params = _heston_params_from(data)
        mc_config = MonteCarloConfig(
            paths=int(data.get("paths", 200_000)),
            steps_per_year=int(data.get("steps_per_year", 252)),
            seed=int(data.get("seed", 7)),
            antithetic=bool(data.get("antithetic", True)),
        )
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    try:
        if params is None:
            spec = _surface_request(data)
            _, calibration = _cached_heston(spec)
            params = calibration.params
            source = f"calibrated:{spec.underlying}"
        else:
            source = "explicit"
        snowball_terms = SnowballTerms(**terms_kwargs)
        market = EquityMarket(S=S0, r=rate, sigma=0.2, q=dividend)
        result = price_snowball_mc(snowball_terms, market, mc_config, heston=params)
    except (OtcPricerError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    payload = result.as_dict()
    payload.update(
        {
            "params": params.as_dict(),
            "params_source": source,
            "terms": dict(terms_kwargs),
            "steps_per_year": mc_config.steps_per_year,
        }
    )
    return jsonify(payload)


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
        data = request.get_json(force=True)
        flag = str(data.get("flag", "c")).lower()
        S = float(data["S"])
        K = float(data["K"])
        T = float(data["T"])
        r = float(data["r"])
        sigma = float(data["sigma"])
        q = float(data.get("q", 0.0))
        n = min(int(data.get("n_paths", data.get("n_simulations", 50000))), 200000)
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    valid, msg = validate_params(S, K, T, r, sigma, q)
    if not valid:
        return jsonify({"error": msg}), 400

    # A local generator, not np.random.seed. The seed is here to make this
    # endpoint reproducible, and against the process-global generator it did
    # not: seeding and drawing are two separate calls, so two concurrent
    # requests can both seed before either draws, and the second one continues
    # the stream instead of starting it. Measured with eight threads against
    # this route, 45% of requests returned a histogram the seed was supposed to
    # rule out.
    #
    # Harmless under the single-threaded development server and reachable the
    # moment this phase put eight waitress threads in front of it. It also
    # stopped mutating global numpy state on every request -- the engines all
    # use default_rng already, so nothing else here depended on it, which is
    # the only reason this was a reproducibility bug rather than a pricing one.
    rng = np.random.default_rng(42)
    Z = rng.standard_normal(n)
    S_T = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    payoffs = np.maximum(S_T - K, 0) if flag == "c" else np.maximum(K - S_T, 0)
    discounted = np.exp(-r * T) * payoffs

    counts, edges = np.histogram(discounted, bins=50)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()

    return jsonify(
        {
            "bins": [round(x, 4) for x in centers],
            "counts": counts.tolist(),
            "stats": {
                "mean": round(float(np.mean(discounted)), 4),
                "median": round(float(np.median(discounted)), 4),
                "ci95_lower": round(float(np.percentile(discounted, 5)), 4),
                "ci95_upper": round(float(np.percentile(discounted, 95)), 4),
                "std": round(float(np.std(discounted)), 4),
                "n_paths": n,
            },
        }
    )


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
        data = request.get_json(force=True)
        flag = str(data.get("flag", "c")).lower()
        S = float(data["S"])
        K = float(data["K"])
        T = float(data["T"])
        r = float(data["r"])
        sigma = float(data["sigma"])
        q = float(data.get("q", 0.0))
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    valid, msg = validate_params(S, K, T, r, sigma, q)
    if not valid:
        return jsonify({"error": msg}), 400

    sigma_offsets = [-0.20, -0.10, 0.0, +0.10, +0.20]
    T_offsets = [-60 / 365, -30 / 365, 0.0, +30 / 365, +60 / 365]
    sigma_labels = ["σ-20%", "σ-10%", "σ base", "σ+10%", "σ+20%"]
    T_labels = ["T-60d", "T-30d", "T base", "T+30d", "T+60d"]

    price_matrix, delta_matrix = [], []
    for ds in sigma_offsets:
        price_row, delta_row = [], []
        for dt in T_offsets:
            s_i = max(sigma + ds, 0.005)
            t_i = max(T + dt, 1 / 365)
            try:
                p = round(float(bs_price(flag, S, K, t_i, r, s_i, q)), 4)
                d = round(float(calc_delta(flag, S, K, t_i, r, s_i, q)), 4)
            except Exception:
                p, d = None, None
            price_row.append(p)
            delta_row.append(d)
        price_matrix.append(price_row)
        delta_matrix.append(delta_row)

    return jsonify(
        {
            "sigma_labels": sigma_labels,
            "t_labels": T_labels,
            "price_matrix": price_matrix,
            "delta_matrix": delta_matrix,
            "base_price": round(float(bs_price(flag, S, K, T, r, sigma, q)), 4),
        }
    )


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
        data = request.get_json(force=True)
        params = data.get("params", {})
        result = data.get("result", {})
        portfolio = data.get("portfolio", {})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    wb = openpyxl.Workbook()

    HDR_FILL = PatternFill("solid", fgColor="0D1626")
    HDR_FONT = Font(color="4E9EFF", bold=True, size=12)
    SUB_FILL = PatternFill("solid", fgColor="0D3A6B")
    SUB_FONT = Font(color="FFFFFF", bold=True)
    BOLD = Font(bold=True)
    CENTER = Alignment(horizontal="center")

    def hdr(ws, row, ncols, text, fill=HDR_FILL, font=HDR_FONT):
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=ncols)
        c = ws.cell(row=row, column=1, value=text)
        c.fill, c.font, c.alignment = fill, font, CENTER

    def row_vals(ws, r, vals, bold=False, fill=None):
        for ci, v in enumerate(vals, 1):
            c = ws.cell(row=r, column=ci, value=v)
            if bold:
                c.font = BOLD
            if fill:
                c.fill = fill

    # ── Sheet 1: Pricing Summary ──────────────────────────
    ws1 = wb.active
    ws1.title = "Pricing Summary"
    ws1.column_dimensions["A"].width = 22
    ws1.column_dimensions["B"].width = 18
    ws1.column_dimensions["C"].width = 36

    r = 1
    hdr(ws1, r, 3, "OTC Option Pricing Summary Report")
    r += 1
    hdr(ws1, r, 3, "Input Parameters", SUB_FILL, SUB_FONT)
    r += 1
    for label, val in [
        ("Option Type", "Call" if params.get("flag") == "c" else "Put"),
        ("Underlying (S)", params.get("S", "")),
        ("Strike (K)", params.get("K", "")),
        ("Maturity T (yr)", params.get("T", "")),
        ("Risk-Free r", params.get("r", "")),
        ("Volatility σ", params.get("sigma", "")),
        ("Div Yield q", params.get("q", "")),
    ]:
        row_vals(ws1, r, [label, val])
        r += 1

    r += 1
    hdr(ws1, r, 3, "Pricing Results", SUB_FILL, SUB_FONT)
    r += 1
    row_vals(ws1, r, ["BSM Price", result.get("bsm_price", "")])
    r += 1
    row_vals(ws1, r, ["Monte Carlo Price", result.get("mc_price", "")])
    r += 1

    r += 1
    hdr(ws1, r, 3, "Greeks at Current S", SUB_FILL, SUB_FONT)
    r += 1
    ws1.cell(row=r, column=1, value="Greek").font = BOLD
    ws1.cell(row=r, column=2, value="Value").font = BOLD
    ws1.cell(row=r, column=3, value="Description").font = BOLD
    r += 1
    greeks_data = result.get("greeks", {})
    for gname, gdesc in [
        ("Delta", "Price change per $1 move in underlying"),
        ("Gamma", "Delta change per $1 move in underlying"),
        ("Vega", "Price change per 1% move in volatility"),
        ("Theta", "Daily time decay"),
        ("Rho", "Price change per 1% move in rate"),
    ]:
        row_vals(ws1, r, [gname, greeks_data.get(gname, ""), gdesc])
        r += 1

    # ── Sheet 2: Price Curve Data ─────────────────────────
    chart_data = result.get("chart", {})
    if chart_data:
        ws2 = wb.create_sheet("Price Curve Data")
        for col, w in [("A", 12), ("B", 14), ("C", 14), ("D", 14)]:
            ws2.column_dimensions[col].width = w
        hdr(ws2, 1, 4, "Option Price Curve Data")
        for ci, h in enumerate(["Underlying S", "BSM Price", "Intrinsic Value", "Time Value"], 1):
            ws2.cell(row=2, column=ci, value=h).font = BOLD
        S_vals = chart_data.get("S_values", [])
        prices = chart_data.get("prices", [])
        intr = chart_data.get("intrinsic", [])
        for i, sv in enumerate(S_vals):
            p = prices[i] if i < len(prices) else ""
            iv = intr[i] if i < len(intr) else ""
            tv = (
                round(p - iv, 4)
                if isinstance(p, int | float) and isinstance(iv, int | float)
                else ""
            )
            ws2.append([sv, p, iv, tv])

    # ── Sheet 3: Greeks Curves Data ───────────────────────
    gc = chart_data.get("greek_curves", {})
    if gc:
        ws3 = wb.create_sheet("Greeks Curves Data")
        headers = ["Underlying S", "Delta", "Gamma", "Vega", "Theta", "Rho"]
        for ci, h in enumerate(headers, 1):
            col = openpyxl.utils.get_column_letter(ci)
            ws3.column_dimensions[col].width = 14
            ws3.cell(row=1, column=ci, value=h).font = BOLD
        for i, sv in enumerate(S_vals):
            row_data = [sv] + [
                gc[g][i] if i < len(gc.get(g, [])) else ""
                for g in ["Delta", "Gamma", "Vega", "Theta", "Rho"]
            ]
            ws3.append(row_data)

    # ── Sheet 4: Portfolio ────────────────────────────────
    positions = portfolio.get("positions", [])
    totals = portfolio.get("totals", {})
    if positions:
        ws4 = wb.create_sheet("Portfolio")
        ph = [
            "#",
            "Label",
            "Type",
            "S",
            "K",
            "T(yr)",
            "σ",
            "Qty",
            "Entry",
            "BSM Price",
            "MTM",
            "P&L",
            "Δ×Qty",
            "Γ×Qty",
            "V×Qty",
            "Θ×Qty",
            "ρ×Qty",
            "Margin",
        ]
        for ci, h in enumerate(ph, 1):
            col = openpyxl.utils.get_column_letter(ci)
            ws4.column_dimensions[col].width = 12
            ws4.cell(row=1, column=ci, value=h).font = BOLD
        for pi, pos in enumerate(positions, 1):
            wg = pos.get("w_greeks", {})
            ws4.append(
                [
                    pi,
                    pos.get("label", ""),
                    "Call" if pos.get("flag") == "c" else "Put",
                    pos.get("S", ""),
                    pos.get("K", ""),
                    pos.get("T", ""),
                    pos.get("sigma", ""),
                    pos.get("qty", ""),
                    pos.get("entry_price", ""),
                    pos.get("bsm_price", ""),
                    pos.get("mtm", ""),
                    pos.get("pnl", ""),
                    wg.get("Delta", ""),
                    wg.get("Gamma", ""),
                    wg.get("Vega", ""),
                    wg.get("Theta", ""),
                    wg.get("Rho", ""),
                    pos.get("margin", ""),
                ]
            )
        # Total row
        tr = len(positions) + 2
        for ci, val in enumerate(
            [
                "",
                "TOTAL",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                totals.get("mtm", ""),
                totals.get("pnl", ""),
                totals.get("Delta", ""),
                totals.get("Gamma", ""),
                totals.get("Vega", ""),
                totals.get("Theta", ""),
                totals.get("Rho", ""),
                totals.get("margin", ""),
            ],
            1,
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


def _snowball_local_vol(d, market):
    """A Dupire surface for a snowball request, or ``None`` for the flat path.

    The engines have taken ``local_vol=`` since Phase 2b, with 26 tests behind
    it, and no route ever passed it -- so the smile was reachable for European
    options and unreachable for the product it matters most to. A barrier
    payoff is concentrated at one spot level, so it depends on the volatility
    *there* rather than on an average, and a flat number cannot express that.

    Opt-in by the presence of ``underlying``. Absence keeps the flat path
    byte-for-byte, which matters: ``market.sigma`` is the project's default
    pricing basis and several anchors are pinned to it exactly.

    What the surface is anchored on is taken from the market being priced
    rather than asked for separately:

    ``rate``
        ``market.r``, so the surface and the discounting agree.
    ``spot``
        ``market.S``. The surface is scale-free -- it works in log-moneyness
        -- but its forward curve needs a level, and the level that matters is
        the one the note is being valued at.
    ``tail_drift``
        ``market.drift``, the caller's own ``r - q - basis``. Not the implied
        carry extrapolated: the options stop at about five months while the
        note runs one to three years, and carrying a -5% implied carry out to
        three years would drop the forward 14% on five-month evidence.

    Returns:
        ``(fitted, local_vol)`` or ``(None, None)``.
    """
    if "underlying" not in d:
        return None, None
    from dataclasses import replace as _replace

    # The market's rate wins over whatever the request said, because the
    # snowball is being priced in that market. Stated as a replace() on the
    # spec so the substitution is visible; the previous positional call passed
    # market.r into a `rate` slot and dropped the parsed one without saying so.
    spec = _replace(_surface_request(d), rate=market.r)
    return _cached_local_vol(spec, market.S, market.drift)


def _snowball_grid(d, local_vol, *, nodes=1600, steps=1008):
    """The grid for a snowball solve, which depends on which path it takes.

    The flat grid's budget does not carry over to a Dupire surface: the fitted
    variance term structure is rough enough that the same accuracy needs
    :data:`~otcpricer.engines.pde.LOCAL_VOL_PDE_CONFIG`'s 4800x8064. Defaulting
    a surface request to the flat grid would hand back a number whose error is
    an order of magnitude larger with nothing saying so.
    """
    if local_vol is None:
        return PDEConfig(
            space_nodes=int(d.get("space_nodes", nodes)),
            steps_per_year=int(d.get("steps_per_year", steps)),
        )
    return PDEConfig(
        space_nodes=int(d.get("space_nodes", LOCAL_VOL_PDE_CONFIG.space_nodes)),
        steps_per_year=int(d.get("steps_per_year", LOCAL_VOL_PDE_CONFIG.steps_per_year)),
    )


def _vol_model_note(local_vol, fitted):
    """What produced the volatility, for the response.

    A price is never reported without this. The flat and surface numbers are
    not interchangeable -- measured on the 510300 surface a 1-year up-and-out
    call moves 94% between them -- and a response that looked identical either
    way would invite them to be compared as if they were the same quantity.
    """
    if local_vol is None:
        return {
            "vol_model": "flat",
            "note": (
                "market.sigma, the project's default pricing basis. Pass "
                "'underlying' to price against a fitted Dupire surface instead."
            ),
        }
    return {
        "vol_model": "dupire-local",
        "underlying": fitted.underlying_code,
        "as_of": str(fitted.as_of),
        "source": fitted.source,
        "atm_vol": round(float(local_vol.atm_total_vol(1.0)), 8),
        "note": (
            "Dupire local volatility on an SSVI fit. market.sigma is not read "
            "on this path. A barrier payoff is concentrated at one spot level, "
            "so the smile matters here more than anywhere else in the project: "
            "measured on 510300, a 1-year up-and-out call moves 94% between "
            "the flat and surface answers."
        ),
    }


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
        fitted, local_vol = _snowball_local_vol(d, market)
        spread = float(d.get("spread", 0.02))
        n_paths = min(int(d.get("n_paths", 50000)), 500000)
    except (KeyError, TypeError, ValueError, OtcPricerError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    try:
        result = price_snowball_mc(
            terms, market, MonteCarloConfig(paths=n_paths, seed=42), local_vol=local_vol
        )
    except OtcPricerError as e:
        return jsonify({"error": str(e)}), 400

    mid = round(result.price, 6)
    return jsonify(
        {
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
            if result.expected_ko_time == result.expected_ko_time
            else None,
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
        }
    )


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
        fitted, local_vol = _snowball_local_vol(d, market)
        config = _snowball_grid(d, local_vol)
    except (KeyError, TypeError, ValueError, OtcPricerError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    try:
        result = price_snowball_pde(terms, market, config, local_vol=local_vol)
    except OtcPricerError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(
        {
            "price": round(result.price, 8),
            "method": "Crank-Nicolson FDM (barrier- and observation-aligned grid)",
            "space_nodes": result.space_nodes,
            "time_steps": result.time_steps,
            "grid_accuracy": {
                "estimated": False,
                "note": (
                    "This price carries the grid's own discretisation error, and "
                    "eight decimals do not mean eight digits. Measured on the "
                    "default 1600x1008: 0.34% on the reference trade, 0.32% at a "
                    "40% volatility, and 0.96% on the same note at the CSI 1000's "
                    "measured carry -- so it is contract-dependent and largest "
                    "where a desk actually works. The convergence is first order "
                    "with a ratio near 2.4 per halving. For a measured bound on "
                    "this specific request rather than these reference figures, "
                    "use /api/snowball_crosscheck, which solves at two grids and "
                    "reports the change plus a Richardson extrapolation."
                ),
            },
            "volatility": _vol_model_note(local_vol, fitted),
            "params": {
                "KO": terms.ko_barrier,
                "KI": terms.ki_barrier,
                "lockup_months": terms.lockup_months,
                "ki_strike_mode": str(terms.ki_strike_mode),
                "basis_pa": market.basis_pa,
            },
        }
    )


@app.route("/api/snowball_quote", methods=["POST"])
def snowball_quote():
    """Solve for the coupon that makes a snowball worth par.

    This is the question a desk actually asks. It does not ask what a note is
    worth at a 20% coupon; it asks what coupon the note can carry -- and that
    number *is* the quote. Every other snowball route takes ``coupon_pa`` as an
    input, so until now nothing here could answer it.

    Exact, not iterated
    -------------------
    The price is exactly affine in the coupon, because the knock-out and
    knock-in events depend only on the spot path against fixed barriers and
    never on the coupon, while every coupon-bearing term is linear in it. So
    ``price(c) = A*c + B`` and the answer is ``-B/A`` from a handful of solves
    rather than from a root search. ``affinity_residual`` in the response is
    that property *measured on this request*: a third solve at the midpoint,
    which must be collinear with the other two to machine precision.

    Why the grid is finer than the pricing routes'
    ----------------------------------------------
    The par coupon inherits the grid's first-order convergence. At 1600x1008
    it comes out 2.4e-04 high -- 1.4 Monte Carlo standard errors at 400,000
    paths, so visible rather than negligible. The default here is therefore
    3200x2016, four times the work of /api/snowball_pde. Pass ``space_nodes``
    and ``steps_per_year`` to trade accuracy for latency, knowing the cost.

    Request
    -------
    The /api/snowball payload, minus ``coupon_pa`` (it is the unknown and is
    ignored if present), plus:

    ``target``
        The price to solve to. 0 is a note struck at par.
    ``knocked_in``
        Solve for a note that has already breached its barrier.
    ``decompose``
        Also solve at a zero carry, to split the quote into what the
        volatility buys and what the futures discount buys. Costs a second
        solve set. Default true.
    ``verify_mc``
        Reprice at the solved coupon on the Monte Carlo, as an independent
        check that the finite-difference price itself is right. Costs 400,000
        paths. Default true.
    """
    try:
        d = request.get_json(force=True)
        terms, market = _parse_snowball_request(d)
        fitted, local_vol = _snowball_local_vol(d, market)
        config = _snowball_grid(
            d,
            local_vol,
            nodes=PAR_COUPON_PDE_CONFIG.space_nodes,
            steps=PAR_COUPON_PDE_CONFIG.steps_per_year,
        )
        target = float(d.get("target", 0.0))
        knocked_in = bool(d.get("knocked_in", False))
        want_split = bool(d.get("decompose", True))
        want_mc = bool(d.get("verify_mc", True))
        mc_paths = min(int(d.get("mc_paths", 400000)), 1000000)
    except (KeyError, TypeError, ValueError, OtcPricerError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    try:
        solution = solve_par_coupon(
            terms,
            market,
            config,
            target=target,
            knocked_in=knocked_in,
            local_vol=local_vol,
        )
        split = (
            decompose_par_coupon(terms, market, config, knocked_in=knocked_in, local_vol=local_vol)
            if want_split
            else None
        )
        verification = (
            verify_par_coupon_mc(
                solution,
                terms,
                market,
                MonteCarloConfig(paths=mc_paths, seed=42),
                knocked_in=knocked_in,
                local_vol=local_vol,
            )
            if want_mc
            else None
        )
    except OtcPricerError as e:
        return jsonify({"error": str(e)}), 400

    payload = {
        **solution.as_dict(),
        "decomposition": None if split is None else split.as_dict(),
        "mc_verification": verification,
        "grid": {
            "space_nodes": config.space_nodes,
            "steps_per_year": config.steps_per_year,
            "note": (
                "The par coupon converges first-order with the grid: 7.2902%, "
                "7.1441%, 7.1187%, 7.1069%, 7.1009% on the reference trade as "
                "it doubles from 400x252, for a Richardson limit near 7.0948%. "
                "At 1600x1008 the quote is 2.4e-04 high, which is 1.4 Monte "
                "Carlo standard errors -- hence the finer default here."
            ),
        },
        "volatility": _vol_model_note(local_vol, fitted),
        "params": {**terms.describe(), **market.as_dict()},
    }
    return jsonify(payload)


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
        fitted, local_vol = _snowball_local_vol(d, market)
        n_paths = min(int(d.get("n_paths", 200000)), 500000)
    except (KeyError, TypeError, ValueError, OtcPricerError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    # The refinement pair scales with the path: a Dupire surface needs the
    # finer grids for the same accuracy, and halving a grid that was already
    # too coarse would make the Richardson step measure the wrong thing.
    if local_vol is None:
        grids = (
            PDEConfig(space_nodes=800, steps_per_year=504),
            PDEConfig(space_nodes=1600, steps_per_year=1008),
        )
    else:
        grids = (
            PDEConfig(
                space_nodes=LOCAL_VOL_PDE_CONFIG.space_nodes // 2,
                steps_per_year=LOCAL_VOL_PDE_CONFIG.steps_per_year // 2,
            ),
            LOCAL_VOL_PDE_CONFIG,
        )

    try:
        mc = price_snowball_mc(
            terms, market, MonteCarloConfig(paths=n_paths, seed=42), local_vol=local_vol
        )
        coarse = price_snowball_pde(terms, market, grids[0], local_vol=local_vol)
        fine = price_snowball_pde(terms, market, grids[1], local_vol=local_vol)
    except OtcPricerError as e:
        return jsonify({"error": str(e)}), 400

    # First-order Richardson extrapolation, and the step-to-step change as the
    # bound on what remains.
    pde_change = abs(fine.price - coarse.price)
    pde_extrapolated = 2.0 * fine.price - coarse.price

    gap = abs(mc.price - pde_extrapolated)
    budget = pde_change + 2.0 * mc.ci95

    return jsonify(
        {
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
            "volatility": _vol_model_note(local_vol, fitted),
        }
    )


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
        fitted, local_vol = _snowball_local_vol(d, market)
        knocked_in = bool(d.get("knocked_in", False))
        config = _snowball_grid(d, local_vol)
    except (KeyError, TypeError, ValueError, OtcPricerError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    try:
        risk = snowball_risk_pde(terms, market, config, knocked_in=knocked_in, local_vol=local_vol)
    except OtcPricerError as e:
        return jsonify({"error": str(e)}), 400

    payload = risk.as_dict()
    payload["knocked_in"] = knocked_in
    payload["volatility"] = _vol_model_note(local_vol, fitted)
    payload["units"] = {
        "delta": "per index point",
        "gamma": "per index point squared",
        "vega": "per 1 percentage point of volatility",
        "theta": "per calendar day",
        "rho": "per 1 percentage point of rate",
        "carry_rho": "per 1 percentage point of carry (q + basis_pa)",
        "basis": "fraction of notional",
    }
    return jsonify(payload)


@app.route("/api/snowball_report", methods=["POST"])
def snowball_report_route():
    """The quote report for a snowball, as plain text.

    Everything this assembles was already reachable piece by piece. What the
    report adds is the *ordering*, and the ordering is the argument: carry
    coverage leads, because 48% of a one-year note's carry and 84% of a
    three-year note's is unmeasured while the carry itself more than doubles
    the quotable coupon. A reader who stops after the first screen should come
    away knowing the quote's weakest input rather than its headline.

    Each section costs solves, so each is opt-in -- with one exception. The
    carry section prints even with nothing measured, and says the carry is an
    assumption over the note's whole life. An omitted heading would read as
    "nothing to report here", which is the opposite of true.

    Request
    -------
    The /api/snowball payload, plus:

    ``underlying`` / ``source``
        Price against a fitted Dupire surface instead of ``sigma``.
    ``carry_as_of``
        Measure carry off the futures board on that date. Needs the Wind
        panel and an underlying with an index future (``IO``, ``HO``, ``MO``).
    ``carry_underlying``
        Which board to measure, when it differs from the pricing surface's
        underlying. Defaults to ``underlying``, then to ``MO``.
    ``tail``
        Required whenever ``carry_as_of`` is given: it governs the majority of
        the carry in the quote.
    ``quote``
        Solve for the par coupon and include it. Default true.
    ``mc``
        Include the outcome probabilities, which are path statistics and so
        come from the simulation rather than the grid. Default true.
    ``crosscheck``
        Run both risk engines and include the reconciliation. The most
        expensive section by far. Default false.
    """
    try:
        d = request.get_json(force=True)
        terms, market = _parse_snowball_request(d)
        fitted, local_vol = _snowball_local_vol(d, market)
        config = _snowball_grid(d, local_vol)
        want_quote = bool(d.get("quote", True))
        want_mc = bool(d.get("mc", True))
        want_crosscheck = bool(d.get("crosscheck", False))
        n_paths = min(int(d.get("n_paths", 200000)), 500000)
        carry_as_of = d.get("carry_as_of")
        carry_code = str(d.get("carry_underlying", d.get("underlying", "MO")))
        carry_tail = d.get("tail")
    except (KeyError, TypeError, ValueError, OtcPricerError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    measurement = None
    carry_error = None
    if carry_as_of is not None:
        try:
            futures_for = resolve_underlying(carry_code)
            if not futures_for.futures_root or not futures_for.index_code:
                raise MarketDataError(
                    f"{futures_for.code} has no index future sharing its expiry "
                    f"calendar; pass carry_underlying as IO, HO or MO"
                )
            if carry_tail is None:
                raise MarketDataError(
                    "carry_as_of needs 'tail' -- it governs 52% of a one-year "
                    "note's carry and 84% of a three-year note's, so there is "
                    "no default"
                )
            # Everything cheap and local first, then the panel. The panel is
            # the one thing a reader may not have, so an argument error must
            # not be reported as a missing data file.
            as_of_date = _panel_date(carry_as_of)
            tail_rule = TailCarry(str(carry_tail).lower())
            panel = _wind_panel()
            measurement = measure_carry(
                panel,
                futures_for.futures_root,
                futures_for.index_code,
                as_of_date,
                tail=tail_rule,
                rate=market.r,
            )
        except (MarketDataError, ValueError) as e:
            # Reported in the response rather than raised. A report whose carry
            # could not be measured is still worth printing -- that is exactly
            # the case its leading section is written for -- and failing the
            # whole request would hide the rest behind a missing data file.
            carry_error = str(e)

    try:
        risk = snowball_risk_pde(terms, market, config, local_vol=local_vol)
        solution = (
            solve_par_coupon(terms, market, config, local_vol=local_vol) if want_quote else None
        )
        split = (
            decompose_par_coupon(terms, market, config, local_vol=local_vol) if want_quote else None
        )
        simulated = (
            price_snowball_mc(
                terms, market, MonteCarloConfig(paths=n_paths, seed=42), local_vol=local_vol
            )
            if want_mc
            else None
        )
        comparison = (
            compare_snowball_risk(
                terms,
                market,
                pde_config=config,
                mc_config=MonteCarloConfig(paths=n_paths, seed=42),
                local_vol=local_vol,
                # The report already computed this, and a PDE risk is seven
                # solves. Without it this route paid for fourteen.
                pde=risk,
            )
            if want_crosscheck
            else None
        )
    except OtcPricerError as e:
        return jsonify({"error": str(e)}), 400

    text = snowball_report(
        terms,
        market,
        risk,
        carry=measurement,
        solution=solution,
        decomposition=split,
        mc=simulated,
        comparison=comparison,
    )
    return jsonify(
        {
            "report": text,
            "sections": {
                "carry": measurement is not None,
                "quote": solution is not None,
                "outcomes": simulated is not None,
                "reconciliation": comparison is not None,
            },
            "carry_error": carry_error,
            "volatility": _vol_model_note(local_vol, fitted),
        }
    )


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
        fitted, local_vol = _snowball_local_vol(d, market)
        mc_paths = min(int(d.get("n_paths", 200000)), 500000)
    except (KeyError, TypeError, ValueError, OtcPricerError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    try:
        # The grid is left to compare_snowball_risk, which already picks
        # LOCAL_VOL_PDE_CONFIG when a surface is supplied -- the flat grid's
        # error budget does not carry over to a Dupire surface, and that
        # choice belongs with the budget rather than here.
        comparison = compare_snowball_risk(
            terms,
            market,
            mc_config=MonteCarloConfig(paths=mc_paths, seed=42),
            local_vol=local_vol,
        )
    except OtcPricerError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({**comparison.as_dict(), "volatility": _vol_model_note(local_vol, fitted)})


# ─────────────────────────────────────────────────────────
# API: Shark Fin (Barrier Option) Pricing — Closed Form
# ─────────────────────────────────────────────────────────


@app.route("/api/shark_fin", methods=["POST"])
def shark_fin():
    """
    Price a shark fin (single-barrier note) on the engines, not inline.

    This endpoint used to evaluate the Rubinstein-Reiner formula in the route
    body, touching no engine, while cn_fdm.py solved a truncated grid for the
    "same" product. They were prices of *different contracts* -- continuous
    versus discrete monitoring -- and nothing compared them. Cross-checking
    against QuantLib found the inline formula was also wrong for **down**
    barriers, by 1.7 to 5.5 on prices of order 1 to 8, despite a comment
    claiming validation to 0.01%; it had only ever been checked on an
    up-and-out call.

    So ``monitoring`` is now required and explicit. A daily-observed knock-out
    is worth about 19% more than a continuously monitored one and a quarterly
    one about 2.4 times as much, so the frequency is part of the term sheet
    rather than a numerical setting.

    Every price is returned with whatever independent values exist for it: the
    analytic closed form when the barrier is continuous, and a Monte Carlo
    cross-check on request.

    Request JSON:
        S, K, T, r, sigma, q   - standard market inputs
        H                      - barrier level
        kind                   - 'uo', 'do', 'ui' or 'di'
        monitoring             - observations a year, or 'continuous'
        rebate                 - cash paid on a knock-out touch (default 0)
        participation          - multiplier on the payoff (default 1)
        flag                   - 'c' or 'p'; inferred from kind when omitted
        cross_check            - also run the Monte Carlo (default false)
        spread                 - bid/ask half-spread as a fraction (default 0)
    """
    try:
        d = request.get_json(force=True, silent=True) or {}
        S = float(d["S"])
        K = float(d["K"])
        T = float(d["T"])
        r = float(d["r"])
        sigma = float(d["sigma"])
        q = float(d.get("q", 0.0))
        H = float(d["H"])
        rebate = float(d.get("rebate", 0.0))
        participation = float(d.get("participation", 1.0))
        spread = float(d.get("spread", 0.0))
        cross_check = bool(d.get("cross_check", False))

        raw_kind = str(d.get("kind", "uo")).lower()
        if raw_kind not in ("uo", "do", "ui", "di"):
            raise ValueError(f"kind must be one of uo/do/ui/di, got {raw_kind!r}")
        kind = BarrierKind(raw_kind)

        # The conventional pairing, so a caller need not restate it: an up
        # barrier goes with a call, a down barrier with a put.
        default_flag = "c" if kind.is_up else "p"
        flag = str(d.get("flag", default_flag)).lower()

        monitoring = d.get("monitoring")
        if monitoring is None:
            raise ValueError(
                "monitoring is required: give observations a year (e.g. 252) or "
                "'continuous'. A daily-observed knock-out is worth about 19% "
                "more than a continuous one, so this is part of the contract"
            )
        if not (isinstance(monitoring, str) and monitoring.lower() == CONTINUOUS):
            monitoring = int(monitoring)
        else:
            monitoring = CONTINUOUS
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    try:
        terms = SharkFinTerms(
            S0=S,
            K=K,
            T=T,
            barrier=H,
            kind=kind,
            option=OptionType.parse(flag),
            monitoring=monitoring,
            rebate=rebate,
            participation=participation,
        )
        market = EquityMarket(S=S, r=r, sigma=sigma, q=q)
        grid = price_shark_fin_pde(terms, market, SHARK_FIN_PDE_CONFIG)
    except (OtcPricerError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    payload = grid.as_dict()
    payload["terms"] = terms.as_dict()
    payload["bid"] = round(grid.price * (1.0 - spread), 10)
    payload["ask"] = round(grid.price * (1.0 + spread), 10)

    # The independent values, where they exist.
    checks = {}
    if terms.is_continuous and rebate == 0.0:
        try:
            analytic = price_shark_fin_analytic(terms, market)
            checks["analytic"] = round(analytic, 10)
            checks["analytic_abs_diff"] = round(abs(grid.price - analytic), 10)
            checks["analytic_note"] = (
                "Rubinstein-Reiner closed form, verified against QuantLib's "
                "AnalyticBarrierEngine to 1e-12 across both barrier directions "
                "and both knock senses."
            )
        except OtcPricerError as e:
            checks["analytic_unavailable"] = str(e)
    else:
        checks["analytic_unavailable"] = (
            "no closed form for a discretely monitored barrier; the analytic "
            "value would be a different contract's price"
        )

    if cross_check:
        simulated = price_shark_fin_mc(
            terms,
            market,
            MonteCarloConfig(paths=200_000, steps_per_year=252, seed=7, antithetic=True),
        )
        checks["monte_carlo"] = simulated.as_dict()
        checks["monte_carlo_z"] = (
            round((simulated.price - grid.price) / simulated.std_error, 4)
            if simulated.std_error > 0.0
            else None
        )

    payload["cross_checks"] = checks
    payload["note"] = (
        "Priced on the finite-difference engine, with the barrier on a grid "
        "node and every observation on a time step. Monitoring frequency is "
        "part of the contract: discrete observation is worth strictly more "
        "than continuous."
    )
    return jsonify(payload)


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
        d = request.get_json(force=True)
        S = float(d["S"])
        T = float(d["T"])
        r = float(d["r"])
        q = float(d.get("q", 0.0))
        direction = str(d.get("direction", "long")).lower()
        notional = float(d.get("notional", 1.0))
        spread_pts = float(d.get("spread", 0.0))
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    F = S * np.exp((r - q) * T)
    cost_of_carry = S * (np.exp((r - q) * T) - 1.0)
    funding_cost = S * (np.exp(r * T) - 1.0)
    div_income = S * (np.exp(q * T) - 1.0)

    mid = round(float(F), 6)
    bid = round(mid - spread_pts, 6)
    ask = round(mid + spread_pts, 6)

    sign = 1 if direction == "long" else -1
    mtm_per_unit = sign * (F - S)  # theoretical P&L vs spot

    return jsonify(
        {
            "forward_price": mid,
            "bid": bid,
            "ask": ask,
            "cost_of_carry": round(float(cost_of_carry), 4),
            "funding_cost": round(float(funding_cost), 4),
            "dividend_income": round(float(div_income), 4),
            "mtm_per_unit": round(float(mtm_per_unit), 4),
            "notional_mtm": round(float(mtm_per_unit * notional), 4),
            "params": {
                "S": S,
                "T": T,
                "r": r,
                "q": q,
                "direction": direction,
                "notional": notional,
            },
        }
    )


#: One WindPanel per process. The panel is 34 MB of parquet and parsing it is
#: the expensive part, so it is built once and asked for dates -- which is the
#: whole reason WindPanelSource binds a date rather than fetch() taking one.
_WIND_PANEL = None


def _panel_date(as_of):
    """Parse a panel date, refusing anything else by shape.

    Raises:
        MarketDataError: on anything that is not ``YYYY-MM-DD``.
    """
    from datetime import date as _date

    try:
        year, month, day = (int(part) for part in str(as_of).split("-"))
        return _date(year, month, day)
    except (TypeError, ValueError):
        raise MarketDataError(f"as_of must look like YYYY-MM-DD, got {as_of!r}") from None


def _wind_panel():
    """The one WindPanel for this process.

    Raises:
        MarketDataError: when the panel is not configured, which is the normal
            case for a checkout without the institutional data. Reported as a
            missing configuration rather than surfaced as a solver failure.
    """
    global _WIND_PANEL
    from otcpricer.core.config import get_settings
    from otcpricer.marketdata.wind import WindPanel

    if _WIND_PANEL is None:
        directory = get_settings().data.wind_dir
        if not directory:
            raise MarketDataError(
                "this needs the Wind panel, which is not configured; set "
                "OTCPRICER_WIND_DIR to the directory holding the six files"
            )
        _WIND_PANEL = WindPanel(directory)
    return _WIND_PANEL


def _panel_rate_curve(as_of):
    """The real government curve for one panel date."""
    # The date is parsed before the panel is touched, and the order matters.
    # Python evaluates arguments left to right, so folding both into one call
    # reported a missing data file for a malformed date -- the caller's error
    # described as someone else's problem. Caught by tests/test_irs.py.
    moment = _panel_date(as_of)
    return rate_curve_from_panel(_wind_panel(), moment)


# ─────────────────────────────────────────────────────────
# API: Interest Rate Swap (IRS) — Simplified DCF
# ─────────────────────────────────────────────────────────


@app.route("/api/carry", methods=["POST"])
def carry():
    """Measure total carry off one root's listed index futures.

    A snowball coupon is paid for by volatility, the futures discount and cash
    management. ``basis_pa`` was a user input defaulting to zero, so the second
    of those was absent from every price this project produced -- while 725
    days of index futures sat in the panel being used only to validate
    forwards. This is that measurement, exposed.

    The board shows a drift, not a basis
    ------------------------------------
    ``log(F/S)/tau = r - q - basis`` is one observable against three unknowns:
    the same forward is consistent with ``q=3%, basis=0`` and with ``q=0,
    basis=3%``. So what is reported is total carry, ``r - drift``, which is the
    one quantity both engines read -- ``dV/dq`` and ``dV/dbasis_pa`` are the
    same derivative. Feed it back as ``q=0, basis_pa=carry`` with the same
    rate and the rate cancels exactly.

    Half the note's life is not measured
    ------------------------------------
    The listed contracts reach 0.4984 years on average, so a one-year note is
    48.5% measured and a three-year note 16.2%. ``tail`` is therefore a
    **required** field with no default: it governs the majority of the carry in
    any quote this feeds, and choosing it silently is how a model comes to
    hold a number nobody picked. ``measured_fraction_1y`` and
    ``measured_fraction_3y`` lead the payload for the same reason.

    Request
    -------
    ``underlying``
        ``IO``, ``HO`` or ``MO``. The futures root and settling index are
        taken from the contract registry rather than from the request, so
        there is one source for the pairing.
    ``as_of``
        ``YYYY-MM-DD``. Required: the panel holds 725 dates and there is no
        sensible default among them.
    ``tail``
        ``zero``, ``hold_last`` or ``explicit``. Required.
    ``tail_carry``
        The number, required for and only accepted with ``explicit``.
    ``rate``
        Rate the carry is quoted against. Only labels the number; the forwards
        are invariant to it under ``hold_last``.

    The panel is institutional data and is not committed to this repository,
    so a deployment without it gets a 400 naming the environment variable
    rather than a 500.
    """
    try:
        d = request.get_json(force=True)
        underlying = resolve_underlying(str(d.get("underlying", "MO")))
        if not underlying.futures_root or not underlying.index_code:
            raise MarketDataError(
                f"{underlying.code} has no index future sharing its expiry calendar, "
                f"so its carry cannot be measured this way; the CFFEX index options "
                f"IO, HO and MO can"
            )
        as_of = _panel_date(d["as_of"])
        tail = TailCarry(str(d.get("tail", "")).lower())
        tail_carry = d.get("tail_carry")
        rate = float(d.get("rate", DEFAULT_RATE))
    except ValueError as e:
        return jsonify(
            {
                "error": f"Invalid parameters: {e}",
                "hint": (
                    "tail is required and must be one of 'zero', 'hold_last' or "
                    "'explicit'. It governs 52% of a one-year note's carry and 84% "
                    "of a three-year note's, so there is deliberately no default."
                ),
            }
        ), 400
    except (KeyError, TypeError, MarketDataError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    # The tail/tail_carry combination is checked against the request alone, so
    # a caller who supplied both gets told that -- not that the panel is
    # missing. measure_carry would raise the same message, but only after its
    # first argument had already been evaluated.
    if tail is TailCarry.EXPLICIT and tail_carry is None:
        return jsonify(
            {"error": "TailCarry.EXPLICIT means the caller supplies the number; pass tail_carry"}
        ), 400
    if tail is not TailCarry.EXPLICIT and tail_carry is not None:
        return jsonify(
            {
                "error": (
                    f"tail_carry={tail_carry} was given alongside tail={tail}, which "
                    f"computes its own. Two sources for one number is how a model ends "
                    f"up quoting a carry nobody chose -- pass tail='explicit' to supply it."
                )
            }
        ), 400

    try:
        panel = _wind_panel()
        measurement = measure_carry(
            panel,
            underlying.futures_root,
            underlying.index_code,
            as_of,
            tail=tail,
            tail_carry=None if tail_carry is None else float(tail_carry),
            rate=rate,
        )
    except MarketDataError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(
        {
            **measurement.as_dict(),
            "underlying": underlying.code,
            "average_carry_1y": round(measurement.average_carry_to(1.0), 8),
            "average_carry_3y": round(measurement.average_carry_to(3.0), 8),
            "feed_as": {
                "q": 0.0,
                "basis_pa": round(measurement.average_carry_to(1.0), 8),
                "r": rate,
                "note": (
                    "For a one-year note. Pass these three into any snowball "
                    "route and its forward reproduces the futures board over "
                    "the measured part of the curve."
                ),
            },
        }
    )


@app.route("/api/irs", methods=["POST"])
def irs():
    """
    Price a plain vanilla interest rate swap off a discount curve.

    Pay-fixed / receive-float, NPV from the fixed payer's perspective:
    ``NPV = PV(float) - PV(fixed)``.

    The floating leg resets off the curve, period by period, at the simply
    compounded forward rate. That matters twice over: it is what a float leg
    actually pays, and it is the only convention under which the leg prices to
    par -- which the response reports as ``float_leg_prices_to_par`` so the
    pricing can be checked rather than trusted.

    The previous version of this endpoint reported a DV01 of
    ``notional * dt * df(T) * 1bp``, using only the final discount factor
    where a fixed leg's sensitivity is the whole annuity. At T=2, freq=2, 2%
    on a million notional that is 48.04 against a true 195.07, while
    bump-and-reprice of its own NPV gave 0.15 -- three unrelated numbers,
    because its floating leg paid the same constant every period and so never
    responded to a curve at all. See otcpricer.products.irs.

    Request JSON:
        notional    - notional principal (default 1,000,000)
        fixed_rate  - annual fixed rate, e.g. 0.035
        T           - maturity in years
        freq        - payments per year (1, 2, 4, 12; default 4)
        r_disc      - flat rate to build the curve from, when no panel date is
                      given. Falls back to float_rate, then to 2%.
        as_of       - a Wind panel date (YYYY-MM-DD) to take the real
                      government curve from instead of a flat rate
        spread      - bid/ask half-spread as an absolute NPV (default 0)
    """
    try:
        d = request.get_json(force=True, silent=True) or {}
        notional = float(d.get("notional", 1_000_000))
        fixed_rate = float(d["fixed_rate"])
        T = float(d["T"])
        freq = int(d.get("freq", 4))
        spread = float(d.get("spread", 0.0))
        as_of = d.get("as_of")
        flat_rate = float(d.get("r_disc", d.get("float_rate", 0.02)))
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    try:
        curve = _panel_rate_curve(str(as_of)) if as_of else flat_curve(flat_rate)
        result = price_swap(
            curve, notional=notional, fixed_rate=fixed_rate, maturity=T, frequency=freq
        )
    except OtcPricerError as e:
        return jsonify({"error": str(e)}), 400

    payload = result.as_dict()
    payload["bid"] = round(result.npv - spread, 2)
    payload["ask"] = round(result.npv + spread, 2)
    payload["curve"] = curve.as_dict()
    payload["params"] = {
        "notional": notional,
        "fixed_rate": fixed_rate,
        "T": T,
        "freq": freq,
    }
    payload["note"] = (
        "dv01 is the fixed-leg annuity and agrees with bump-and-reprice. The "
        "float leg resets at the curve's simply compounded forwards, so "
        "float_leg_prices_to_par should be true for any curve shape."
    )
    return jsonify(payload)


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
        d = request.get_json(force=True)
        product_type = str(d.get("product_type", "snowball")).lower()
        S0 = float(d["S0"])
        T = float(d["T"])
        r = float(d["r"])
        sigma = float(d["sigma"])
        q = float(d.get("q", 0.0))
        # The shark fin reads KO_pct here; the snowball reads it inside
        # _parse_snowball_request, which is now the only place that defaults
        # it. Two defaults for one field used to live here -- 1.05 locally and
        # 1.03 in the parser -- so a request omitting KO_pct was priced at
        # 1.03 and reported at 1.05.
        KO_pct = float(d.get("KO_pct", 1.03))
        rebate = float(d.get("rebate", 0.0))
        K_strike = float(d.get("K_strike", S0))
        mc_paths = min(int(d.get("mc_paths", 50000)), 200000)
        calc_g = bool(d.get("calc_greeks", True))
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    if product_type not in ("snowball", "shark_fin"):
        return jsonify({"error": "product_type must be 'snowball' or 'shark_fin'"}), 400
    if T <= 0 or sigma <= 0 or S0 <= 0:
        return jsonify({"error": "S0, T and sigma must be positive"}), 400

    # What the response echoes, and where it comes from. The old version
    # rebuilt this block by re-reading the request body, which is how it came
    # to report a barrier the engine had not priced. Everything below is read
    # off the objects that were actually priced instead.
    echoed: dict[str, object] = {}

    try:
        if product_type == "snowball":
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
            echoed = {**risk_terms.describe(), **risk_market.as_dict()}

        if product_type == "snowball" and not calc_g:
            # The last legacy snowball price. It ignored the lock-up, the
            # knock-in strike mode and the basis, and its PDE and Monte Carlo
            # struck the knock-in put at different levels -- so the two halves
            # of one class were pricing different contracts. Reviewed engine
            # now, same as every other snowball route.
            price = price_snowball_pde(
                risk_terms,
                risk_market,
                PDEConfig(space_nodes=1600, steps_per_year=1008),
            ).price
            greeks_out = {}
        elif product_type == "snowball":
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
        elif product_type == "shark_fin":
            # Migrated off the legacy class. The legacy grid ran at
            # M_per_year=252, so daily observation is the contract it was
            # actually pricing -- that is preserved here, but stated rather
            # than implied, because a request that cannot express its own
            # monitoring frequency is the ambiguity this migration existed to
            # remove. On the legacy test case the daily note is worth 43.02
            # against 35.17 continuous -- a 22% difference the old code
            # conflated.
            monitoring = int(d.get("monitoring", 252))
            fin_terms = SharkFinTerms(
                S0=S0,
                K=K_strike,
                T=T,
                barrier=S0 * KO_pct,
                kind=BarrierKind.UP_AND_OUT,
                option=OptionType.CALL,
                monitoring=monitoring,
                rebate=rebate,
            )
            fin_market = EquityMarket(S=S0, r=r, sigma=sigma, q=q)
            fin_result = price_shark_fin_pde(fin_terms, fin_market, SHARK_FIN_PDE_CONFIG)
            price = fin_result.price
            echoed = {**fin_terms.as_dict(), **fin_market.as_dict()}
            greeks_out = (
                {
                    # The legacy three-method output is gone. Those numbers came
                    # from the same code whose snowball deltas were wrong in
                    # sign; none of them was ever validated for a barrier.
                    "resimulation_fd": None,
                    "pathwise": None,
                    "likelihood": None,
                    "notes": {
                        "engine": (
                            "Migrated to engines/barrier.py. Greeks are not yet "
                            "exposed for this product; the legacy three-method "
                            "output was removed rather than carried forward, "
                            "because it was never validated for a barrier and "
                            "came from the code whose snowball deltas were "
                            "wrong in sign."
                        ),
                        "monitoring": (
                            f"{monitoring} observations a year, which is what the "
                            f"legacy grid priced. Pass 'monitoring' to change it, "
                            f"or use /api/shark_fin for the full interface."
                        ),
                    },
                }
                if calc_g
                else {}
            )
        else:  # pragma: no cover - product_type is validated above
            raise AssertionError(f"unreachable product_type {product_type!r}")

    except OtcPricerError as e:
        # A structurally impossible trade is the caller's error, not a crash.
        # It used to land in the blanket handler below and come back as a 500.
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"CN-FDM solver error: {e}"}), 500

    # ── Continuous-barrier closed form, as a cross-check ──
    # Was computed inline here with the same reflection sign the old
    # /api/shark_fin used, which is correct for an up-and-out call and wrong
    # for a down barrier. Now delegated to the verified implementation, which
    # agrees with QuantLib's AnalyticBarrierEngine to 1e-12 in both
    # directions. It prices the *continuous* contract, so it sits below the
    # daily-observed grid price rather than beside it.
    haug_price = None
    if product_type == "shark_fin":
        try:
            continuous_terms = SharkFinTerms(
                S0=S0,
                K=K_strike,
                T=T,
                barrier=S0 * KO_pct,
                kind=BarrierKind.UP_AND_OUT,
                option=OptionType.CALL,
                monitoring=CONTINUOUS,
            )
            haug_price = round(
                price_shark_fin_analytic(
                    continuous_terms, EquityMarket(S=S0, r=r, sigma=sigma, q=q)
                ),
                6,
            )
        except Exception:
            haug_price = None

    return jsonify(
        {
            "price": round(float(price), 6),
            "haug_price": haug_price,
            "pde_grid": greeks_out.get("pde_grid"),
            "resimulation_fd": greeks_out.get("resimulation_fd", {}),
            "pathwise": greeks_out.get("pathwise"),
            "likelihood": greeks_out.get("likelihood"),
            "greek_notes": greeks_out.get("notes"),
            "method": (
                "Crank-Nicolson FDM"
                if product_type == "shark_fin"
                else "Crank-Nicolson FDM + Monte Carlo cross-check"
            ),
            "product_type": product_type,
            # Read off the objects that were priced, not re-parsed from the
            # request. That is what makes the echo trustworthy, and it is also
            # how lockup_months, ki_strike_mode and basis_pa reach a response
            # for the first time -- they have been priced since Phase 1 and
            # were invisible here.
            "params": {
                **echoed,
                "mc_paths": mc_paths,
                "echo_note": (
                    "Read from the priced SnowballTerms/EquityMarket rather "
                    "than re-parsed from the request. The previous version "
                    "re-read the body with a different KO_pct default, so a "
                    "request omitting it was priced at 1.03 and reported at "
                    "1.05."
                ),
            },
        }
    )


if __name__ == "__main__":
    print("=" * 60)
    print("  Option Pricing Tool -- Web Application")
    print("  Open in browser: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)
