# 📈 Option Pricing Tool

> OTC Derivatives · European Vanilla Options · Black-Scholes-Merton + Monte Carlo  
> GitHub Portfolio Project | OTC Derivatives Trading Assistant

---

## Features

- **Black-Scholes-Merton pricing** with continuous dividend yield (vanilla OTC options)
- **Monte Carlo simulation** (risk-neutral measure, 100,000 paths)
- **Full Greeks**: Delta / Gamma / Vega / Theta / Rho
- **Implied Volatility solver** (Newton-Raphson iteration)
- **Web UI** (Flask + ECharts): interactive price & Greeks charts in browser
- **CLI demo + interactive mode**: `main.py` for terminal use

---

## Project Structure

```
期权定价工具/
├── app.py               # Flask web backend (API + serve HTML)
├── bs_pricing.py        # BSM pricing + Monte Carlo + IV solver
├── greeks_calc.py       # Greeks calculation module (Delta/Gamma/Vega/Theta/Rho)
├── visualization.py     # Matplotlib charts (price curve + Greeks curves)
├── main.py              # CLI entry point (demo mode + interactive mode)
├── templates/
│   └── index.html       # Single-page web frontend (ECharts)
├── requirements.txt     # Dependencies
└── README.md
```

---

## Tech Stack

| Library | Min Version | Role |
|---|---|---|
| `py_vollib` | ≥ 1.0.1 | Core pricing: BSM, Greeks, IV (analytical) |
| `numpy` | ≥ 1.21 | Monte Carlo simulation, numerical computation |
| `pandas` | ≥ 1.3 | Greeks summary table output |
| `matplotlib` | ≥ 3.4 | Price & Greeks chart (PNG export) |
| `flask` | ≥ 2.0 | Web application backend |

---

## Installation

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install py_vollib numpy pandas matplotlib flask
```

> ⚠️ If `py_vollib` fails to install, try:
> ```bash
> pip install py_vollib --no-build-isolation
> ```

---

## Quick Start

### Option A — Web Interface (Recommended)

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

**Features:**
- Input any option parameters (S, K, T, r, σ, q, Call/Put)
- Click **Calculate** — instantly see BSM price, Monte Carlo price, all Greeks
- **Price Curve tab**: BSM price vs intrinsic value (ECharts interactive)
- **Greeks Curves tab**: 5 mini-charts for Delta/Gamma/Vega/Theta/Rho
- **IV Solver**: enter market price → get implied volatility
- Press `Enter` to calculate quickly

### Option B — CLI Mode

```bash
python main.py
```

```
  1. Demo mode  — auto-run CSI 500 index option test cases + save PNG charts
  2. Interactive — manually enter parameters, real-time output
```

---

## Sample Output (CLI Demo)

### Test Case 1: ATM Call (CSI 500 Index Option)

| Param | Value | Notes |
|---|---|---|
| S | 5500 | Underlying (CSI 500 index, points) |
| K | 5500 | At-the-money |
| T | 0.25 yr | ~3 months |
| r | 2% | 1-year T-bond yield reference |
| σ | 20% | CSI 500 historical volatility |
| q | 1% | Index dividend yield |

```
──────────────────────────────────────────────────────────────
  Option Type : Call Option
  S = 5500   K = 5500   T = 0.25 yr   r = 2.00%   σ = 20.00%   q = 1.00%
──────────────────────────────────────────────────────────────

  [BSM Theoretical Price]  = 303.xxxx
  [Monte Carlo Price]      = 303.xxxx  (vs BSM Δ < 2.0000)

  [Greeks Summary]

  Greek      Value     Business Meaning
  Delta     0.4982     Direction risk — hedge ratio
  Gamma     0.000xxx   Delta convexity — re-hedge cost
  Vega      xx.xxxx    P&L per 1% vol move
  Theta     -x.xxxx    Daily time decay (buyer cost)
  Rho       xx.xxxx    P&L per 1% rate move
```

### Saved Charts

| File | Content |
|---|---|
| `option_price_curve.png` | BSM price + intrinsic value + time value fill |
| `greeks_curve.png` | 5-panel Greeks vs underlying price |

---

## Model Reference

### Black-Scholes-Merton (with dividend yield q)

$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$
$$P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)$$
$$d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2) T}{\sigma \sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

### Greeks

| Greek | Formula | Trading Use |
|---|---|---|
| **Delta** | ∂V/∂S | Daily delta-hedging (buy/sell underlying) |
| **Gamma** | ∂²V/∂S² | Re-hedge frequency & convexity cost |
| **Vega** | ∂V/∂σ | Volatility risk — hedge with listed options |
| **Theta** | ∂V/∂t | Time decay — buyer's daily holding cost |
| **Rho** | ∂V/∂r | Rate sensitivity — material for long-dated options |

---

## Parameter Validation

| Param | Valid Range | Error if violated |
|---|---|---|
| S (underlying) | > 0 | "S must be greater than 0" |
| K (strike) | > 0 | "K must be greater than 0" |
| T (expiry, yr) | > 0 | "T must be greater than 0" |
| σ (volatility) | (0, 2] | "sigma must be in range (0, 2]" |
| q (dividend) | ≥ 0 | "q must be >= 0" |

---

## Target Roles

- OTC derivatives trading assistant / structuring
- Asset management / financial engineering
- Quantitative research internship

---

## License

MIT License — for personal learning and portfolio use

