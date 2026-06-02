# 📈 OTC Option Pricing & Risk Analytics Tool

> **场外衍生品定价与风险管理工具** | OTC Derivatives · Vanilla + Structured Products · BSM + Monte Carlo  
> Campus Recruitment Portfolio Project | 校招实战项目 — 场外衍生品交易助理方向

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![ECharts](https://img.shields.io/badge/ECharts-5.x-orange.svg)](https://echarts.apache.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Project Overview

A professional-grade **OTC derivatives pricing and risk analytics web application** built for the trading desk workflow. Covers the full lifecycle from product quoting to portfolio exposure monitoring — directly aligned with the daily responsibilities of an **OTC Derivatives Trading Assistant (场外衍生品交易助理)**.

**Key selling points for campus recruitment:**
- Implements real desk tools: BSM pricing, Monte Carlo, implied vol, barrier pricing, snowball autocallable
- Portfolio-level Greeks aggregation (not just single-option pricing)
- Stress testing / sensitivity matrices across vol and time shifts
- Structured product quoting (bid/mid/ask) with spread overlay
- CN/EN bilingual interface

---

## ✨ Features

### 📈 Option Pricing (Vanilla)
- **Black-Scholes-Merton** closed-form pricing with continuous dividend yield
- **Monte Carlo simulation** (risk-neutral GBM, up to 200k paths)
- **Full Greeks**: Δ Delta / Γ Gamma / ν Vega / Θ Theta / ρ Rho
- **Implied Volatility solver** (Newton-Raphson, < 1ms per solve)
- **Interactive charts**: ECharts price curve + 5 Greeks curves
- **Unit auto-conversion**: input T in days or years, σ/r in % or decimal

### 🔬 Structured Products Greeks (v2.0 — Glasserman & Broadie)
- **3-Method MC Greeks** engine for Snowball & Shark Fin barrier options
- **Resimulation FD**: Central difference with common random numbers — most robust
- **Pathwise Differentiation**: SDE path derivative + conditional MC barrier smoothing
- **Likelihood Ratio**: Score Function method — naturally handles discontinuous payoffs
- Single-run parallel output → grouped dicts for cross-validation & risk control
- CN-FDM PDE (Crank-Nicolson + Rannacher damping) for independent price validation

### 💰 Structured Products Quoting
| Product | Pricing Model | Greeks |
|---|---|---|
| **Snowball / Autocallable (雪球)** | Monte Carlo path simulation + CN-FDM PDE (Crank-Nicolson) | 3-Method MC Greeks (Resimulation FD / Pathwise / Likelihood Ratio) |
| **Shark Fin / Barrier Option (鲨鱼鳍)** | Rubinstein-Reiner closed-form (UOC/DOP) + CN-FDM PDE validation | 3-Method MC Greeks |
| **OTC Equity Forward (场外远期)** | Cost-of-carry: F = S·e^((r−q)T) with funding/dividend decomposition | Analytical Δ = ±e^(−qT) |
| **Interest Rate Swap (利率互换)** | Flat-curve DCF; full cash flow schedule + DV01 sensitivity | DV01 per +1bp |

All products output **Bid / Mid / Ask** with configurable spread (bps).

> 🆕 **v2.0 Upgrade**: CN-FDM now outputs **3 parallel Greeks** per Glasserman & Broadie (2004):
> ① **Resimulation FD** — central difference + common random numbers for variance reduction
> ② **Pathwise Differentiation** — SDE path derivative with conditional MC barrier smoothing
> ③ **Likelihood Ratio** — Score Function method, naturally handles discontinuous barrier payoffs
> All three sets computed in a single run, returned as grouped dicts for cross-validation.

### 📊 Portfolio Exposure Monitor
- Add multiple vanilla positions with qty / entry price
- **Aggregated Greeks**: net Δ, Γ, ν, Θ, ρ across entire portfolio
- **MTM valuation** and per-position P&L (mark-to-market vs entry)
- **Margin estimation** (notional × fixed% + |Δ| × variable%)
- Structured product positions: full Greeks via CN-FDM (Δ/Γ/ν/Θ) for snowball and shark fin, analytical delta for forwards
- **Portfolio CSV export**: one-click unified report of all positions (vanilla + structured) + totals row

### 🔬 Analytics Panels (inline in Pricing tab)
- **Volatility Smile / Skew**: input market prices at different strikes → compute IV → plot smile curve
- **MC P&L Distribution**: histogram of terminal payoffs with mean, median, 95% CI
- **Sensitivity / Stress Test**: 5×5 price and delta matrix across ±20% vol × ±60 day shifts
- **Structured Sensitivity**: product-specific stress grids (Snowball / SharkFin: spot×vol, Forward: spot×rate, IRS: parallel rate shifts ±200bps)

### 🌐 CN/EN Bilingual Interface
- Full Chinese / English language toggle
- All labels, metrics, and product descriptions translated

---

## 🗂 Project Structure

```
期权定价工具/
├── app.py               # Flask backend — all pricing API endpoints
├── bs_pricing.py        # BSM pricing + Monte Carlo simulation + IV solver
├── greeks_calc.py       # Greeks calculation (Delta/Gamma/Vega/Theta/Rho)
├── visualization.py     # Matplotlib charts (CLI use, PNG export)
├── main.py              # CLI entry point (demo mode + interactive mode)
├── templates/
│   └── index.html       # Single-page frontend (ECharts, pure HTML/CSS/JS)
├── requirements.txt     # Python dependencies
└── README.md
```

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/calculate` | POST | BSM + MC pricing, full Greeks, price curve data |
| `/api/greeks_curve` | POST | Greeks vs spot arrays for ECharts rendering |
| `/api/iv` | POST | Implied volatility solver |
| `/api/portfolio` | POST | Portfolio-level Greeks aggregation + MTM |
| `/api/mc_distribution` | POST | MC payoff distribution histogram data |
| `/api/sensitivity` | POST | Price/delta sensitivity matrix |
| `/api/vol_smile` | POST | Implied vol from market prices (smile curve) |
| `/api/smile_presets` | POST | Auto-fill smile table from BSM prices |
| `/api/snowball` | POST | Snowball autocallable MC pricing |
| `/api/shark_fin` | POST | Shark fin barrier option pricing (UOC/DOP/UIC/DIC) |
| `/api/forward` | POST | OTC equity forward pricing |
| `/api/irs` | POST | Interest rate swap NPV + cash flows |
| `/api/cn_fdm` | POST | CN-FDM PDE pricing + **3-method MC Greeks** (Resimulation FD/Pathwise/Likelihood) — independently validated |

---

## 🛠 Tech Stack

| Library | Version | Role |
|---|---|---|
| `py_vollib` | ≥ 1.0.1 | BSM analytical pricing, Greeks, IV solver |
| `numpy` | ≥ 1.21 | Monte Carlo GBM simulation, vectorized math |
| `pandas` | ≥ 1.3 | Portfolio aggregation, data structuring |
| `scipy` | ≥ 1.7 | Statistical distributions, optimization |
| `flask` | ≥ 2.0 | REST API backend |
| `matplotlib` | ≥ 3.4 | CLI chart output (PNG) |
| **ECharts 5** | CDN | Interactive web charts (no npm required) |

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/chenseldon/Option-Pricing-Tool.git
cd Option-Pricing-Tool

# Install dependencies
pip install -r requirements.txt
```

> ⚠️ If `py_vollib` fails on Windows, try:
> ```bash
> pip install py_vollib --no-build-isolation
> ```

---

## 🚀 Quick Start

### Web Interface (Recommended)

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

**Workflow:**
1. Select **Product Type** from the left sidebar (Call / Put / Snowball / Shark Fin / OTC Forward / IRS)
2. Input parameters — T accepts days or years, σ/r accept % or decimal
3. Click **▶ Calculate** (or **▶ Price Product** for structured products)
4. View results, Greeks, charts on the right panel
5. Add positions to Portfolio for combined exposure monitoring
6. Expand inline analytics panels for smile curve, MC distribution, stress test
7. Export CSV report

### CLI Mode

```bash
python main.py
```

```
  1. Demo mode  — auto-run CSI 500 ATM call test + save PNG charts
  2. Interactive — manually enter parameters
```

---

## 📐 Pricing Models

### Black-Scholes-Merton (with continuous dividend yield q)

$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$
$$P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)$$
$$d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)\,T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

### Greeks Summary

| Greek | Formula | Desk Use |
|---|---|---|
| **Delta** | ∂V/∂S | Delta-hedging ratio — buy/sell underlying to stay flat |
| **Gamma** | ∂²V/∂S² | Re-hedge cost — convexity; high gamma → frequent rebalancing |
| **Vega** | ∂V/∂σ | P&L per 1% vol move — hedge with listed options/variance swaps |
| **Theta** | ∂V/∂t | Daily time decay — buyer's carry cost, seller's premium income |
| **Rho** | ∂V/∂r | Rate sensitivity — material for long-dated or deep ITM options |

### Snowball Autocallable (Monte Carlo)

$$\text{Payoff} = \begin{cases} \text{Coupon} \times t_i & \text{if } S_{t_i} \geq KO \\ -\max(0, 1 - S_T/KI) \times \text{Notional} & \text{if } S_T \leq KI \\ 0 & \text{otherwise} \end{cases}$$

Monte Carlo samples 10k–200k GBM paths; outputs KO/KI probabilities and 95% CI.

### Shark Fin (Rubinstein-Reiner Closed Form)

Up-and-Out Call / Down-and-Out Put priced via 4-term closed-form barrier formula with optional cash rebate on breach. Reduces to vanilla BSM when H is far from S.

### OTC Forward

$$F = S \cdot e^{(r-q)T}$$

Decomposed into funding cost and dividend income. MTM = discounted difference between current forward price and contracted price.

### Interest Rate Swap (Flat Curve DCF)

$$NPV = PV_{\text{float}} - PV_{\text{fixed}} = \sum_{i} \frac{L \cdot r_{flt} \cdot \Delta t_i}{(1+r_{disc})^{t_i}} - \sum_{i} \frac{L \cdot r_{fix} \cdot \Delta t_i}{(1+r_{disc})^{t_i}}$$

DV01 (Dollar Value of 01) = NPV change per +1bp parallel rate shift.

---

## 📊 Sample Output

### ATM Call Option (CSI 500)

| Input | Value |
|---|---|
| S (spot) | 5,500 |
| K (strike) | 5,500 (ATM) |
| T | 90 days (0.25 yr) |
| r | 2.0% |
| σ | 20.0% |
| q | 1.0% |

| Output | BSM | Monte Carlo |
|---|---|---|
| Price | ~303 | ~303 ± 2 |
| Delta | 0.498 | — |
| Gamma | 0.00032 | — |
| Vega | 6.64 (per 1%) | — |
| Theta | −2.08 (per day) | — |

### Snowball (Typical China OTC)

| Input | Value |
|---|---|
| S₀ | 5,000 |
| KO level | 103% |
| KI level | 75% |
| Coupon | 20% p.a. |
| Tenor | 1 year |

Output: Mid ≈ 8–12% of notional, KO prob ≈ 50–70%, KI prob ≈ 15–25%

---

## 🎓 Target Roles

This project directly targets the following campus recruitment positions:

- **场外衍生品交易助理** (OTC Derivatives Trading Assistant)
- **结构化产品助理** (Structured Products Associate)
- **金融工程 / 量化研究实习** (Financial Engineering / Quant Research Intern)
- **资管风控** (Asset Management Risk Control)

**Business scenarios covered:**
- Quoting vanilla and structured products from market parameters
- Greeks-based delta hedging and re-hedge cost estimation
- Portfolio exposure monitoring and MTM valuation
- Stress testing across vol and rate scenarios
- Implied volatility smile construction from market prices

---

## ⚠️ Parameter Validation

| Param | Valid Range | Notes |
|---|---|---|
| S, K | > 0 | Positive prices only |
| T | > 0 | Days or years (auto-converted) |
| σ | (0%, 500%] | Annualized volatility |
| q | ≥ 0 | Continuous dividend yield |
| KO level | > 100% (snowball) | Knock-out above initial price |
| H barrier | > S for UOC | Barrier above current spot |

---

## 📄 License

MIT License — free for personal learning, portfolio, and interview demonstration use.

---

*Built with Python + Flask + ECharts · 2024–2025*

