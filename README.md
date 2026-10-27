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

### 🔬 Structured Products Greeks
- **PDE grid differentiation** — delta and gamma read straight off the
  Crank-Nicolson solution surface. The solver already evaluates the value at
  every node, so they cost nothing beyond the price and carry no sampling noise
- **Theta from two adjacent time steps** of the same solve, which keeps the
  observation schedule fixed (bumping the tenor would delete a knock-out date)
- **Monte Carlo resimulation with common random numbers** as an independent
  cross-check, with a per-Greek verdict against a measured error budget
- **Spot delta, not re-struck delta**: barriers stay where they were struck
  while the market moves, which is what makes it a hedge ratio
- **Seasoned trades** can be revalued and risked, including after a knock-in

### 💰 Structured Products Quoting
| Product | Pricing Model | Greeks |
|---|---|---|
| **Snowball / Autocallable (雪球)** | Monte Carlo path simulation + Crank-Nicolson PDE, agreeing to ~2 MC confidence widths | PDE grid differentiation, cross-checked against MC resimulation |
| **Shark Fin / Barrier Option (鲨鱼鳍)** | Rubinstein-Reiner closed-form (UOC/DOP) + CN-FDM PDE validation | 3-Method MC Greeks |
| **OTC Equity Forward (场外远期)** | Cost-of-carry: F = S·e^((r−q)T) with funding/dividend decomposition | Analytical Δ = ±e^(−qT) |
| **Interest Rate Swap (利率互换)** | Flat-curve DCF; full cash flow schedule + DV01 sensitivity | DV01 per +1bp |

All products output **Bid / Mid / Ask** with configurable spread (bps).

> **Structured-product Greeks** come from differencing the Crank-Nicolson
> solution surface, cross-checked against Monte Carlo resimulation with common
> random numbers and anchored to two closed-form cases. See
> [*Model Validation*](#-model-validation) and
> [`docs/models/greeks.md`](docs/models/greeks.md).
>
> Pathwise differentiation is deliberately **not** offered: it requires a
> Lipschitz payoff, and a snowball's payoff jumps at both barriers. The
> reasoning is documented rather than papered over.

### 🌋 Implied Volatility Surface (real market data)

Fits an **arbitrage-free SSVI surface** (Gatheral–Jacquier) to a live or
snapshotted SSE ETF option chain:

| Step | What happens |
|---|---|
| Fetch | Whole expiry per request — a four-expiry surface costs four calls |
| Forward | Implied from put-call parity; the rate is an **input**, not a fitted parameter |
| Implied vol | Black-76 via the self-built BSM, then six screens with every rejection counted |
| Calibrate | SLSQP with the butterfly condition as a nonlinear constraint and the calendar condition built into the parameterisation |
| Verify | Independent pointwise check of `g(k) > 0` and `∂w/∂T ≥ 0` on a dense grid, out to three years |

Measured on the committed snapshots (2026-10-27): **0.645** and **0.654**
volatility points rms across 39 and 42 quotes, both arbitrage-free and both
usable for local volatility.

> **Why arbitrage-free matters here, specifically.** The Dupire local-variance
> denominator is *identically* Gatheral's butterfly function `g(k)` —
> verified symbolically — so an arbitrage-free surface yields a non-negative
> local variance by construction rather than by clipping negatives afterwards.
> That is the precondition for local-volatility pricing.
> See [`docs/models/volatility_surface.md`](docs/models/volatility_surface.md).

Supported underlyings: **华夏上证50ETF (510050)** and
**华泰柏瑞沪深300ETF (510300)**. Offline snapshots are the default source, so
nothing touches the network unless asked.

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
├── src/otcpricer/              # The pricing library
│   ├── core/                   # Types, day counts, validation, config, logging
│   │   ├── types.py            #   OptionType, BarrierKind, Greeks, KIStrikeMode
│   │   ├── market.py           #   EquityMarket (incl. index futures basis)
│   │   ├── daycount.py         #   ACT/365F, ACT/360, ACT/ACT, 30/360, tenors
│   │   └── validation.py       #   One source of truth for input checks
│   ├── models/
│   │   ├── bsm.py              #   Self-built BSM: price, Greeks, implied vol
│   │   ├── svi.py              #   SVI slice + SSVI surface, analytic derivatives
│   │   ├── svi_calibration.py  #   Constrained SSVI fit (no-arbitrage imposed)
│   │   ├── arbitrage.py        #   Independent dense-grid arbitrage checker
│   │   └── surface.py          #   Chain -> fitted surface, in one call
│   ├── marketdata/             # Real option chains
│   │   ├── contracts.py        #   Code parsing + pinned feed-symbol registry
│   │   ├── sources.py          #   akshare (retry/backoff), snapshot, CSV, cache
│   │   ├── snapshots.py        #   Capture and replay, with a format version
│   │   ├── forward.py          #   Forward from put-call parity + residuals
│   │   └── surface_input.py    #   Implied vols and the six quote screens
│   ├── products/snowball.py    # Snowball terms + observation schedule
│   ├── reports/                # Model validation output
│   │   └── surface_report.py   #   Calibration/parity/rejection/arbitrage report
│   └── engines/
│       ├── mc.py               #   Path Monte Carlo (batched, antithetic)
│       ├── pde.py              #   Crank-Nicolson FDM + Rannacher damping
│       └── risk.py             #   Greeks: PDE grid differentiation + MC check
├── tests/                      # 583 tests: analytic, QuantLib, convergence
├── data/snapshots/             # Committed real option chains (test fixtures)
├── docs/models/                # Model derivations, assumptions, rejected methods
├── app.py                      # Flask backend — pricing API endpoints
├── bs_pricing.py               # Compatibility layer over otcpricer.models.bsm
├── greeks_calc.py              # Compatibility layer over otcpricer.models.bsm
├── cn_fdm.py                   # Legacy FDM (still used by the shark-fin path)
├── visualization.py            # Matplotlib charts (CLI use, PNG export)
├── main.py                     # CLI entry point (demo + interactive)
├── templates/index.html        # Single-page frontend (ECharts, no build step)
├── run_app.bat / run_app.ps1   # Launchers (pin the conda env)
├── pyproject.toml              # Package, ruff/mypy/pytest config
└── requirements.txt            # Runtime deps (requirements-dev.txt for tests)
```

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/calculate` | POST | BSM + MC pricing, full Greeks, and the price/Greeks curve arrays |
| `/api/iv` | POST | Implied volatility solver |
| `/api/portfolio` | POST | Portfolio-level Greeks aggregation + MTM |
| `/api/mc_distribution` | POST | MC payoff distribution histogram data |
| `/api/sensitivity` | POST | Price/delta sensitivity matrix |
| `/api/vol_smile` | POST | Implied vol from hand-entered prices, one expiry, no fit |
| `/api/vol_surface/fit` | POST | Fit SSVI to a real option chain; returns the fit quality, rejected quotes, parity residuals and arbitrage verdict |
| `/api/vol_surface/query` | POST | Sample the fitted surface on a (K, T) grid for a heat map |
| `/api/vol_surface/snapshots` | GET | List the offline chains available |
| `/api/smile_presets` | POST | Auto-fill smile table from BSM prices |
| `/api/snowball` | POST | Snowball autocallable — path Monte Carlo |
| `/api/snowball_pde` | POST | Snowball — independent Crank-Nicolson valuation |
| `/api/snowball_crosscheck` | POST | Runs both snowball engines and reports agreement against a measured error budget |
| `/api/snowball_risk` | POST | Snowball Greeks from the PDE solution surface |
| `/api/snowball_risk_crosscheck` | POST | PDE vs Monte Carlo risk, with a per-Greek verdict |
| `/api/shark_fin` | POST | Shark fin barrier option pricing (UOC/DOP/UIC/DIC) |
| `/api/forward` | POST | OTC equity forward pricing |
| `/api/irs` | POST | Interest rate swap NPV + cash flows |
| `/api/cn_fdm` | POST | Legacy CN-FDM pricing + 3-method MC Greeks (Resimulation FD/Pathwise/Likelihood) — ⚠️ see *Known limitations* |

---

## 🛠 Tech Stack

| Library | Version | Role |
|---|---|---|
| *(self-built)* | — | BSM pricing, Greeks, implied vol — see `src/otcpricer/models/bsm.py` |
| `numpy` | ≥ 1.21 | Monte Carlo GBM simulation, vectorized math |
| `pandas` | ≥ 1.3 | Portfolio aggregation, data structuring |
| `scipy` | ≥ 1.7 | Statistical distributions, optimization |
| `flask` | ≥ 2.0 | REST API backend |
| `matplotlib` | ≥ 3.9 | CLI chart output (PNG) |
| **ECharts 5** | CDN | Interactive web charts (no npm required) |
| `QuantLib` | ≥ 1.38 | **Test-time only** — independent benchmark, never a runtime dependency |
| `pytest` | ≥ 8 | 400 tests: analytic identities, convergence, regression |

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/chenseldon/Option-Pricing-Tool.git
cd Option-Pricing-Tool

# Install dependencies
pip install -r requirements.txt
```

> The pricing models are self-contained — there is no `py_vollib` dependency
> any more, so the install that used to fail on Windows no longer applies.
>
> For the test suite as well:
> ```bash
> pip install -r requirements.txt -r requirements-dev.txt
> pytest
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

### Snowball Autocallable

Three mutually exclusive outcomes. The **payment date** matters as much as the
amount, because a knocked-out note pays early:

$$\text{Payoff} = \begin{cases}
c \cdot t_i & \text{paid at } t_i, \ \text{if } S_{t_i} \geq KO \text{ at a monthly observation } t_i > t_{\text{lockup}} \\
-\max\left(0,\ 1 - S_T / S_0\right) & \text{paid at } T, \ \text{if } S_t \leq KI \text{ on any trading day and no knock-out occurred} \\
c \cdot T & \text{paid at } T, \ \text{otherwise}
\end{cases}$$

Note the strike of the knock-in leg is the **initial level** $S_0$, not the
knock-in barrier — the A-share convention, and configurable for variants. A
note that knocks in and later recovers above $KO$ on an observation date still
knocks out and collects its coupon.

Priced two independent ways:

- **Path Monte Carlo** (`engines/mc.py`): exact log-space GBM sampling,
  antithetic variates, batched to a memory budget. Each path is discounted at
  its own payment date.
- **Crank-Nicolson PDE** (`engines/pde.py`): two coupled states (knocked-in and
  not), on a grid that pins both barriers to exact nodes and every observation
  date to an exact time step, with Rannacher start-up damping.

The risk-neutral drift is $r - q - \text{basis}$, so an index futures discount
is priced; discounting remains at $r$.

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

All figures below are **measured from the current code**, not illustrative.
They are asserted as regression tests in `tests/`, so they cannot drift
silently.

### ATM Call Option (CSI 500)

| Input | Value |
|---|---|
| S (spot) | 5,500 |
| K (strike) | 5,500 (ATM) |
| T | 0.25 yr |
| r | 2.0% |
| σ | 20.0% |
| q | 1.0% |

| Output | BSM | Monte Carlo (200k paths) |
|---|---|---|
| Price | **225.4236** | 225.74 |
| Delta | 0.528570 | — |
| Gamma | 0.00072151 | — |
| Vega | 10.912784 (per 1%) | — |
| Theta | −1.263217 (per day) | — |
| Rho | 6.704273 (per 1%) | — |

Cross-validated against QuantLib's `AnalyticEuropeanEngine` to ~1e-13 on price
and ~1e-16 on the Greeks, and against `py_vollib` across 7,200 parameter
combinations to the same precision.

### Snowball (Typical China OTC)

| Input | Value |
|---|---|
| S₀ | 5,000 |
| KO level | 103%, observed monthly |
| KI level | 75%, observed **daily** |
| Lock-up | 3 months (no KO observation) |
| Coupon | 20% p.a. |
| Tenor | 1 year |

| Output | Monte Carlo (500k paths) | CN-FDM PDE |
|---|---|---|
| Price (fraction of notional) | **0.063403** ± 0.000391 | 0.063087 |
| KO probability | 0.6527 | — |
| KI probability | 0.1402 | — |
| Mean KO time | 0.4445 yr | — |

The two engines are independent implementations, and the gap between them
(0.0003, ~0.5%) sits inside the combined Monte Carlo and discretisation error.
`/api/snowball_crosscheck` reports that comparison with its error budget
broken out.

---

## 🔍 Model Validation

Correctness here is demonstrated, not asserted. The test suite is layered by
strength of evidence:

**Model-free identities** hold for any correct implementation, so they do not
depend on a second implementation being right: put-call parity (exact across
2,560 parameter sets), delta parity, no-arbitrage price bounds, Greek signs and
ranges, monotonicity in spot/vol/strike, call-spread slope bounded by `e^(−rT)`,
and butterfly convexity in strike.

**Independent benchmarks**: QuantLib for European options; closed-form
benchmarks for the degenerate snowball cases (unreachable barriers reduce it to
a zero-coupon bond, matched to 1e-8), a semi-annual knock-out, and a terminal
knock-in.

**Convergence studies**: the PDE converges at clean first order under joint
grid refinement (measured ratios 1.98, 2.01, 1.97), and its Richardson limit
agrees with a 12-seed Monte Carlo mean to 0.074%.

**Greeks**: delta and gamma are checked against closed-form anchors rather
than only against each other, compared with the secant of the price curve they
came from (which catches a wrong chain rule even when the prices are right),
and swept across spot through the knock-out barrier to confirm the second
difference does not oscillate near the kink.

**Mutation testing**: the BSM suite was itself validated by injecting eight
deliberate bugs one at a time — dropping theta's `/365`, swapping `N(d1)` for
`N(d2)` in delta, negating the drift in `d1`, omitting vega's rescale, and
others. All eight were caught. A suite that cannot fail is not evidence.

### Snowball defects found and fixed

The project previously described its PDE as an independent cross-validation of
its Monte Carlo, while the two read 0.046047 and 0.037848 — **18% apart**,
because they were valuing different products. Six modelling defects and three
numerical faults were found and corrected:

| Defect | Effect |
|---|---|
| MC struck the knock-in put at the barrier, the PDE at S₀ | The two engines priced different products |
| MC discounted every path from maturity | Understated the note; ~65% of paths pay early, at a mean 0.44 yr |
| No lock-up period | KO probability 0.73 instead of 0.65 |
| No index futures basis (贴水) | A first-order driver of the quotable coupon was inexpressible |
| Knock-in applied every time step, not on its daily schedule | Refining the grid changed the contract: 0.0668 → 0.0612 with no convergence |
| Knocked-in notes could never knock out | Dropped the coupon on paths that knock in then recover — 9σ at KI=85% |
| Barriers sat between grid nodes | Price oscillated under refinement instead of converging |
| Observation dates rounded onto the time grid | A knock-in at maturity was unrepresentable |
| Grid padding fixed at 0.6/0.3 in log space | Barely 1σ wide at σ=0.35; a boundary error does **not** shrink with `dx` |

After the fixes the two engines agree to within 1.94 Monte Carlo confidence
widths across 20 parameter sets, spanning barrier levels, tenors to three
years, 15–50% volatility, rates, the futures basis, lock-ups and observation
frequencies.

### Snowball conventions implemented

Following A-share market practice:

- **Knock-out** observed monthly; pays the coupon accrued *to that date*
- **Knock-in** observed every trading day
- **Lock-up** of 3 months by default, suspending knock-out but not knock-in
- On knock-in without knock-out, a short put struck at **S₀** (configurable to
  the barrier for variant products)
- A knocked-in note **retains** its knock-out feature
- **Futures basis** enters the drift as `r − q − basis`, while discounting
  stays at `r` — two roles a single dividend yield cannot carry. A deeper
  discount lowers the note's value, which is what lets a desk quote a higher
  coupon against it.

### Volatility surface (real market data)

Fitted to the committed chains of 2026-10-27:

| | expiries | quotes | rms (vp) | max (vp) | min `g(k)` | min `∂w/∂T` |
|---|---|---|---|---|---|---|
| 510050 | 4 | 39 | **0.645** | 1.350 | 0.2520 | 1.905e-02 |
| 510300 | 3 | 42 | **0.654** | 1.977 | 0.2628 | 3.356e-02 |

Both under one volatility point, both arbitrage-free on 9,660 grid points out
to three years. Pinned as regression anchors.

What is checked, and against what:

- **`∂w/∂k`, `∂²w/∂k²`, `∂w/∂T`** against Richardson-extrapolated central
  differences across the surface, with the numeric reference good to ~1e-10 so
  a genuine 1e-6 error cannot hide in truncation error. Phase 2b's Dupire
  formula is built entirely from these three.
- **Self-recovery.** Generate a surface from known parameters, sample it,
  refit: `ρ` to 0.02, `η` to 10%, `γ` to 0.05, every `θ` to 0.2%, rms under
  1e-4 vol points. Without this, a tidy fit to real data would prove nothing —
  the objective could be minimised by the wrong surface.
- **The checker can fail.** Surfaces with deliberate butterfly and calendar
  arbitrage are constructed and must be caught. A checker that cannot fail is
  worthless.
- **Screening reconciles.** Every quote in a slice is either kept or booked
  against exactly one rejection reason, and the counts add up to the raw chain.

#### Two defects found while building it

**A flat variance tail silently zeroed local volatility.** `θ` was extrapolated
past the last quoted expiry at PCHIP's own slope there, which PCHIP clamps to
*exactly zero* whenever the variance term structure is concave enough at the
end — i.e. whenever the volatility term structure slopes down, which is
ordinary in a stressed market. Measured: 0.000000 where the realised rate was
0.066667. Total variance then goes flat forever, so local variance is
identically zero beyond the last expiry — and the surfaces stop at 0.41 years
while a snowball runs one to three. Nothing flagged it, because flat total
variance is calendar-arbitrage-*free*. Fixed by extrapolating at the last
quoted secant with the interpolant's final derivative matched to it, and the
checker now also measures `∂w/∂T` relative to `θ/T` so degeneracy is reported
rather than passing.

**`ρ = 1.0` was accepted.** The range check defaulted to a closed upper bound
while every docstring said the open interval. At `|ρ| = 1` the SSVI square root
vanishes at one point in `k`: total variance goes to zero there, both
`k`-derivatives return NaN, and `∂²w/∂k²` is identically zero everywhere, so
the slice has no convexity at all.

Full derivations, the rejected alternatives, and the measurements behind every
tolerance are in
[`docs/models/volatility_surface.md`](docs/models/volatility_surface.md).

### Known limitations

These are open, measured, and stated here rather than left to be discovered:

**Structured-product Greeks are validated as of the current version.** This
entry previously recorded that they were not. The three Monte Carlo methods
disagreed in *sign*, so at most one could have been right — and none was:

| Method | Was | Correct |
|---|---|---|
| Resimulation FD | −8.4e-05 | **+4.9e-05** |
| Pathwise | +1.0e-05 | **+4.9e-05** |
| Likelihood Ratio | +9.0e-06 | **+4.9e-05** |

All three were built on the Monte Carlo payoff carrying the knock-in strike and
discounting defects listed above, and the likelihood ratio carried three
further errors of its own — a sign error in the volatility score, a terminal
score used for a path-dependent payoff, and a missing discount term in rho. The
derivations are in [`docs/models/greeks.md`](docs/models/greeks.md).

Greeks now come from the PDE solution surface, anchored to closed form:
unreachable barriers give delta, gamma and vega of 1.0e-11, −7.0e-14 and
−5.2e-09 against an exact zero, and a terminal-only knock-in converges to its
closed-form delta and gamma at clean first order.

Two methods remain withdrawn on purpose. **Pathwise differentiation** requires
a Lipschitz payoff and a snowball's jumps at both barriers, so it does not
apply; the sound alternative is a mixed pathwise/likelihood estimator, which is
not implemented. The **likelihood ratio** could be rebuilt correctly, and the
corrected score functions are documented for whoever does, but it would add a
high-variance third estimate without adding confidence. Both return `null` with
a stated reason rather than a plausible-looking number.

**Vanilla Greeks are unaffected** — those are closed-form, checked against
QuantLib and against finite differences, and carry the regression anchors above.

**Shark fin** still runs on the legacy `cn_fdm.py`, including its Greeks. Its
closed-form and PDE prices have not been cross-checked the way the snowball's
now are, so giving it Greeks before its prices are validated would be the wrong
order of work.

**The pricing engines still run on flat volatility.** An arbitrage-free SSVI
surface is now fitted from real chains and verified to be usable for local
volatility, but nothing consumes it yet: Dupire itself, and the engine changes
to carry a variable local volatility through the PDE coefficients and the Monte
Carlo step loop, are the next piece of work. Flat volatility stays the default
basis deliberately, so the validated snowball numbers above keep working as
regression anchors rather than being contaminated by calibration error.

**The surface tail is an extrapolation.** `θ` continues past the last quoted
expiry at the last quoted secant rate. The listed options stop at roughly five
months while the products run one to three years, so that convention — not
market information — stands behind any long-dated number.

**A flat rate curve.** A desk discounts off a bootstrapped curve. Only the
forward uses the rate at present, and barely: 100bp moves it by 0.003%, which
is why the surface takes the rate as an input rather than fitting it.

**Unit inconsistency in the API**: `/api/iv` returns implied volatility as a
decimal (0.20) while `/api/vol_smile` returns a percentage (19.9997). Pinned by
a test so that unifying it is a deliberate change rather than an accident.

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

Validation lives in one place (`src/otcpricer/core/validation.py`) and raises
rather than returning a flag, so a failed check cannot be ignored by forgetting
to inspect a return value.

| Param | Valid Range | Notes |
|---|---|---|
| S, K | > 0 | Positive prices only |
| T | > 0 | Years; the API also accepts days and converts |
| σ | (0%, 500%] | Annualized volatility |
| r | any sign | **Negative rates are accepted** — they are real, and a CNY desk sees negative net carry once the futures basis is included |
| q | ≥ 0 | Continuous dividend yield |
| basis | ≥ 0 | Annualized futures discount (贴水); enters the drift only |
| KI level | **strictly below** KO | Crossed barriers leave the payoff undefined |
| Lock-up | **strictly shorter** than T | A lock-up spanning the tenor would silently turn the note into a plain short put |
| Observation freq | ≥ 1 per year | Separate frequencies for knock-out and knock-in |
| H barrier | > 0 | An already-breached barrier is a legitimate state to revalue, so no side constraint is imposed |

---

## 📄 License

MIT License — free for personal learning, portfolio, and interview demonstration use.

---

*Built with Python + Flask + ECharts · 2024–2026*

