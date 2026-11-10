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
  observation schedule fixed (bumping the tenor would delete a knock-out date).
  The simulation can now do it too, by advancing the **valuation date** instead
  of the maturity — which changes no term of the contract — so theta finally
  has a second engine, agreeing at 0.28 standard errors
- **Carry rho**, one Greek and deliberately not two: `q` and `basis_pa` reach
  the engines only through their sum, so the two bumped prices come back
  *bit-identical*. Reporting both would manufacture an independence the model
  does not have
- **Monte Carlo resimulation with common random numbers** as an independent
  cross-check, with a per-Greek verdict against a measured error budget
- **Spot delta, not re-struck delta**: barriers stay where they were struck
  while the market moves, which is what makes it a hedge ratio
- **Seasoned trades** can be revalued and risked, including after a knock-in

### 💰 Structured Products Quoting
| Product | Pricing Model | Greeks |
|---|---|---|
| **Snowball / Autocallable (雪球)** | Monte Carlo path simulation + Crank-Nicolson PDE, agreeing to ~2 MC confidence widths | PDE grid differentiation, cross-checked against MC resimulation |
| **Shark Fin / Barrier Option (鲨鱼鳍)** | Haug closed-form for a continuous barrier (matching QuantLib to 3.5e-14) + a discretely-monitored PDE, with monitoring frequency a **required** field | None — the legacy 3-method output was withdrawn rather than carried forward |
| **OTC Equity Forward (场外远期)** | Cost-of-carry: F = S·e^((r−q)T) with funding/dividend decomposition | Analytical Δ = ±e^(−qT) |
| **Interest Rate Swap (利率互换)** | Flat-curve DCF; full cash flow schedule + DV01 sensitivity | DV01 per +1bp |

All products output **Bid / Mid / Ask** with configurable spread (bps).

#### The snowball quotes itself

A desk does not ask what a note is worth at a 20% coupon; it asks what coupon
the note can carry. `/api/snowball_quote` answers that, and **exactly** rather
than by root-finding: the price is affine in the coupon, because the knock-out
and knock-in events depend only on the spot path against fixed barriers and
never on the coupon. So `price(c) = A·c + B` and the par coupon is `−B/A`.

The affinity is not assumed. A third solve at the midpoint measures the
collinearity on every call — **7.8e-16** on the reference trade — and the
solver raises rather than return a confident wrong answer if it ever breaks.

What pays for the coupon, measured rather than asserted:

| input | par coupon |
|---|---|
| zero carry, flat vol | 6.53% |
| **CSI 1000 carry, measured +10.15%** | **15.02%** |
| flat vol → fitted 510300 surface | 11.75% |

So the futures discount more than **doubles** the quotable coupon and the smile
adds **80%**. Neither is a refinement.

> **Structured-product Greeks** come from differencing the Crank-Nicolson
> solution surface, cross-checked against Monte Carlo resimulation with common
> random numbers and anchored to two closed-form cases. See
> [*Model Validation*](#-model-validation) and
> [`docs/models/greeks.md`](docs/models/greeks.md).
>
> **The snowball itself** — the five ways its price was wrong, the affine
> coupon solve, what actually pays for the coupon, and why a grid that cannot
> hold the observation schedule is now refused rather than coarsened — is in
> [`docs/models/snowball.md`](docs/models/snowball.md).
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

### 🗄 A 725-Day Institutional Panel

Pricing no longer stands on one afternoon's quotes and a hardcoded 2% rate.
The primary underlyings are now the **CFFEX index options** (IO/HO/MO), and
the reasons are measured:

| | akshare captures | CFFEX index options |
|---|---|---|
| Usable expiries | 3 | **6** |
| Longest tenor | 0.41 y | **1.008 y** |
| Adjusted contracts | 1,812 of 11,372 | **0** |
| Parity residual | 0.08–0.69% | **0.02–0.08%** |
| Matching index future | none | **IF / IH / IM** |

Fit quality is now a **distribution over 2,070 fits**, not a single number —
IO median **0.476** vol points, HO 0.565, MO 0.588, against the captures'
0.587 and 0.654, on roughly three times the points. All 2,070 converged and
all 2,070 are arbitrage-free.

Two defects the data caught, both of which would have passed silently:

- **97,884 rows carry zero volume and a non-zero close.** `has_traded` asks
  whether the last price is positive, so every untraded contract would have
  read as a live quote — and they are not stale marks (only 2.7% match the
  prior day; median daily move 3.25%). CFFEX marks untraded strikes
  *theoretically*; fitting those fits the exchange's own interpolation.
- **The settlement beats the close by 5.7×** on parity residuals, winning 12
  of 12 slices, because a settlement is synchronous by construction.

> Details, including the futures forward check and the two findings only a
> panel could produce, are in [`docs/data/wind_panel.md`](docs/data/wind_panel.md).

### 📉 Futures carry, measured off the board

`basis_pa` was a user input defaulting to **zero**, so the second of a
snowball's three coupon sources was absent from every price — while 725 days of
CFFEX index futures sat in the panel being used only to validate forwards.

What the board shows is a **drift**, not a basis: `log(F/S)/τ = r − q − basis`
is one observable against three unknowns, so nothing there separates the
dividend yield from the financing spread. What is reported is total carry,
`r − drift`, which is the single quantity both engines read.

Two facts govern the design, and both are measured:

- **The board reaches 0.4984 years on average.** A one-year note is 48.5%
  measured and a three-year note 16.2%, so the tail rule is a **required**
  choice with no default — on the CSI 1000 it moves the three-year carry
  between 1.64% and 10.15%, a factor of six.
- **The front contract cannot be annualised.** Its implied drift has a standard
  deviation of 0.473 inside 0.03 years against 0.026 for the stable aggregate,
  and reaches −563% a year. Contracts inside 0.05 years are rejected and
  counted.

Carry is negative on 8–10% of IF and IH days, and IH futures trade *above*
their index on 40.9% of them — which is why `basis_pa ≥ 0` had to go. The CSI
1000 is the exception and the interesting one: positive on **every** panel day,
averaging 10.35%.

> Full measurements, the error I made in the carry convention, and what is
> deliberately not measured are in
> [`docs/data/futures_carry.md`](docs/data/futures_carry.md).

### 🌡 Dupire Local Volatility

Turns the fitted surface into a pricing volatility, so a structured product is
priced against the whole smile rather than one number:

```
sigma_loc^2(S,t) = (dw/dt) / g(y),   y = log(S / F(t))
```

The denominator is *identically* Gatheral's butterfly function, so an
arbitrage-free surface yields a non-negative local variance by construction.
Both the formula and that identity are verified symbolically.

| What | Result |
|---|---|
| **Dupire round trip** (reprice the vanillas the surface was built from) | **0.0064** vol points on 510300, **0.426** on 510050 — against calibration errors of 0.654 and 0.587 |
| Local vol at an 80% knock-in | Roughly **2x** the at-the-money level |
| Effect on a 1-year 103/80 snowball | **−32%** (510050), **−48%** (510300) vs a flat quote at the same at-the-money vol |

> **Flat volatility remains the default.** Local volatility is opt-in, and the
> flat code path is byte-identical — the Phase 1 reference price is still
> exactly `0.0683263509055402`. Derivations, the five defects found building
> this, and every measurement are in
> [`docs/models/local_volatility.md`](docs/models/local_volatility.md).

### 🎲 Heston Stochastic Volatility

Dupire reprices today's vanillas by construction and then implies a forward
smile that flattens. Heston gives up some of today's fit for a forward smile
that is a model's rather than an artefact's — and the trade is stated in the
unflattering direction too: **Heston describes today's quotes worse than SSVI
does**, by 10–12% on three expiries and 38–41% on six, because SSVI buys one
parameter per expiry.

European prices come from a self-built **COS** (Fourier-cosine) engine, which
matches QuantLib's analytic Heston engine to **5e-09** — and to 3e-13 at the
calibrated parameters, closer than QuantLib's own COS engine sits to its
analytic one. Paths come from **Andersen's QE scheme**, which is a requirement
rather than a refinement here:

| params | Feller margin | variance cells exactly zero |
|---|---|---|
| textbook | +0.070 | **0.000%** |
| 510050 (calibrated) | −3.360 | **45.80%** |
| 510300 (calibrated) | −1.525 | **31.60%** |

Nearly half of all simulated variance cells are exactly zero at the calibrated
parameters, so a truncated Euler scheme would be clamping a negative value on
half its cells — a bias that does not shrink with the step.

> **This is the one place in the project where a price has only one engine.**
> A Heston snowball's partner would be a 2D PDE with a cross-derivative and an
> ADI solver, which is deliberately not built. The response says so in its own
> JSON (`cross_checked_by_a_second_engine: false`) rather than only in prose.
> Three numerical traps found on the way — the Little Heston Trap overflowing
> at 1.85 years, a transcribed cumulant wrong by 5×, and a payoff coefficient
> returning 3.9e13 as a price — are in
> [`docs/models/heston.md`](docs/models/heston.md).

### 🦈 Shark Fin: Two Engines At Last

The last product priced by two unconnected pieces of code that disagreed about
what the contract **was** — an analytic formula inlined in a Flask route
(continuous monitoring) and a legacy grid (discrete). Nothing compared them. On
the legacy test case that is **43.02 daily against 35.17 continuous**, a 22%
difference, so `monitoring` is now a required field with no default.

Cross-validation immediately found that **the shipped analytic formula was
wrong for down barriers**:

| kind | before | QuantLib | error |
|---|---|---|---|
| up-and-out call | 1.12969334 | 1.12969334 | 2e-14 |
| down-and-out put | 4.87975297 | 1.82112396 | **3.06** |
| down-and-out put (K=110) | 6.58113901 | 1.09472507 | **5.49** |

Haug's reflection terms carry `η = −1` for an up barrier and `+1` for a down
one; the old code applied the up sign unconditionally. It had only ever been
checked on an up-and-out call — while carrying a comment claiming validation to
0.01%. All sixteen cases now agree to **≤3.5e-14**.

Barriers also price on the fitted surface, where the smile matters more than
anywhere else in the project, because a barrier payoff is concentrated at one
spot level rather than averaged:

| contract | flat @ ATM | local vol | difference |
|---|---|---|---|
| 1y up-and-out call, 110% | 0.00689 | 0.01335 | **+93.7%** |
| 1y down-and-out put, 80% | 0.08602 | 0.05457 | **−36.6%** |

> The discrete-to-continuous premium halves as the observation frequency
> quadruples — the `1/√m` rate Broadie, Glasserman and Kou give — and two more
> defects (solving across the barrier costing an order of convergence, and
> Rannacher damping needing to follow *every* observation) are in
> [`docs/models/barrier.md`](docs/models/barrier.md).

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
│   │   └── vol.py              #   LocalVolatility protocol (keeps engines off marketdata)
│   ├── models/
│   │   ├── bsm.py              #   Self-built BSM: price, Greeks, implied vol
│   │   ├── svi.py              #   SVI slice + SSVI surface, analytic derivatives
│   │   ├── svi_calibration.py  #   Constrained SSVI fit (no-arbitrage imposed)
│   │   ├── arbitrage.py        #   Independent dense-grid arbitrage checker
│   │   ├── surface.py          #   Chain -> fitted surface, in one call
│   │   ├── local_vol.py        #   Dupire local volatility + diagnostics
│   │   ├── heston.py           #   Heston CF (little-trap form only) + cumulants
│   │   └── heston_calibration.py #  Heston fit, on SSVI's own objective
│   ├── marketdata/             # Real option chains
│   │   ├── contracts.py        #   Code parsing + pinned feed-symbol registry
│   │   ├── sources.py          #   akshare (retry/backoff), snapshot, CSV, cache
│   │   ├── snapshots.py        #   Capture and replay, with a format version
│   │   ├── forward.py          #   Forward from put-call parity + residuals
│   │   ├── surface_input.py    #   Implied vols and the six quote screens
│   │   ├── wind.py             #   The 725-day institutional panel (settle prices)
│   │   └── rates.py            #   Two-tenor government curve (NOT a bootstrap)
│   ├── products/
│   │   ├── snowball.py         #   Snowball terms + observation schedule
│   │   ├── shark_fin.py        #   Barrier note; monitoring is required, not default
│   │   └── irs.py              #   Swap on a curve, float leg resetting off it
│   ├── validation/             # Checks whose output is evidence, not a price
│   │   └── futures_check.py    #   Parity forward vs the matching index future
│   ├── reports/                # Model validation output
│   │   ├── layout.py           #   Shared plain-text report formatting
│   │   ├── surface_report.py   #   Calibration/parity/rejection/arbitrage report
│   │   ├── local_vol_report.py #   Extrapolation, forward curve, Dupire round trip
│   │   └── heston_report.py    #   Leads with the caveats, not the fit quality
│   └── engines/
│       ├── mc.py               #   Path MC (batched, antithetic, local vol, Heston QE)
│       ├── pde.py              #   Crank-Nicolson FDM; constant + variable coefficients
│       ├── european.py         #   European local-vol PDE/MC -- the validation vehicle
│       ├── fourier.py          #   COS method -- the Fourier engine long promised
│       ├── barrier.py          #   Barrier PDE + MC + verified closed form
│       └── risk.py             #   Greeks: PDE grid differentiation + MC check
├── scripts/build_wind_anchor.py # The 11-minute panel sweep, run by hand
├── tests/                      # 1,252 tests: analytic, QuantLib, convergence
├── data/snapshots/             # Committed real option chains (test fixtures)
├── data/wind/                  # The 725-day anchor + its threshold manifest
├── docs/models/                # Model derivations, assumptions, rejected methods
├── app.py                      # Flask backend — pricing API endpoints
├── bs_pricing.py               # Compatibility layer over otcpricer.models.bsm
├── greeks_calc.py              # Compatibility layer over otcpricer.models.bsm
├── visualization.py            # Matplotlib charts (CLI use, PNG export)
├── main.py                     # CLI entry point (demo + interactive)
├── templates/index.html        # Single-page frontend (ECharts, no build step)
├── serve.py                    # Production entry point (waitress, debugger off)
├── scripts/check_deps.py       # Can this interpreter run the app? Used by both launchers
├── run_app.bat / run_app.ps1   # Launchers (pick the first interpreter that passes the probe)
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
| `/api/local_vol/surface` | POST | Dupire local volatility on a (spot, time) grid, with implied vol alongside |
| `/api/local_vol/validate` | POST | The Dupire round trip — the correctness evidence, with a pass/fail verdict |
| `/api/european` | POST | European option priced on the local volatility (skew-consistent) |
| `/api/irs` | POST | Interest rate swap on a discount curve, with a DV01 that survives bump-and-reprice |
| `/api/heston/calibrate` | POST | Fit Heston to a chain, serving the SSVI comparison as data |
| `/api/heston/european` | POST | European option by COS under Heston |
| `/api/heston/snowball` | POST | Snowball by QE simulation — one engine, and the response says so |
| `/api/shark_fin` | POST | Barrier note on the engines, with explicit monitoring and its cross-checks |
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
> `requirements.txt` kept pinning it for four phases after the code stopped
> importing it, and both launchers aborted when it was absent; that is fixed.
>
> **Python 3.12 and 3.13 are both supported and both tested** — CI runs the
> suite on each. This file previously claimed 3.13 did not work, which was true
> only while `py_vollib` was a dependency.
>
> `pyarrow` is needed to read the Wind panel, which is parquet. It was an
> undeclared dependency until now: it worked on the development machine only
> because conda supplied it, and CI never noticed because the panel is
> institutional data that is not committed, so those tests skip. Anyone who
> copied the data directory and installed from `requirements.txt` got an
> `ImportError` from inside pandas instead.
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
python serve.py          # or:  .\run_app.ps1     /     run_app.bat
```

Open **http://127.0.0.1:5000** in your browser.

`serve.py` runs the app behind **waitress** with the debugger off. The entry
point used to be `app.py`, which starts Flask's development server with
`debug=True` — fine for development and wrong to hand anyone, because the
debugger renders an interactive Python console on any unhandled traceback and
this application authenticates nothing. `python app.py` still works and still
starts that server, deliberately: two commands with visibly different names
rather than one whose behaviour depends on a variable.

Configuration is by environment variable, same `OTCPRICER_*` scheme as the rest
of the project:

| variable | default | |
|---|---|---|
| `OTCPRICER_HOST` | `127.0.0.1` | Loopback by default. Binding publicly is a decision, not a default — see below. |
| `OTCPRICER_PORT` | `5000` | |
| `OTCPRICER_THREADS` | `8` | Serves concurrent requests. Does **not** make any one request faster. |
| `OTCPRICER_DEBUG` | unset | `1` switches to the development server, with a warning on stderr. |

> **What the production server does not fix.** Said plainly, because a
> production-looking server invites the assumption that the rest is production
> too: there is **no authentication, no authorisation and no rate limiting**.
> Anyone who can reach the port can price anything and can make the process
> spend 30 seconds of CPU per request. There are no request timeouts and no
> HTTPS. The default bind is loopback for exactly those reasons; put a
> reverse proxy in front of it before it goes anywhere else.
>
> **Threads do not speed up a request.** Within one request the solves are
> GIL-bound — the PDE time loop is Python-level — and threading them was
> measured at **0.55×**, i.e. 1.8 times *slower*, on a 32-core machine.

If a launcher cannot find a usable interpreter it says which ones it tried and
what each was missing. To point it at a specific one:

```bash
set OTCPRICER_PYTHON=C:\path\to\python.exe   # or $env:OTCPRICER_PYTHON in PowerShell
```

To check an interpreter without starting anything:

```bash
python scripts/check_deps.py
```

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

### Local volatility (Phase 2b)

The decisive test is the **Dupire round trip**: price the vanillas the surface
was built from and check its own smile comes back. Dupire guarantees that in
continuous time, so any residual is the engine's discretisation — which makes
the right budget the surface's own calibration error.

| | worst round trip | calibration rms | share |
|---|---|---|---|
| 510050 | 0.426 vp | 0.587 vp | 73% |
| 510300 | **0.0064 vp** | 0.654 vp | **1.0%** |

Three tests exist purely to *localise* a failure, each seeing something the
others cannot:

- **Constant local vol** lands on the flat engine's grid node for node (the
  padding rules reduce exactly), so a discrepancy is indexing, not
  discretisation. Agrees to **8.3e-17** on price and 2.7e-15 across the
  surface, and all five Greeks to 1e-17–1e-15.
- **Smooth time-varying vol** is the only test that can see the theta scheme
  taking coefficients from the wrong time level — invisible under constant
  coefficients, and first-order in `dt`, so it reads as slow convergence
  rather than a wrong answer. Measured order **1.87**.
- **Two maturities** pin the time direction, because a single European price is
  invariant to a reversed time axis.

#### Five defects found while building it

| Defect | How it was caught |
|---|---|
| `ForwardCurve.drift()` off by one segment | Integrated drift vs the curve's own log-forward, off by a constant 1.62e-03. Inspection did not catch it |
| **Rho sign error**: +0.0036 → −0.0001, because the drift lives on the surface's forward curve and bumping `market.r` moved only the discounting | Degenerate check on *all five* Greeks, not just the price |
| **A Phase 2a screening defect**: 510050's 1-day expiry cleared the 4-point screen with exactly 4 quotes, fitted *worse* (0.645 → 0.587 vp), pinned γ at its bound, and kinked θ(T) | Round-trip error 5.82 vp with it, 3.49 without. A one-week minimum maturity now applies |
| Strike-on-node shift subtracted the remainder, leaving `m·dx + 2·rem` | Its own test |
| My first round trip inverted implied vol at **spot** instead of the forward, showing 2–5.5 vp of fake error | Running the constant-vol test *first*: it came back clean at 1e-5 and sent me to the test rather than the engine |

That last one is the methodological point: **localising tests first, decisive
test second.**

#### What had to change, and what did not

Delta, gamma and theta needed nothing — the first two are differenced off the
solution surface, theta off two adjacent time levels. **Vega** now shifts the
*surface* (`θ_i → (√(θ_i/T_i) + δ)²T_i`), because bumping `market.sigma` would
not touch the diffusion but *would* change the flat grid's padding, returning a
small number produced by discretisation. A plausible, meaningless vega is
harder to catch than a zero, and there is a test that bumping sigma from 0.05
to 0.60 leaves the price exactly equal.

The local-vol vega is a **different derivative** from the flat one, not the same
one measured better: SSVI's curvature scale depends on the variance level, so a
parallel at-the-money shift also tilts the smile. Hence the method tag
`pde-grid+ssvi-parallel-vega`.

#### The Monte Carlo is no longer exact

A constant-coefficient log step integrates GBM exactly over any step size; a
state-dependent one does not. So `steps_per_year` becomes an accuracy
parameter. **But the bias is not measurable**: averaging six 400,000-path runs
per configuration, it sits within 1.7 standard errors of zero at 63, 252 and
1008 steps/year with no trend — bounded below 1.3e-04, or 0.11% of price.

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

**Shark fin has no Greeks.** Its *prices* are now validated — the migration to
`engines/barrier.py` found that the shipped closed form was wrong for every
**down** barrier, by 1.7 to 5.5 on prices of order 1 to 8, despite carrying a
comment claiming validation to 0.01%; it had only ever been checked on an
up-and-out call. All sixteen cases now match QuantLib to 3.5e-14. But the
legacy three-method Greeks were *withdrawn* rather than carried forward, because
they came from the same code whose snowball deltas were wrong in sign and were
never validated for a barrier at all. `cn_fdm.py` has since been deleted.

**Heston has no Greeks, and its snowball has one engine.** The only number in
this project without a second engine. Its partner would be a two-dimensional
PDE with a cross-derivative term and an ADI solver, which is not built — and the
response says so in a field rather than leaving it to be assumed.

**66% of a three-year horizon is extrapolated.** Improved from 86.5% by the
move to index options reaching 1.008 years, but not solved — it remains the
largest single assumption behind any long-dated number.

**SSVI's `gamma` sits at its upper bound on 46% of fits.** The power law makes
the short-maturity wing diverge like `t^(-gamma/2)`, so on nearly half the
725-day panel that divergence rate is set by the bound rather than by the
quotes. Invisible on a single date, and it bears directly on local volatility.

**A Heston snowball has one engine.** Every other number here is confirmed by
a second independent engine; this one's partner would be a two-dimensional PDE
with a cross-derivative and an ADI solver, which was deliberately not built.
What is verified is that QE reproduces the COS European prices — the step to a
path-dependent payoff is not.

**A government curve is not a funding curve.** An OTC desk discounts above the
CGB yield, and the futures check *cannot* measure the spread: the rate moves
the parity forward by under 0.05% per 100bp against gaps of 0.08–0.46%, so no
plausible rate explains them. No spread is invented in its place.

The akshare captures still carry the old 86.5%, since they stop at 0.41 years;
that is the figure the Phase 2b numbers below were measured against. Either way
the share is reported as `extrapolated_fraction` and led with in the
local-volatility report, because no amount of grid refinement touches it.

**Flat volatility remains the default pricing basis.** Local volatility is
opt-in, so the validated Phase 1/1.5 numbers keep working as regression anchors
rather than being contaminated by calibration error. The flat code path is
byte-identical, not merely close.

**The flat grid budget does not carry over.** At the flat defaults the two
engines disagreed by 9.4 standard errors on a 510050 one-year snowball;
refining closed it monotonically to 2.0. `LOCAL_VOL_PDE_CONFIG` is sized at the
point where the comparison stops being limited by the grid. None of this is
about local volatility as such — the smoother 510300 surface already agrees at
the flat defaults.

**Local volatility is calibrated, not dynamic.** Dupire reprices today's
vanillas by construction but implies a forward smile that flattens, which is
known to misprice forward-starting and cliquet features. A snowball's knock-out
is not forward-starting, so this is second-order here, but it is a real
limitation of the model class. Stochastic volatility is Phase 3's business.

**The forward curve has a kink** at the last quoted expiry, +0.054 to +0.088,
the acknowledged cost of reverting to the caller's own carry past the quotes
rather than extrapolating a −5% implied carry out to three years.

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

