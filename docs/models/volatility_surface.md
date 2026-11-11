# Implied volatility surface

How the SSVI surface is built from a real option chain, what guarantees it
carries, which choices were made against the obvious alternative and why, and
what each claim is checked against.

The purpose is narrow and worth stating up front: **Phase 2b needs a surface
from which Dupire yields a real, non-negative local volatility.** Every design
decision below follows from that, and the one result that organises all of them
is in the next section.

---

## The result everything rests on

Write the Dupire formula in terms of total implied variance `w(k, T) = σ²(k,T)·T`
with `k = log(K/F)`:

```
                               ∂w/∂T
σ²_loc = ───────────────────────────────────────────────────────────────
         1 − (k/w)·∂w/∂k + ¼(−¼ − 1/w + k²/w²)(∂w/∂k)² + ½·∂²w/∂k²
```

Gatheral's butterfly-arbitrage function is

```
g(k) = (1 − k·w′/(2w))² − (w′/2)²·(1/w + ¼) + w″/2
```

**These are the same expression.** Verified symbolically with sympy: the
difference simplifies to exactly 0.

The consequence decides the whole architecture:

| Condition | What it guarantees |
|---|---|
| Butterfly: `g(k) > 0` | Dupire's **denominator** is positive |
| Calendar: `∂w/∂T ≥ 0` | Dupire's **numerator** is non-negative |

So an arbitrage-free surface produces a non-negative local variance *by
construction*, not by clipping negatives afterwards. The literature is blunt
about the alternative: feed raw or badly interpolated quotes into Dupire and any
arbitrage that leaks through comes out as a **negative local variance** — and
building the clean surface is most of the work. That is the case for a
parameterised surface rather than numerical differentiation of market prices.

This is also why `arbitrage.local_variance_is_well_defined()` exists as a named
function. The Phase 2b precondition should be something the code states, not
something a reader has to re-derive.

---

## Pipeline

```
chain ──► forward per expiry ──► implied vols + screening ──► SSVI fit ──► check
        (put-call parity)        (Black-76 + identifiability)  (constrained)  (independent)
```

Each stage is one module, and each refuses rather than guessing when its inputs
cannot support an answer:

| Stage | Module | Refuses when |
|---|---|---|
| Fetch | `marketdata/sources.py` | ticker unsupported; feed down after retries |
| Forward | `marketdata/forward.py` | fewer than 4 paired traded strikes; parity residual > 1% of forward |
| Implied vol | `marketdata/surface_input.py` | six screens, below |
| Fit | `models/svi_calibration.py` | too few points to identify the parameters; fit lands on butterfly arbitrage |
| Check | `models/arbitrage.py` | `g(k) ≤ 0`, `∂w/∂T < 0`, or degenerate |
| Orchestrate | `models/surface.py` | fewer than 2 usable expiries |
| Report | `reports/surface_report.py` | — renders the verdict either way |

---

## SVI and SSVI

### Raw SVI, one expiry

```
w(k) = a + b·{ ρ(k − m) + √[(k − m)² + σ²] }
```

Five parameters per slice. Wings are asymptotically linear with slopes
`b(ρ ∓ 1)`, which is the shape's defining property and is checked directly.

Implemented and tested, but **not** what the surface is built from — see the
trade-off below.

### SSVI, the whole surface

Gatheral–Jacquier:

```
w(k, θ) = (θ/2)·{ 1 + ρφ(θ)k + √[ (φ(θ)k + ρ)² + (1 − ρ²) ] }
```

where `θ(T)` is the at-the-money total variance term structure and `φ` is the
power law

```
φ(θ) = η / ( θ^γ · (1 + θ)^(1−γ) ),   γ ∈ (0, ½]
```

Three global parameters (`ρ`, `η`, `γ`) plus one `θ` per expiry. At `k = 0` the
square root collapses to 1 and `w(0) = θ` exactly, independent of `η` and `γ` —
a clean algebraic anchor, and a test.

`ρ` lives on the **open** interval. At `|ρ| = 1` the square root becomes a bare
absolute value that vanishes at `k = −ρ/φ`: total variance goes to zero there,
both `k`-derivatives return NaN, and `∂²w/∂k²` is identically zero everywhere so
the slice has no convexity at all. The validation originally admitted `ρ = 1.0`
and a test caught it.

### Why SSVI rather than per-slice SVI

SSVI fits **less tightly** — far fewer free parameters. That is the deliberate
trade:

| | per-slice SVI | SSVI |
|---|---|---|
| Parameters | 5 per expiry | 3 + 1 per expiry |
| Calendar arbitrage | must be imposed across slices | **structural** (see below) |
| Fit tightness | tighter | looser |

Calendar-arbitrage-freedom coming for free is worth more than the residual,
because the residual is measurable and reportable while a calendar violation
silently produces a negative local variance. When a day's fit genuinely exceeds
tolerance, the report says so rather than the constraints being quietly relaxed.

### Analytic derivatives

Phase 2b's Dupire formula is built entirely from these three, so they are closed
forms rather than finite differences, and each is checked against
Richardson-extrapolated central differences across the whole surface. With
`u = φk + ρ` and `R = √(u² + 1 − ρ²)`:

```
∂w/∂k    = (θφ/2)·(ρ + u/R)

∂²w/∂k²  = θφ²(1 − ρ²) / (2R³)                    — strictly positive for |ρ| < 1

∂w/∂θ    = ½(1 + ρφk + R) + (θ/2)·k·φ′·(ρ + u/R)
∂w/∂T    = ∂w/∂θ · dθ/dT
```

---

## No-arbitrage, imposed rather than checked

The conditions go to the optimiser as constraints. A surface that violates them
is not a slightly worse fit, it is unusable for the purpose — so fitting freely
and rejecting afterwards would mean discarding the work with nothing to fall
back on.

**Butterfly** — Gatheral–Jacquier sufficient conditions, at every fitted
maturity, as an SLSQP `NonlinearConstraint`:

```
θ·φ(θ)·(1 + |ρ|) < 4
θ·φ(θ)²·(1 + |ρ|) ≤ 4
```

**Calendar** — `θ` non-decreasing in maturity. Expressed by optimising over a
first level plus **non-negative increments** rather than the levels themselves.
Monotonicity then holds identically at every point the optimiser visits; it
cannot be violated even transiently. A test fits deliberately inconsistent data
(long-end volatility dragged below the short end) and requires the term
structure to come back monotone anyway.

**Objective** — weighted least squares on implied **volatility**, not variance.
Volatility is what the market quotes and what an error budget is stated in, so
"half a vol point" is directly interpretable, while total-variance residuals mix
maturities. Weights are each quote's vega, so the fit is pulled toward the
quotes carrying volatility information rather than treating a one-tick wing as
equal to an at-the-money quote.

---

## The variance term structure, and a defect it hid

`θ(T)` is interpolated through the fitted points **and the origin**, where total
variance must vanish.

Monotone cubic (PCHIP-derived), not linear, for a specific reason: Dupire needs
`∂w/∂T`, and piecewise-linear `θ` makes it piecewise *constant* — a jump in
local volatility at every expiry boundary. A monotone cubic is C¹ and preserves
the monotonicity that *is* the calendar condition, keeping both properties at
once.

### Beyond the last quoted expiry

This region is not an edge case. The listed ETF options stop at roughly five
months; the snowballs priced off the surface run **one to three years**. Almost
all the pricing happens in the extrapolated tail.

Three candidate tail slopes, and two of them are wrong:

| Choice | Problem |
|---|---|
| Average rate `θ/T` | Kink at `T_last`: analytic `∂w/∂T` disagreed with a central difference by **1.04e-03** exactly there while matching to 1e-9 everywhere else — a jump in local volatility at the edge of the quoted surface |
| PCHIP's own slope at `T_last` | Continuous, but **can be exactly zero** |
| **Last quoted secant** (chosen) | C¹ and strictly positive |

The second one shipped, and the failure mode is worth recording. PCHIP computes
its endpoint derivative from a one-sided three-point formula and **clamps it to
zero** whenever the result's sign disagrees with the adjacent secant. That
happens whenever the variance term structure is concave enough at the end —
i.e. whenever the volatility term structure slopes down, which is ordinary in a
stressed market. Measured on maturities `(0.25, 1.0)` with `θ = (0.05, 0.10)`:

```
PCHIP endpoint slope   0.000000
realised secant        0.066667
```

Total variance then goes **flat forever** past the last expiry. Implied vol
decays as `1/√T` and local variance is identically **zero** across the entire
range where the product lives.

And nothing flagged it, which is the more serious half: flat total variance is
calendar-arbitrage-**free**. The checker reported `min ∂w/∂T = 0.000e+00` and
pronounced the surface sound.

The fix takes the tail rate from the last quoted secant and overrides the
interpolant's final derivative to match, so `θ` stays C¹ across the seam. The
last segment stays monotone — hence still calendar-arbitrage-free — because a
cubic Hermite segment is monotone when both end derivatives lie in `[0, 3s]`,
and here one of them *is* `s` while PCHIP guarantees the other. Verified on four
surfaces: kink at `T_last` between 4.5e-10 and 2.8e-08 (finite-difference
truncation level), monotone over 20001 points to 3 years, `∂w/∂T` strictly
positive throughout.

### Degeneracy is now reported, because it is not arbitrage

`∂w/∂T = 0` satisfies the calendar condition and still gives a local variance of
exactly zero: well defined, non-negative, and useless. So the condition is also
measured **relative** to the surface's own average variance rate `θ(T)/T`, which
is scale-free and separates the cases with room to spare:

| Surface | `min(∂w/∂T) / (θ/T)` |
|---|---|
| 510050 fitted | 0.862 |
| 510300 fitted | 0.849 |
| synthetic, concave term structure | 0.387 |
| synthetic, steeply concave | 0.119 |
| flat tail (the defect) | **0.000** |

The degeneracy floor is 1%, twelve times below the tightest legitimate case.
`arbitrage_free` still reports the truth — a flat surface genuinely admits no
arbitrage — while `local_variance_is_well_defined()` fails it and says plainly
that no arbitrage is present. Reporting only one of those two facts would
mislead in one direction or the other.

---

## Forward from parity, rate as an input

A surface parameterised in `k = log(K/F)` cannot do without the forward. Parity
supplies it from the board alone:

```
C − P = (F − K)·D,     F = mean_over_strikes[ (C − P)/D + K ]
```

which is the least-squares solution for the single free parameter. Averaging is
what makes it robust: one stale quote shifts the estimate by `1/n` of its error
instead of all of it — verified to the exact factor.

Fitting `F` **and** the rate is tempting and wrong. Measured on a live 50ETF
slice with 29 days to run:

| Method | implied r | implied F | residual rms |
|---|---|---|---|
| r fixed at 1.5% | 1.500% | 2.85426 | 0.00257 |
| r fixed at 2.0% | 2.000% | 2.85421 | 0.00250 |
| r fixed at 2.5% | 2.500% | 2.85417 | 0.00243 |
| free two-parameter | **12.480%** | 2.85327 | 0.00162 |

The forward is essentially invariant — 100bp of rate moves it by 0.003%. The
implied *rate* is not identified at all: the free fit returns 12.48%, impossible
for CNY, while improving the residual only from 0.0025 to 0.0016. Over 0.08
years the discount signal (`D ≈ 0.990`) is barely larger than the parity
residuals themselves, so that extra parameter is fitting quote noise.

So: **take the rate as given, imply only the forward, and report the residual as
a data-quality measure.** A proper discount curve is Phase 3's business.

The residual is not decoration. The board carries last traded prices with **no
bid/ask**, so quotes across a slice are non-synchronous; the residual is the
most direct measure of how badly, and a slice above 1% of forward is refused
rather than used.

---

## Quote screening

Six filters, every rejection counted by reason and reported. A pipeline that
silently discards half its inputs is worse than one that fails.

| # | Filter | Why |
|---|---|---|
| 1 | Not traded today | A previous settlement is yesterday's price and says nothing about today's volatility |
| 2 | In the money | Industry convention: out-of-the-money quotes are the volatility-sensitive, reliable ones |
| 3 | Outside no-arbitrage bounds | Cannot be inverted meaningfully |
| 4 | Volatility not identifiable | `vol_uncertainty = tolerance / vega`; where vega underflows, the quote does not pin down a volatility at all |
| 5 | Price at tick resolution | A quote sitting at one tick is a **floor**, not a price |
| 6 | Beyond 3 standard deviations | See below |

Filter 4 is why `bsm.vol_uncertainty()` was built in Phase 1. Filter 5 is the
same measure with `price_tolerance = tick_size`: a 0.0001 price produced a
129.65% implied volatility, 20 standard deviations out.

### The moneyness cap, and what it was distorting

Beyond roughly three standard deviations an option's vega is negligible against
quote noise, and its price stops being a volatility and becomes the market
maker's floor. The captured 300ETF 0.156-year slice shows it plainly: across
`k` from +0.249 to +0.382 — a 14% move in strike — the price goes
0.0022, 0.0020, 0.0019, 0.0018. A real smile would fall far faster. Because the
price is nearly flat while the strike rises, implied volatility is forced
*upward*, reaching 39.5%.

These quotes pass the tick screen (18–22 ticks) and violate no arbitrage bound,
so nothing else catches them. And they were not merely fitting badly in place:

| cap | points | rms (vp) | max (vp) | γ |
|---|---|---|---|---|
| none | 52 | 1.779 | 7.906 | 0.486 |
| 4.0 sd | 50 | 1.194 | 4.003 | 0.500 |
| 3.5 sd | 49 | 1.061 | 2.974 | 0.500 |
| **3.0 sd** | 42 | **0.654** | 1.977 | 0.216 |

`γ` sits pinned against its upper bound until the far wing is removed, then
falls to 0.216 — those quotes were bending the **global** shape parameters, not
just their own corner. Three standard deviations is generous rather than
aggressive: desks commonly fit 5-delta to 95-delta, around 1.65. It is
configurable, and the sensitivity above is recorded so the choice can be
re-examined rather than taken on trust.

---

## The data feed, and one trap

The board endpoint returns **a whole expiry at once, calls and puts together**,
so a four-expiry surface costs four requests rather than one per contract.

The live feed is an undocumented scrape and earned its caution during
development — a `RemoteDisconnected` and an `HTTPError` within minutes of each
other. Hence retry with exponential backoff, a TTL cache, and offline snapshots
as the **default** source. Column names arrive localised and have moved between
upstream versions, so rows are read positionally and the layout is validated
once per pull.

**The trap:** the feed is addressed by a Chinese product name, and the wrong one
fails *silently*. Asking for `嘉实沪深300ETF期权` returns **136 rows whose strike
column is uniformly 10000** — a different payload shape, not an error. Anything
built on it looks plausible and is nonsense. 510300 must use
`华泰柏瑞沪深300ETF期权`. The mapping is pinned in `contracts.py` with a test, so
a well-meaning edit cannot quietly change it.

A second guard: the strike is read **twice**, from the feed's own column and from
the strike embedded in the contract code, and the two must agree. Two
independent encodings of the same field is cheap insurance against a format
change going unnoticed. Verified to agree on every row of both underlyings
(24/24 and 28/28).

---

## What is verified, and against what

| Claim | Check |
|---|---|
| Dupire denominator ≡ `g(k)` | sympy; difference simplifies to exactly 0 |
| `∂w/∂k`, `∂²w/∂k²`, `∂w/∂T` | Richardson-extrapolated central differences over a moneyness × maturity grid; reference good to ~1e-10 |
| `∂w/∂T` across the tail seam | Same, straddling `T_last`; agreement to 1e-5 relative |
| `φ′(θ)` | Central difference across `η ∈ {0.5, 1.2, 4}`, `γ ∈ {0.05, 0.25, 0.5}` |
| `w(0) = θ` | Algebraic identity, to 1e-14 |
| Calibration recovers its own surface | Generate → sample → refit: `ρ` to 0.02, `η` to 10%, `γ` to 0.05, every `θ` to 0.2%, rms < 1e-4 vol points |
| …and survives noise | Half a vol point of noise keeps the refit within one vol point of the clean surface |
| SVI wings | Slopes approach `b(ρ ∓ 1)` |
| Degenerate limits | `b = 0` flat; `ρ = 0` symmetric; single maturity fits |
| **Checker can fail** | Deliberate butterfly arbitrage at three `η`; deliberate calendar arbitrage; the degenerate flat tail |
| Screening is auditable | Every quote in a slice is kept or booked against exactly one reason; the counts reconcile |
| Term structure monotone | 20001 points out to 3–5 years |
| Offline | Default source is `snapshot`; the one network test is deselected |

### Measured on the committed snapshots (2026-10-27)

| | expiries | points | rms (vp) | max (vp) | min `g` | min `∂w/∂T` | relative |
|---|---|---|---|---|---|---|---|
| 510050 | 4 | 39 | 0.645 | 1.350 | 0.2520 | 1.905e-02 | 0.862 |
| 510300 | 3 | 42 | 0.654 | 1.977 | 0.2628 | 3.356e-02 | 0.849 |

Both under one volatility point, both arbitrage-free out to three years, both
usable for local volatility. These are pinned as regression anchors.

---

## Known limitations

- **Non-synchronous quotes** are the dominant data risk and cannot be solved
  from the data itself. Mitigated by using out-of-the-money quotes only,
  discarding untraded contracts, and reporting the parity residual — the point
  being to make it *visible* rather than smooth it away.
- **Short-dated slices** (under two weeks) are thin and noisy. The 510050
  one-day expiry survives screening with 4 points; the 510300 one-day expiry
  does not, and is reported as excluded with its reason.
- **One rate, flat.** Adequate because only the forward uses it, and barely.
  A bootstrapped curve is Phase 3.
- **The tail is an extrapolation.** `θ` continues at the last quoted secant
  rate, which is a defensible convention and not market information. It is the
  single largest modelling assumption behind any long-dated number, which is
  why `/api/vol_surface/query` labels the extrapolated region in its response.
- **Local volatility is not yet derived.** The precondition is established and
  checked; Dupire itself, and the engine changes to consume it, are Phase 2b.
  The default pricing basis remains flat volatility until then.

---

## References

- Gatheral, J. and Jacquier, A. (2014). *Arbitrage-free SVI volatility
  surfaces.* Quantitative Finance 14(1). — SSVI, the sufficient conditions, the
  power-law `φ`.
- Gatheral, J. (2004). *A parsimonious arbitrage-free implied volatility
  parameterization.* — raw SVI, `g(k)`.
- Dupire, B. (1994). *Pricing with a smile.* Risk 7(1).
- Fritsch, F. N. and Carlson, R. E. (1980). *Monotone piecewise cubic
  interpolation.* SIAM J. Numer. Anal. 17(2). — the `[0, 3s]` monotonicity
  condition the tail fix relies on.
