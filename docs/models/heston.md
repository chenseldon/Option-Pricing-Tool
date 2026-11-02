# Heston: stochastic volatility, and what it costs

Dupire reprices today's vanillas **by construction** and then implies a forward
smile that flattens — which is known to be wrong, and matters for anything
path-dependent. Heston gives up some of today's fit in exchange for a forward
smile that is at least a model's rather than an artefact's.

That trade is the whole argument, and it needs stating in the unflattering
direction too: **Heston describes today's quotes worse than SSVI does.**

---

## Why this is not behind `core.vol.LocalVolatility`

That Protocol asks for `local_vol(S, t)` — a volatility that is a *function* of
spot and time. Heston's variance is a **second state variable** with its own
Brownian motion, so no such function exists and no interface widening would
produce one. The Protocol's docstring said this deliberately from the start.

The consequence is structural: a local-volatility PDE is one-dimensional in
space, a Heston PDE is two-dimensional with a cross-derivative term and needs
an ADI solver. **That is not built here**, which is why the Heston snowball
rests on one engine — see the last section.

---

## Fit quality: it loses, and predictably

Same quotes, same vega-weighted objective on implied volatility, same per-slice
normalisation as `svi_calibration`:

| chain | expiries | pts | SSVI params | SSVI rms | Heston rms | ratio |
|---|---|---|---|---|---|---|
| 510050 (akshare) | 3 | 35 | 6 | 0.587 | 0.644 | **1.10×** |
| 510300 (akshare) | 3 | 42 | 6 | 0.654 | 0.730 | **1.12×** |
| HO (CFFEX) | 6 | 93 | 9 | 0.460 | 0.648 | **1.41×** |
| IO (CFFEX) | 6 | 121 | 9 | 0.670 | 0.924 | **1.38×** |
| MO (CFFEX) | 6 | 130 | 9 | 1.384 | 1.447 | **1.05×** |

Volatility points. The pattern is the **parameter budget**, not a defect: SSVI
carries three global parameters plus one per expiry, so a six-expiry chain gives
it nine against Heston's five in total, and its advantage grows exactly as the
term structure gets richer.

The objective is matched to SSVI's on purpose. Two models scored on two
different objectives are not comparable, and the comparison is the point.

---

## Three numerical traps, all found by measurement

### 1. The Little Heston Trap

The characteristic function has two algebraically identical forms differing only
in which square root is taken — `exp(+dT)` or `exp(-dT)`.

| T | classic, largest adjacent jump | stable form |
|---|---|---|
| 0.5–1.0 | agrees to **1e-14** | agrees to 1e-14 |
| 2 | **7.2e-02** | 7.9e-03 |
| 20 | **not finite** | 2.9e-02 |

At the parameters calibrated to *this* data (σ = 1.96) the classic form stops
being finite at **1.85 years** — inside a snowball's life. So the stable form is
a hard requirement here, not a textbook footnote.

The classic form is kept in the module **solely so the test can require it to
fail**. A numerical hazard nobody can reproduce is one someone will eventually
simplify away. The threshold also depends on how far out in `u` you evaluate
(COS reaches ≈240 at the default settings), which is why a coarser method might
not notice.

### 2. A transcribed cumulant, wrong by a factor of five

Fang–Oosterlee give `c2` in closed form; it is five terms. Mine returned
**0.0888** where the true value is **0.4424** at
`(v0, κ, θ, σ, ρ) = (0.25, 0.5, 0.30, 1.5, −0.9)`.

Caught by two independent routes that agreed with each other to 0.5% and with my
formula not at all: differentiating the characteristic function, and a
400,000-path Euler simulation.

Since `c2` sizes the COS truncation window, the consequence was a window half as
wide as needed, a clipped density, and prices that were **finite, plausible and
wrong**. `cumulants()` now differentiates `log φ` numerically — exact up to the
step, automatically consistent with whatever the engine integrates, and immune to
transcription by construction. An `abs()` I had placed around `c2` was hiding
it; that is now a floor-at-zero with a comment saying why.

### 3. The call payoff coefficient overflows

`chi` contains `exp(upper)`. On a three-year high-volatility density at L = 20
the upper bound reaches 30.8, and `exp(30.8) ≈ 2.4e13` — **the engine returned
3.9e13 as a price.**

Puts are integrated instead (their coefficients stay O(1)) and calls come from
put-call parity. That single change made accuracy **monotone** in both `L` and
`N`; the apparent degradation I had measured at wide ranges was this overflow all
along, and my first docstring confidently attributed it to an interior optimum
that does not exist.

### Defaults, measured rather than assumed

| | L=12, N=128 | L=20, N=512 | L=30, N=1024 | L=40, N=2048 |
|---|---|---|---|---|
| worst error vs QuantLib | 3.6e-04 | 9.6e-07 | **6.2e-10** | 7.7e-13 |

**L = 30, N = 1024**, costing 265 µs a price. Not the paper's suggested L = 10:
the narrow settings do not converge slowly, they converge to the *wrong answer*
— at L = 12 the error floors at 3.8e-05 unchanged from 256 terms to 2048,
because the clipped tail is simply absent from the integral. There is a test
asserting exactly that, since the distinction between a tolerance and a defect
is the whole point.

---

## Feller is violated, and that is the point

Calibrated to real quotes, `2κθ − σ²` comes back negative every time (−3.36,
−1.52, −2.04, −1.19, −0.22). The condition is **reported, not enforced**: a
constructor that rejected it would reject the model's own best fit to its own
data, which is a sign the constraint does not belong there.

What it means is measurable, and it is not a corner case:

| params | Feller | variance cells exactly zero | paths touching zero |
|---|---|---|---|
| textbook | +0.070 | **0.000%** | 0.00% |
| 510050 | −3.360 | **45.80%** | **100.00%** |
| 510300 | −1.525 | **31.60%** | 99.03% |

**Nearly half of all simulated variance cells are exactly zero** at the
calibrated parameters. A truncated Euler scheme would be clamping a negative
value on half its cells, and that bias does not shrink with the step because the
event being clamped is not a discretisation artefact. The Feller-satisfied row
is the control: the atom is the model's behaviour, not the scheme's.

So **Andersen's QE scheme is a requirement, not a refinement.** It matches the
exact conditional moments and samples from a distribution supported on the
non-negatives, with a genuine atom at zero.

---

## Verification, and one methodological mistake

- **COS against QuantLib's analytic Heston engine**: 5e-09 worst case across
  three parameter sets, four maturities and three strikes — and 3e-13 at the
  calibrated parameters, closer than QuantLib's *own* COS engine sits to its
  analytic one.
- **COS against a Black-Scholes transform**: 9.8e-14, which checks the
  quadrature and payoff coefficients with no stochastic volatility in the way.
- **The martingale identity** `φ(−i) = exp(drift·T)`: held to 1e-11 across four
  parameter sets, four maturities and three drifts. One evaluation touches every
  term, so almost any algebra slip breaks it.
- **QE against COS**: within three standard errors on nine comparisons.

### The mistake worth recording

A single QE run priced three strikes and all three came out **1.7–1.9 standard
errors high**, with a 6.5bp martingale error to match. That looks exactly like
discretisation bias.

It was not. The three strikes **share one set of paths**, so their errors are
correlated and a common sign is what *one seed's* noise looks like. Six seeds
per configuration put the bias at **−0.0018 ± 0.0150** at 252 steps a year —
inside one standard error of zero — and showed the apparent convergence trend in
my single-seed table was noise as well.

The bias tests now average across seeds, and one of them pins the seed-to-seed
spread so that reading stays visibly uninformative.

### A calibration flag that was doing two jobs

On synthetic data recovered to four decimals, one start of four reached rms
4.6e-06 while the next best stopped at 4.2e-03 — a thousand times worse, κ at
8.09 against a true 2.00 — and **both reported successful convergence**.

So the fit was excellent and corroborated by nothing, which is a different fact
from the parameters being ambiguous. The result reports `converged`,
`confirmed_by_multiple_starts` and `ambiguous` **separately**, because collapsing
them hides the failure that actually occurs. On 510300 all three are informative:
converged, three of four starts agree on quality, and they disagree on
parameters by 0.80 — the κ/σ trade-off exactly as predicted.

---

## One engine, and saying so

Everything above concerns **European** prices, which are cross-checked twice
over. A Heston **snowball** is not.

The second engine would be the two-dimensional PDE with a cross-derivative and an
ADI solver that this phase deliberately did not build. So a Heston snowball rests
on one engine where every other number in this project rests on two.

What *is* established is the layer below: QE reproduces the COS European prices,
so the simulation is consistent with the model. The step from there to a
path-dependent payoff is not independently confirmed.

This is carried in the data, not just the prose —
`SnowballMCResult.cross_checked_by_a_second_engine` is `False`, and `as_dict()`
adds a `single_engine_note` — because a caller reading JSON has no other way to
know. **A Heston snowball should not be read as carrying the same weight as the
flat or local-volatility one.**

---

## Cost

| | |
|---|---|
| One European price (COS) | 265 µs |
| One calibration | 56–75 s |
| A 725-day panel sweep | ≈ 39 hours |

Which is why Heston parameters are **not** swept across the panel the way the
SSVI anchor is.
