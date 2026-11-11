# Dupire local volatility

How the fitted surface becomes a pricing volatility, what each claim is checked
against, and the five defects found getting there.

The companion to [`volatility_surface.md`](volatility_surface.md), which builds
the implied surface. That file ends with the precondition; this one consumes it.

---

## The formula, and the one convention that matters

In forward-moneyness coordinates `y = log(S/F(t))` with total implied variance
`w(y,t)`:

```
                               ∂w/∂t
σ²_loc = ───────────────────────────────────────────────────────────────
         1 − (y/w)·∂w/∂y + ¼(−¼ − 1/w + y²/w²)(∂w/∂y)² + ½·∂²w/∂y²
```

**Verified symbolically before any of this was written.** Starting from the
normalised Black call `c̃ = N(d₁) − e^y N(d₂)` with `d₁,₂ = (−y ± w/2)/√w` and
solving Dupire's equation in these coordinates,

```
∂c̃/∂t = (v/2)(∂²c̃/∂y² − ∂c̃/∂y)
```

the resulting `v` minus the expression above simplifies to **exactly 0**.

That settles the convention this whole phase hinges on: **`∂w/∂t` is the partial
at fixed `y`**, not at fixed strike, because the derivation lives in `(y,t)`
coordinates throughout. It is therefore exactly what `SSVISurface.dw_dT` already
returns, and **no forward-curve correction term enters the numerator** —
provided the engine's drift is the curve the surface was fitted against.

### The denominator is not transcribed

It is identically Gatheral's butterfly function `g(y)`, which `butterfly_g` has
computed since Phase 2a, so `local_vol.py` calls that instead of writing the
expanded form a second time.

Not merely tidier. The grouped form keeps the `1/w²` terms *inside a square*
rather than as a free-floating sum of large cancelling quantities, so it is
better conditioned as `w → 0`. The two agree to **4.4e-16** across `|y| ≤ 1.2`
for `t` from 3 years down to 1e-08, both stay well conditioned at `t = 1e-12`,
and the equality is pinned by a test so they cannot silently fork.

The consequence is structural: because the denominator *is* the butterfly
expression, a surface that passed `local_variance_is_well_defined()` has a
positive denominator and a non-negative numerator **by construction** — not by
clipping negatives afterwards. Measured on the committed chains over
`|y| ≤ 1.2`, `t ≤ 3`: **zero** negative local variances in 96,400 points, with
the denominator bottoming at 0.252 and 0.247.

---

## Forward consistency is decisive, not cosmetic

The surface is parameterised in `log(K/F)`, so anything pricing off it must use
the *same* forward curve. Otherwise the local volatility is evaluated at the
wrong moneyness and nothing downstream says so.

Measured on the committed chains:

| | implied carry | `r−q` at 3%/1% | 1-year forward error |
|---|---|---|---|
| 510050 | −4.16% p.a. | +2.00% | **+6.4%** |
| 510300 | −5.92% p.a. | +2.00% | **+8.2%** |

Against a snowball with barriers at 103% and 80%, a 6–8% forward error swamps
every other modelling choice in the engine.

### `ForwardCurve` stores the shape, not the level

What it holds is the instantaneous drift `d log F/dt`, piecewise constant
between quoted expiries. Levels come from whatever spot the caller prices at.
That keeps it scale-free: a surface fitted on an ETF at 2.85 can price a note
quoted on an index at 5000, because moneyness only cares about ratios and a
structured note's barriers are percentages.

### Beyond the quoted range the caller's carry governs

Past the last quoted expiry the drift reverts to the caller's `r − q − basis`.
The alternative — extrapolating the implied carry — would carry −5% measured on
five-month options out to three years, dropping the forward 14% on that
evidence alone.

The price of that choice is a **kink** in the drift at the handover, measured at
+0.0536 (510050) and +0.0882 (510300). It is reported in the response and in
the report rather than smoothed, because on this data it is not small.

---

## The decisive test: the Dupire round trip

A local volatility is *derived* from an implied volatility surface, so there is
exactly one way to know it is right: **price the vanillas the surface was built
from and check its own smile comes back.** Dupire guarantees the round trip in
continuous time, so any residual is an error — in the formula, the forward
curve, the moneyness mapping, the operator coefficients, the index alignment,
or the discretisation.

That makes the natural budget **the surface's own calibration error**: an engine
adding less noise than the data already carries is as accurate as the inputs can
justify.

Measured at a production grid (1600 nodes, 1008 steps/year, 800-step floor):

| | worst round trip | calibration rms | share |
|---|---|---|---|
| 510050 | 0.426 vp | 0.587 vp | 73% |
| 510300 | 0.0064 vp | 0.654 vp | **1.0%** |

The 510050 residual converges away cleanly: 0.874, 0.401, 0.175, 0.0735, 0.0298
volatility points for 400 to 6400 time steps.

### Three tests exist only to localise a failure

Each sees something the others cannot, which is the point:

| Test | What only it can see |
|---|---|
| **Constant local vol** | Lands on the flat engine's grid *node for node*, so a discrepancy is indexing or coefficients, not discretisation |
| **Smooth time-varying vol** | Whether the theta scheme takes coefficients from the right time levels — invisible under constant coefficients, since the two levels coincide |
| **Two maturities** | A reversed time axis. A single European price cannot detect one: `∫₀ᵀf(T−t)dt = ∫₀ᵀf(u)du` |

The second measured order **1.87** against the theoretical 2, so the silent
`O(dt)` degradation is ruled out.

---

## The engines

### PDE: variable coefficients, flat path untouched

`price_snowball_pde(..., local_vol=...)` dispatches to a separate
variable-coefficient grid. Omit it and the original scalar path runs unchanged,
so flat results are identical **because the same bytes execute** — the Phase 1
reference price is still exactly `0.0683263509055402`.

In log space the operator is

```
L V = ½σ²(S,t)·V_xx + (μ(t) − ½σ²(S,t))·V_x − r·V
```

Central differences on a uniform grid stay second-order with coefficients at
the nodes. Two details are invisible at flat volatility, so no existing test
could catch either, and both are first-order in `dt` — they read as slow
convergence, not as a wrong answer:

- **The explicit stencil takes the old level; the Dirichlet folds take the
  new one.** The folded terms come from the *implicit* matrix. The
  constant-coefficient version uses one scalar for both and is right only
  because the levels coincide.
- **The bands are sliced asymmetrically**: `lower = −θ·dt·a_new[1:]`,
  `upper = −θ·dt·c_new[:-1]`, `main` full length. Verified against entrywise
  construction to **0.0**; the naive `a[:-1]/c[1:]` differs by 1.0e-02, so the
  test has teeth.

The grid geometry was extracted (`_build_geometry`) so both paths share one copy
of the observation and barrier alignment — the most bug-prone code in the
module, where two Phase 1 defects lived. Verified as pure code motion: the
reference price is bit-identical before and after.

Coefficients roll: the new level of backward step `k` is the old level of step
`k+1`, so eleven evaluations cover ten iterations. The cache is keyed on the
step index, not a call counter, because the knocked-in state is stepped before
the un-knocked one and both want the same operator.

`bc_ki_low` is a discounted forward intrinsic, so it carries the drift and had
to stop using a constant carry. It is spelled as the single `exp(−carry·τ)` the
constant-carry algebra collapses to, which is bitwise identical only that way.

### Monte Carlo: exactness given up, honestly

`simulate_log_paths(..., local_vol=...)` replaces the cumulative sum with a
sequential loop, because the increment over a step now depends on the spot at
its start.

This has to be said plainly because the module's own docstring advertises the
opposite for the flat path. A constant-coefficient log step integrates GBM
**exactly** over any step size, so `steps_per_year` there is purely a schedule
for the observation dates. Log-Euler with state-dependent volatility is biased,
and `steps_per_year` becomes an accuracy parameter.

**But the bias is not measurable at practical path counts.** Averaging six
400,000-path runs per configuration against the Black-76 reference, it sits
within 1.7 standard errors of zero at 63, 252 and 1008 steps per year with no
trend — bounded below 1.3e-04, or 0.11% of price. Real in principle, dominated
by sampling error in practice. The daily knock-in monitoring already pins the
step count at ≥252, so nothing was added but loop overhead and one vectorised
volatility evaluation per step.

The degenerate check agrees to 2.9e-15 and is honest about its limits: for a
*constant* volatility log-Euler is already exact, so it validates that the loop
reproduces the cumulative sum and is silent about bias by construction.

### The scalar-time contract

`local_vol(S, t)` is **vectorised over `S` and scalar in `t`**. That is a
contract, not an accident: the variance term structure depends only on `t`, so
one call per time level costs one spline evaluation for the whole grid. Calling
with a meshgrid instead would turn thousands of spline evaluations into
millions.

---

## Grid sizing: the flat budget does not carry over

At the flat grid defaults the two engines disagreed by **9.4 standard errors** on
a 510050 one-year snowball — against Phase 1's 1.94 for the flat engines.
Refining the PDE closed it monotonically:

| nodes × steps/yr | gap (MC std errors) | cost at T=1 |
|---|---|---|
| 800 × 504 | −30.9 | |
| 1600 × 1008 | −14.5 | 0.3 s |
| 3200 × 2016 | −4.5 | |
| 3200 × 4032 | −2.3 | 1.9 s |
| **4800 × 8064** | **−2.0** | 4.7 s |
| 6400 × 8064 | −1.9 | 5.6 s |

So the engines do agree; the flat grid budget simply cannot be inherited. A
Dupire surface has a rough variance term structure and resolving it costs
resolution the constant operator never needed. `LOCAL_VOL_PDE_CONFIG` is set at
the point where the comparison stops being limited by the grid, which is what a
validation configuration is for. The residual two standard errors is the Monte
Carlo's own draw: a 400,000-path run puts the same comparison at 0.9.

**None of this is about local volatility as such.** The smoother 510300 surface
already agrees at the flat defaults — 0.2 and 0.02 standard errors at one and
two years. It is about how rough one particular fitted term structure is.

### Boundary placement

Four at-the-money standard deviations under-pads badly once there is a skew:
1.37e-02 probability of finishing beyond the lower boundary on the 510050 fit at
three years, because implied volatility at the boundary strike is far above at
the money. So `boundary_moneyness` solves the fixed point
`y = ∓n·√(w(y,T))`, measuring the z-score at the boundary itself.

But that fixed point is **self-reinforcing** under a steep skew: on 510300 it
runs to `y = −6.38`, leaving a 1600-node grid 37 intervals across the barriers.
Capped at 8× at-the-money total volatility, which lands at 1.3e-04 and 1.1e-03.

| cap × atm | 510050 | 510300 |
|---|---|---|
| 4 | 2.99e-03 / 162 intervals | 1.07e-02 / 139 |
| **8** | 6.05e-05 / 85 | 7.25e-04 / 73 |
| none | 2.58e-06 / 68 | 5.26e-07 / 37 |

Eight is chosen because the *inherited* flat grid, which Phase 1 validated to
within two confidence widths, sits at **7.1e-03** by this same measure — worse
than any row above. **Reach probability is therefore a diagnostic, not a gate:**
what carries a boundary is the accuracy of the condition imposed there.

Under constant volatility the fixed point lands at 4× atm, the cap never binds,
and the rule reduces to `∓4σ√T` exactly (0.0 and 4.4e-16) — which is what lets
the constant-volatility stand-in land on the flat grid node for node.

---

## The wings diverge at short maturity, at a derivable rate

As `t → 0` the at-the-money local volatility converges to a finite level (0.150
on 510050) but away from the money it diverges. Not a defect: at zero time to
expiry an option struck away from spot has no time value, so only an unbounded
local volatility can generate any smile.

The rate **follows from the power law** rather than being observed. As `θ → 0`
the curvature `φ ~ η·θ^−γ` blows up, so at fixed `y ≠ 0` the square root is
dominated by `|φy|` and `w ~ θ·φ·|y| ~ θ^(1−γ)`. With `θ` linear in `t` near the
origin, `w ~ t^(1−γ)` and `∂w/∂t ~ t^−γ`, while the denominator tends to a
constant. Hence

```
σ_loc ~ t^(−γ/2)
```

***not*** `t^(−1/2)`, which is what a first reading of the formula suggests and
what the test originally asserted before the measurement corrected it:

| γ | predicted | measured |
|---|---|---|
| 0.2 | 1.5849 | 1.6150 |
| 0.3 | 1.9953 | 2.0054 |
| 0.4 | 2.5119 | 2.5135 |
| 0.5 | 3.1623 | 3.1605 |

(ratio for a hundredfold drop in `t`, at `y = −0.357`)

So the cap genuinely binds in the far wing at the very short end, where it is
also harmless — those nodes sit many standard deviations from spot and
contribute nothing to a price at spot. Every hit is counted and reported.

### Numerical conditioning

Max cell Péclet number (`|advection|·dx / (2·diffusion)`) on a three-year
snowball grid: **0.022** at 400 nodes, **0.0054** at 1600 — two orders of
magnitude below the oscillation threshold of 2. Central differencing is safe and
no upwinding is needed; the only oscillation source remains the discontinuous
barrier payoff, already handled by Rannacher.

`MIN_TIME` is one second, deliberately five orders of magnitude below a daily
step. Theta is read from two adjacent PDE levels, so a floor anywhere near `dt`
would hand both the same frozen coefficients and silently zero the
term-structure contribution to the one Greek this project anchors most
carefully.

---

## Risk

Delta, gamma and theta needed no change: the first two are differenced off the
solution surface and theta off two adjacent time levels, none of which cares
where the volatility came from. Both of the others were bumping a market field
that no longer drives anything.

### Vega: shift the surface

`θ_i → (√(θ_i/T_i) + δ)²·T_i`, keeping `ρ`, `η` and `γ`; re-check no-arbitrage;
reprice. That is the sensitivity a desk hedges, because a desk hedges the
implied surface.

Bumping `market.sigma` instead would have been **worse than useless**. It does
not touch the diffusion here — but it *does* set the flat grid's padding, so the
bump would change the grid and return a small number produced by
discretisation. A plausible, meaningless vega is harder to catch than a zero,
and there is a test that bumping sigma from 0.05 to 0.60 leaves the price
exactly equal.

**This is a different derivative from the flat vega, not the same one measured
better.** SSVI's curvature scale depends on the variance level, so a parallel
shift of the at-the-money level also tilts the smile at every maturity. That is
what re-calibrating to a shifted at-the-money level would give, and it is the
right answer, but the two should not be compared — hence the method tag
`pde-grid+ssvi-parallel-vega`.

Measured: the shift survives comfortably on both real surfaces, with butterfly
margins of 2.20–2.77 even at ±5 vol points, so the no-arbitrage guard is a real
guard rather than an obstacle. It is tested firing on a constructed inverted
term structure.

### Rho had a sign error

Found by running the degenerate check on **all five Greeks** rather than just the
price. Four matched the flat engine to 1e-17 and rho did not: **+0.0036 flat
against −0.0001 local**. The drift lives on the surface's forward curve, so
bumping `market.r` moved only the discounting and left rho missing its drift
half — enough to flip the sign.

`with_rate_shift` fixes it, and the interesting part is *what it moves*.
Forwards implied from the option chain are **observations** and do not respond
to a hypothetical rate; only the extrapolated tail, built from the caller's own
`r − q − basis`, does. With that, rho matches the flat engine to 2.1e-16.

The temptation was to document rho as "discount-only on this path" and move on.
It would have been a true sentence about a wrong number.

---

## Measured effect on a snowball

103/80 barriers, 20% coupon, against a flat benchmark at each surface's own
at-the-money implied volatility with matched drift:

| | 1 year | 2 years | 3 years |
|---|---|---|---|
| 510050 | −32.3% | −7.9% | +4.9% |
| 510300 | −47.6% | −4.3% | +14.3% |

The sign flip with tenor is consistent with what the surface says: local
volatility at the money is well *below* the at-the-money implied (0.139 against
0.162 on 510050) while at the 80% knock-in it is roughly double. A shorter note
loses coupon from suppressed early knock-out; a longer one gains from the same
effect outlasting the knock-in risk.

---

## Defects found while building this

| # | Defect | How it was caught |
|---|---|---|
| 1 | `ForwardCurve.drift()` off by one segment — `searchsorted` returns the *knot* index | Integrated drift vs the curve's own log-forward, off by a constant 1.62e-03. Inspection did not catch it |
| 2 | **Rho sign error** (+0.0036 → −0.0001) | Degenerate check on all five Greeks, not just the price |
| 3 | **Phase 2a screening**: 510050's 1-day expiry cleared the 4-point screen with exactly 4 quotes, fit *worse* (0.645 → 0.587 vp), pinned γ at its bound, and kinked θ(T) (`max|dσ/dt|` 10.61 → 3.18) | Round-trip error 5.82 vp with it, 3.49 without |
| 4 | Strike-on-node shift subtracted the remainder, leaving `m·dx + 2·rem` | Its own test |
| 5 | My first round trip inverted implied vol at **spot** instead of the forward, showing 2–5.5 vp of fake error | Running the degenerate constant-vol test *first*, which came back clean at 1e-5 and sent me to the test rather than the engine |

Defect 5 is the methodological point of the phase: **localising tests first,
decisive test second.** Without the constant-volatility check I would have spent
that time looking for a bug in the operator.

Also tried and only partly kept: grading the time mesh by equal increments of
total variance, the natural substitution for the `t^−γ` singularity. It is on by
default in the European engine and buys about 20% — real, but not the fix I
expected, which is why the singularity is not the headline. The snowball solver
deliberately does **not** grade: its grid must put every observation on an exact
step, and getting observation dates wrong was a measured Phase 1 defect worth
more than a convergence order. Both docstrings say so, so the difference does
not read as an oversight.

---

## Known limitations

- **86.5% of a three-year horizon is extrapolated.** The chains stop at 0.41
  years. Out there the variance term structure continues at the last quoted
  secant rate, which is a convention and not market information. No grid
  refinement touches it, and on this data it is the largest single assumption
  behind any long-dated number — larger than everything else here combined.
  Reported as `extrapolated_fraction` and led with in the report.
- **The forward curve has a kink** at the last quoted expiry, +0.054 to +0.088.
  The acknowledged cost of reverting to the caller's carry.
- **The Monte Carlo is biased in principle.** Bounded below 1.3e-04 and not
  measurable at practical path counts, but the old "exact for GBM" claim does
  not carry over.
- **Local-vol vega is not comparable with flat vega.** Different derivative;
  see above.
- **Local volatility is calibrated, not dynamic.** Dupire reprices today's
  vanillas by construction but implies a forward smile that flattens, which is
  known to misprice forward-starting and cliquet features. A snowball's
  knock-out is not forward-starting, so this is second-order here — but it is a
  real limitation of the model class, not of the implementation. A stochastic
  volatility model is Phase 3's business.
- **Flat volatility remains the default.** The Phase 1/1.5 anchors continue as
  regression baselines, and nothing on the flat path moved.
- **Shark fin is untouched.** It still runs on the legacy `cn_fdm.py` and its
  prices have not been cross-validated; giving it a local volatility before
  that would be the wrong order of work.

---

## References

- Dupire, B. (1994). *Pricing with a smile.* Risk 7(1).
- Gatheral, J. (2006). *The Volatility Surface.* Wiley. — the
  forward-moneyness form of the local variance, eq. 1.10.
- Gatheral, J. and Jacquier, A. (2014). *Arbitrage-free SVI volatility
  surfaces.* Quantitative Finance 14(1).
- Andersen, L. and Brotherton-Ratcliffe, R. (1998). *The equity option volatility
  smile: an implicit finite-difference approach.* Journal of Computational
  Finance 1(2). — finite differences on a local volatility.
- Rannacher, R. (1984). *Finite element solution of diffusion problems with
  irregular data.* Numerische Mathematik 43. — the start-up damping.
