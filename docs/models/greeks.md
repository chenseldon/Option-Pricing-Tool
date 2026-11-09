# Structured-product Greeks

How the snowball sensitivities are computed, which methods were rejected and
why, and what each claim is verified against.

Vanilla Greeks are a separate matter: they are closed-form, checked against
QuantLib and against finite differences of the price, and documented in
`src/otcpricer/models/bsm.py`. This file is about the path-dependent products,
where the sensitivities are genuinely hard.

Flat volatility is the default basis throughout, and since Phase 2b the same
Greeks can be taken against a fitted Dupire surface — reachable from the API
since Phase 5. Where the two differ the difference is noted; where a sensitivity
means something different on the surface path, that is said rather than
smoothed over.

---

## Conventions

Matching the vanilla Greeks, so a portfolio can aggregate across both without
rescaling:

| Greek | Meaning | Unit |
|---|---|---|
| Delta | ∂V/∂S | per index point |
| Gamma | ∂²V/∂S² | per index point squared |
| Vega | ∂V/∂σ | per **1 percentage point** of volatility |
| Theta | ∂V/∂t | per **calendar day** |
| Rho | ∂V/∂r | per **1 percentage point** of rate |
| Carry rho | ∂V/∂carry | per **1 percentage point** of carry |

All values are fractions of notional, so a snowball gamma lands around 1e-07.
That is why the API serialises risk to 12 decimal places: the 8 used for prices
would leave gamma with two significant figures.

### What "delta" means here

Two different derivatives could both be called delta, and they are not
interchangeable:

**Spot delta (what is computed).** The barriers stay at the levels they were
struck at, and the market moves. This is the hedge ratio — the quantity of
underlying the desk trades — because a contract's barriers do not move when
the index does.

**Re-struck delta (not computed, and not useful).** Bump the initial fixing and
let the barriers re-scale with it. That is the derivative of a *different
contract* at each bump, and answers no question anyone asks.

The engines keep these apart structurally: `SnowballTerms.S0` is the initial
fixing and sets the barriers; `EquityMarket.S` is current spot. They coincide
only at inception. Conflating them was a real defect — the PDE read its price
at the fixing and therefore returned the same value at every spot (0.063087 at
4500, 5000 and 5500 alike), which also made a spot-bumped delta identically
zero.

---

## Production method: PDE grid differentiation

A finite difference solver evaluates `V` at every grid node on the way to the
price, so the value *function* is already in hand. The grid is uniform in
`x = log S`, so the chain rule gives

```
delta = (1/S) · ∂V/∂x
gamma = (1/S²) · (∂²V/∂x² − ∂V/∂x)
```

Derivatives are formed across the whole grid and then interpolated at spot,
rather than differenced around the nearest node. Spot generally falls between
nodes, and snapping to the nearest one introduces an error of up to half a grid
step that does **not** shrink under refinement.

Delta and gamma therefore cost nothing beyond the price. Vega and rho need the
parameter changed, so they are central differences over re-solves — clean,
because the solver is deterministic and contributes no noise.

### Theta, and why not by bumping the tenor

The observation schedule is derived from the tenor. Shortening `T` by one day
deletes an entire monthly knock-out observation:

```
T = 1.0        → 9 observations, last at 1.000000
T = 1.0 − 1/365 → 8 observations, last at 0.916667
```

A maturity-bumped theta would measure that contract change, not the passage of
time. Instead theta is read from two adjacent time steps of the *same* solve.
Backward induction already passes through `t = dt` on its way to `t = 0`, so
capturing it is free and the schedule is fixed by construction:

```
theta_per_day = (V(dt) − V(0)) / dt / 365
```

Verified against the zero-coupon anchor, where theta must equal `r·V/365`:
agreement to 1.9e-05 relative.

#### The simulation can do it too, by moving the valuation date

The paragraph above is the reason a *maturity-bumped* theta is wrong, and it
still stands. What it does not rule out is moving the **valuation date**, which
changes no term of the contract at all: the schedule and the coupon accrual stay
written from inception, so a knock-out at `t` still pays `coupon_pa · t` and the
monthly dates stay where the term sheet put them. Only the simulation's start,
the surviving observations and the discounting origin move.

So `price_snowball_mc` takes `t0`, and theta — the last Greek with no
independent confirmation — now has a second engine:

```
PDE   +0.00025218
MC    +0.00027274 ± 7.28e-05       gap 0.28 standard errors
```

The step is one of the simulation's own steps, so `dt` is identical in both runs
and the seasoned run reuses every increment but the first. That makes the
difference a genuinely **paired** estimator.

Which matters, because the obvious error bound is useless here. Quadrature on
the two runs' separate standard errors describes a pair of independent runs that
were never performed: it came to **347%** of theta's value at 252 steps a year
and **12,412%** at 1008 — numbers that say nothing about the estimate. The error
is therefore computed from the per-path difference, with antithetic pairs
averaged first.

That estimator is then checked against reality rather than trusted: run it over
several seeds and the seed-to-seed standard deviation must match the error each
run claims for itself. Measured **8.05e-05 observed against 7.28e-05 claimed**,
and the error halves as the path count quadruples. A quadrature bound would fail
that check by a factor of four.

Read the row with its caveat, though. The two numbers are difference quotients
of the same derivative over horizons **four times apart** — the PDE steps by its
grid's `dt` (1/1008 y), the simulation by one of its own (1/252 y) — so an O(h)
gap is expected before either is wrong. And three sampling errors come to about
80% of the value at 200,000 paths, against 27% for delta. The comparison
publishes `mc_error_share_of_value` so a reader can tell agreement from
confirmation.

### Carry rho, and why there is only one of it

`q` and `basis_pa` reach both engines only through their sum, `market.carry`.
So `dV/dq` and `dV/dbasis_pa` are not two close numbers — they are one number
computed twice, and the two bumped prices come back **bit-identical**, asserted
with `==` rather than a tolerance.

Reporting both would manufacture an independence the model does not have, and a
desk hedging each separately would double-count the same exposure. One Greek is
reported.

It is not `−rho`, and the gap is the discounting. `rho` moves the discount
factor *and* the drift; `carry_rho` moves the drift only, so their sum isolates
the discounting half:

```
rho        = +0.002729
carry_rho  = -0.003080      (drift half: +0.003080)
sum        = -0.000351      (discounting half)
```

The drift is **8.8×** the discounting on the reference trade, and
`carry_rho/rho` is −1.129 rather than −1. If it were −1 the Greek would be
redundant; that ratio is how far from redundant it is.

Why it is worth having at all: the carry is now measured rather than assumed,
and the measurement **moves**. Across the CFFEX panel the spot-to-far carry has
a standard deviation of 2.1% to 3.9% a year, against a policy rate that moved
about 70bp over the same three years. It is the larger of the two drift
exposures by an order of magnitude, and it had no sensitivity until Phase 4.

It lives on `SnowballRisk` rather than in `Greeks`, which is a fixed five-field
record shared with the vanilla products. A futures basis is not a vanilla
concept.

On the local-volatility path the drift comes from the surface, so the surface is
shifted too — by `with_carry_shift`, not `with_rate_shift`. The distinction is
sharper than it looks: `ForwardCurve` stores a rate and a drift and its carry is
their difference, so a rate shift moves both together and leaves the carry
*exactly* unchanged. Using it would have returned a clean zero while every other
Greek looked right.

### Why the holder's theta is positive

Unlike a long vanilla option, a snowball holder usually has **positive** theta:
the accrued coupon grows as time passes without a knock-in. That is the
economic point of the structure, not a sign error.

---

## Cross-check: Monte Carlo resimulation

Kept because two independent methods agreeing is the only reason to believe
either. A different formulation — sampled rather than solved.

Every valuation reuses one set of normal draws (**common random numbers**).
Without that, the difference of two independent simulations is swamped by
sampling noise; with it, the two runs differ only through the parameter and
most of the variance cancels.

The spot bump is 1%, larger than one would choose for a smooth payoff.
Deliberately: the payoff still steps as paths cross barriers, so too small a
bump leaves the difference dominated by that discreteness rather than by the
derivative.

### Gamma is not usable from Monte Carlo

A simulated gamma divides the second difference of a *discontinuous* estimator
by `h²`. The sampling error can exceed the value itself, and no path count
fixes it — the difficulty is in the estimator, not the sample size.

This is why `compare_snowball_risk` judges gamma against the simulation's own
standard error rather than demanding the signs match. Requiring sign agreement
would be incoherent given that the same comparison declares the estimator
unreliable. On a trade with a remote knock-in, gamma is about −2.9e-08 against
a Monte Carlo standard error near 2e-07: the simulated sign is a coin toss
while the two values are entirely consistent.

Monte Carlo theta returns `nan`, for the schedule reason above. A number would
be wrong; `nan` is honest.

---

## Rejected: pathwise differentiation

Pathwise differentiation exchanges the derivative and the expectation:

```
dV/dθ = E[ ∂(payoff)/∂θ ]
```

That exchange requires the payoff to be **Lipschitz** in the parameter. A
snowball's is not:

- on a knock-out the value jumps from its continuation value to `coupon · t`
- on a knock-in the entire leg switches from coupon accrual to a short put

So the method does not apply, and the previous implementation's "conditional MC
smoothing" was patching a method that does not hold rather than fixing a bug in
one that does.

The sound alternative is a **mixed estimator**: pathwise on the smooth part of
the payoff, likelihood ratio on the jump. It is materially more work and more
error surface, and is deliberately not implemented. Excluding a method honestly
is worth more than shipping a biased estimate described as validated.

---

## Rejected: the previous likelihood ratio implementation

The likelihood ratio (score function) method differentiates the *density*
rather than the payoff, so it does handle discontinuous payoffs — it is the
right family of method for this product. The implementation carried three
errors.

**1. Sign error in the volatility score.** The code had

```
∂log p/∂σ  =  −1/σ + innov²/(σ³T) + innov/σ        ← as written
∂log p/∂σ  =  −1/σ − innov/σ + innov²/(σ³T)        ← correct
```

Verified symbolically: with `μ = r − q − σ²/2` and
`log p = −log(σ√T) − (X − μT)²/(2σ²T)`, the corrected expression differs from
`∂log p/∂σ` by exactly zero, while the coded one differs by
`(−T(2q − 2r + σ²) − 2X)/σ`.

**2. A terminal score used for a path-dependent payoff.** The code used the
innovation over the whole life, `innov_T/(S₀σ²T)`. But in the joint density of
the path, only the **first** increment involves `S₀`, so the valid score is
`innov₁/(S₀σ²Δt)`. These are different random variables — their scales differ
by `√(T/Δt)`, a factor of 15.9 on a daily grid over a year — so
`E[payoff × score]` is not `dV/dS₀`.

**3. Rho missing its discount term.** With `V = e^{−rT}·E[payoff]`,

```
dV/dr = −T·e^{−rT}·E[payoff] + e^{−rT}·E[payoff × score_r]
```

Only the second term was computed. The first dominates rho.

On top of all three, every method was built on the Monte Carlo payoff that
carried the knock-in strike and discounting defects corrected in Phase 1.

A corrected likelihood ratio estimator would be a legitimate third view, and
the derivations above are recorded so it can be written later. It is not
implemented now because the PDE and the resimulation cross-check, anchored to
closed-form cases, already establish the numbers — and a high-variance third
estimate would add surface without adding confidence.

### What the three legacy methods reported

Delta on the reference trade (S₀ = 5000, 1 year, KO 103%, KI 75%, coupon 20%):

| Method | Legacy | Correct |
|---|---|---|
| Resimulation FD | −8.4e-05 | **+4.9e-05** |
| Pathwise | +1.0e-05 | **+4.9e-05** |
| Likelihood Ratio | +9.0e-06 | **+4.9e-05** |

All three were wrong. The first has the wrong sign; the other two are roughly a
fifth of the true value. They disagreed with each other in sign, so at most one
could have been right — and none was.

---

## Verification

### Anchor 1 — unreachable barriers

Place both barriers out of reach and the note becomes a zero-coupon bond paying
`c·T` at `T`. Every Greek is then known exactly, and three are exactly zero,
which cannot be passed by accident.

| Quantity | Exact | Measured |
|---|---|---|
| Price | 0.1960397347 | 0.1960397329 |
| Delta | 0 | 1.0e-11 |
| Gamma | 0 | −7.0e-14 |
| Vega | 0 | −5.2e-09 |
| Rho (per 1%) | −0.0019603973 | error 1.4e-09 |
| Theta (per day) | +1.0741903e-05 | error 2.0e-10 |

### Anchor 2 — terminal-only knock-in

With the knock-out unreachable and a single knock-in observation at maturity,
the note is European and has a closed form:

```
V(S) = e^{−rT}[ −N(d) + S·e^{(r−q)T}·N(d − σ√T)/K_put + c·T·(1 − N(d)) ]
d    = (ln(KI/S) − (r − q − σ²/2)T) / (σ√T)
```

with `KI` and `K_put` held fixed, so differentiating in `S` gives a spot delta.
Closed form: delta `+7.82267e-05`, gamma `−1.289855e-07`.

The PDE converges to it at clean first order, the error halving on each
refinement:

| Grid | Delta error | Gamma error |
|---|---|---|
| 800 × 504 | 8.71e-07 | 7.39e-10 |
| 1600 × 1008 | 4.36e-07 | 3.73e-10 |
| 3200 × 2016 | 2.18e-07 | 1.88e-10 |
| 6400 × 4032 | 1.09e-07 | 9.41e-11 |

### Structural checks

- Delta compared against the secant of the price curve it came from, which
  would fail if the log-space chain rule factors were wrong even though the
  prices are right.
- Gamma swept across spot through the knock-out barrier and required not to
  oscillate — the reference trade sits only 3% below it, which is where a grid
  method is most likely to misbehave.
- Both Greeks required to settle under joint grid refinement.

---

## A result that looks like a bug and is not

**Snowball delta changes sign with the knock-in level.**

| KI level | Price | Delta |
|---|---|---|
| 50% | +0.12482 | −7.73e-05 |
| 60% | +0.11880 | −6.02e-05 |
| 70% | +0.08960 | +3.01e-06 |
| 75% | +0.06309 | +4.95e-05 |
| 80% | +0.03373 | +9.23e-05 |
| 90% | +0.00142 | +1.22e-04 |

With the knock-in remote, the dominant exposure is that a rally triggers an
*early* knock-out, paying `coupon · t_ko` instead of the full `coupon · T`. The
holder is then effectively **short the upside**, and delta is negative. As the
knock-in moves closer, downside exposure takes over and delta turns positive.
The crossover sits between 60% and 70%.

A model that could not reproduce this would be missing the economics of the
product, so it is pinned as a test. It also means "a snowball holder is long
the underlying" is true of the reference trade but is **not** a universal law,
and no test asserts it as one.

---

## Not covered

- **Shark fin Greeks.** Its *price* is now cross-validated — Phase 3c migrated
  it to `engines/barrier.py`, found that the shipped analytic was wrong for
  every down barrier, and `cn_fdm.py` has since been deleted. But it still has
  no Greeks. The legacy three-method output was removed rather than carried
  forward, because it came from the same code whose snowball deltas were wrong
  in sign and was never validated for a barrier at all.
- **Heston Greeks.** A Heston snowball prices and has no sensitivities. That is
  deliberate: it is already the one price in this project with a single engine,
  and giving it Greeks would add five more numbers with nothing to check them
  against.
- **Vanna, volga, and cross-gammas.** Not implemented. The PDE surface supports
  them, but there has been no call for them yet.
- **Bucketed vega.** The flat path's vega is a parallel shift of one number; the
  surface path's is a parallel shift of the whole implied surface at the money,
  which also changes curvature at every maturity because SSVI's curvature scale
  depends on the variance level. **They are different derivatives**, and the
  `method` field on every response says which one was taken so they are not
  compared as the same quantity. Sensitivity to each *region* of the surface
  separately is the meaningful question once a surface is fitted, and is not
  answered here.
