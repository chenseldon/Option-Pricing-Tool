# The snowball: five phases, and the defects each one found

The flagship product, and the last one to get a document — which is backwards,
and is the reason this file exists. The surface, the local volatility, Heston
and the barrier notes all had one while the thing actually quoted did not.

Greeks are a separate file: [`greeks.md`](greeks.md). Carry is a separate file:
[`../data/futures_carry.md`](../data/futures_carry.md). This one is about the
**price**, the **quote**, and the contract's **schedule**.

---

## What the contract is

A monthly-observed autocallable. Three outcomes, and a path resolves to exactly
one of them:

| State | Payoff | Paid at |
|---|---|---|
| Knocked out | `coupon_pa × t_ko` | `t_ko` |
| Knocked in, never out | `−max(0, 1 − S_T / strike)` | `T` |
| Neither | `coupon_pa × T` | `T` |

Knock-out is observed monthly after a lock-up (three months, the A-share
standard). Knock-in is observed **daily, for the whole life including the
lock-up** — the lock-up suspends early termination, not downside monitoring.

The holder is long a coupon stream and short a down-and-in put. That is why the
holder's theta is normally positive, and why the structure is written on the
most heavily discounted index available.

---

## Phase 1: the price was wrong in five separate ways

The two legacy engines were **18% apart** on the reference trade while the code
described them as cross-validation. Each defect below was found by making them
agree.

### Discounting ignored the payment date

The legacy engine applied `exp(−r·T)` to every path, including the majority
that terminate early. A knocked-out path pays at `t_ko` — measured expected
knock-out time on the reference trade is **0.45 years**, not 1.0 — so a year of
discounting was applied to payments made in six months.

Each path is now discounted at its own payment date. How much of the 18% that
one line was is no longer measurable — the legacy engine is deleted — so this
is recorded as one of five causes rather than ranked among them.

### Knock-in was monitored on the knock-out dates

Twelve observations a year instead of 252. A barrier checked monthly is
breached far less often than one checked daily, so the short put leg was
systematically too cheap and the note too valuable.

### The lock-up did not exist

The legacy model observed knock-out from month one. With a three-month lock-up
on a one-year note the schedule is **nine observations, not twelve**, and the
three extra ones are the most expensive to get wrong — because a knock-out in
month 1 pays a twelfth of the coupon while a knock-out in month 12 pays all of
it. Measured:

| lock-up | KO dates | price | P(knock out) | E[t_ko] |
|---|---|---|---|---|
| 0 months | 12 | 0.037122 | 73.2% | 0.2557 |
| **3 months** (standard) | 9 | **0.062788** | 65.2% | 0.4440 |
| 6 months | 6 | 0.089374 | 58.0% | 0.6445 |

Ignoring the lock-up **understates the note by 41%**. The mechanism is in the
last column rather than the fourth: the extra observations pull expected
knock-out from month five back to month three, so the note terminates on almost
no accrued coupon.

I first wrote that the three extra dates were "the ones most likely to trigger".
That is backwards — under a positive drift a later observation is *more* likely
to sit above the barrier, not less. The table is what the claim should have been
in the first place.

### A knocked-in note was modelled as a plain short put

A note that knocks in and later recovers above the knock-out barrier still
terminates and collects its coupon. Treating the knocked-in state as a pure
short put drops that coupon.

Measured: **0.12% of paths** take that route, and omitting them left the PDE
**0.46% below** the Monte Carlo. Small, and one-signed, which is the
combination that survives inspection — it looks like noise until the two
engines are made to agree to three digits.

### The grid did not hold the contract

Two misalignments, both of which made the price **oscillate under refinement
rather than converge**. That is the symptom worth recognising: refinement that
does not settle means the grid is changing which contract it prices.

Observation dates not landing on time steps:

```
252 → 504 → 1008 steps/year:   0.06253, 0.06311, 0.06275
```

Non-monotone, moving by ~5e-04 with no trend. Fixed by snapping the step count
to a whole multiple of the knock-in frequency.

Barriers not landing on grid nodes:

```
200 → 3200 nodes:   0.06149, 0.06328, 0.06352, 0.06286, 0.06311
```

A barrier between nodes is represented by the nearest one, displacing it by up
to `dx/2` and degrading the scheme from second order to first. Fixed by
choosing `dx` to divide the knock-in/knock-out span exactly.

And a third, worse than either: the knock-in barrier was applied at **every
time step**, which makes monitoring finer as the grid is refined, so the engine
priced a progressively different contract:

```
63 → 1008 steps/year:   0.06678 → 0.06121, no sign of settling
```

---

## Phase 4: the note can be quoted

A desk does not ask what a note is worth at a 20% coupon. It asks what coupon
the note can carry — and that number *is* the quote. Until Phase 4 `coupon_pa`
was an input everywhere and the only par-solve in the repository was the
interest-rate swap.

### The price is exactly affine in the coupon

This is the whole design, so it is worth stating precisely. The knock-out and
knock-in events depend only on the spot path against fixed barriers — **never
on the coupon** — and every coupon-bearing term in both engines is linear in
it. So

```
price(c) = A·c + B           exactly
A = E[ disc · (t_ko on knock-out paths + T on clean paths) ]
B = E[ disc · knock-in loss ]                                  (negative)
par coupon = −B / A
```

`A` is the snowball's **coupon annuity** — the discounted expected time the
note spends alive and paying — which is the same object `products/irs.py`
already uses to get a par swap rate analytically instead of by bisection.

Measured residual against that straight line across six coupons from 0 to 1:
**7.8e-16**. The Monte Carlo's is *identically zero*, because a fixed seed
gives bit-identical paths across coupons.

### Three solves, so the affinity is an invariant

Two solves answer the question. The third, at `c = 0.5`, checks that the
question was the right one: `|0.5A + B − price(0.5)|` is computed on every call
and raises above 1e-12.

That matters more than the 50% it costs. A coupon-dependent grid or boundary
appearing in the engine would make `−B/A` simply the wrong answer — returned
with full confidence and no symptom — and every other test would still pass,
because they all rest on the same assumption. The test injects a quadratic
break of 1e-9 of price, far below any tolerance elsewhere in the project and
still three orders above this limit, and the solver refuses it.

### The answer carries the grid's own bias

Exact in the coupon is not exact in the price. The par coupon inherits the
grid's first-order convergence:

| grid | par coupon |
|---|---|
| 400 × 252 | 7.2902% |
| 800 × 504 | 7.1441% |
| 1600 × 1008 | 7.1187% |
| 3200 × 2016 | 7.1069% |
| 6400 × 4032 | 7.1009% |

Differences halve, so the Richardson limit is about **7.0948%** and the
1600×1008 quote is 2.4e-04 high — **1.4 Monte Carlo standard errors** at
400,000 paths. `/api/snowball_quote` therefore defaults to 3200 × 2016, four
times the work of the pricing routes, and says so in its response.

### What a reference implementation does instead

`carrey325/Autocall-pricing-and-risk-engine` solves this by Brent bisection
over `[0.01, 0.40]`: roughly thirty solves for an answer two give exactly. And
worse — an iterative solver converges to a *tolerance*, so it can never reveal
that the relationship is linear. Here the affinity is the test.

---

## What actually pays for the coupon

Three drivers, and two of them are now measured rather than assumed. At 20%
volatility on the reference trade, all at `/api/snowball_quote`'s own
**3200 × 2016** grid so the rows are comparable:

| input | par coupon |
|---|---|
| zero carry | 6.5276% |
| IH carry, measured +1.83% | 7.6221% |
| IF carry, measured +3.40% | 8.6891% |
| **IM carry, measured +10.15%** | **15.0155%** |
| IH at a 2% premium | 5.4928% |

So the futures carry **more than doubles** the quotable coupon. The smile does
something comparable: on the fitted 510300 surface the same note quotes
**11.75%** against 6.53% flat, **+80%** — though that row cannot share the grid,
because a Dupire surface needs 4800 × 8064 for the same accuracy and comparing
it on the flat grid would be comparing discretisation errors.

Neither is a refinement. A quote computed at a zero carry on a flat volatility
is not a rough version of the real one.

(The solver's own docstring quotes 6.54% and 15.04% for the first and fourth
rows. Those are the same measurements at 1600 × 1008, where the par coupon runs
2.4e-04 high — see the convergence table above. The difference between the two
sets *is* that bias.)

The carry relationship is also **not linear**: `d(par)/d(carry)` averages 0.599
over `[0, 1.83%]` and 0.838 over `[0, 10.15%]`, with a local slope of 1.196 at
the IM level — 1.997× the near-end average. A desk extrapolating the near-end
sensitivity to a deep discount would understate the coupon by a third, which is
why `decompose_par_coupon` reports a difference of two solves rather than a
sensitivity times a level.

---

## The schedule is part of the contract

Both engines used to drop observation dates the time grid could not hold, and
they dropped them **differently**.

At 252 steps a year, with `obs_freq` of 252, 300, 400, 600 and 1000 — notes
asking for 189, 225, 300, 450 and 750 knock-out observations — the Monte Carlo
returned

```
0.04795316632228654
```

for all five. **Bit-identical prices for five different contracts.** The PDE
kept the later contractual date per step and paid the contractual accrual where
the simulation paid the grid's, so across the same five it returned five
different prices spread over 4.22e-04 with no order to them.

That difference is what made this invisible: the cross-engine check was
comparing two substituted contracts, and both substitutions moved the price the
same way.

Worth distinguishing from the Phase 1 defects above. Those made the price
*jitter* under refinement — visibly wrong. This is quieter and **one-signed**: a
note with fewer knock-out chances is worth strictly more, so dropping dates
flatters the quote. It is also not a divisibility problem — `obs_freq=5` with
`ki_obs_freq=252` converges first-order and monotonically, because the dates
still land on distinct steps. The requirement is capacity, not a common factor.

Both engines now refuse, report the same collision count, and name the grid
that works. Every frequency a term sheet carries — monthly, fortnightly,
weekly, daily — fits a 252-step year, so this is a gate rather than a
constraint on anything real.

---

## Phase 5: the last legacy number

`/api/cn_fdm` with `calc_greeks=false` still ran `CN_FDM.price()`. That class
ignored the lock-up, the knock-in strike mode and the basis, and its own PDE and
Monte Carlo struck the knock-in put at different levels — so the two halves of
one class priced different contracts.

```
legacy branch      0.021250
reviewed engine    0.038599
gap               -0.017349   (44.9% of price)
```

A caller reached that by passing one flag, and the flag was supposed to control
whether Greeks were computed. **Asking for Greeks moved the price by 45%.**
`cn_fdm.py` is deleted; a test asserts the two branches now return the *same*
price, which is the property that says they are one engine.

The same route reported a barrier it had not priced: it defaulted `KO_pct` to
1.05 for its response echo and the parser defaulted it to 1.03 for the pricing,
so a request omitting the field was priced at 5150 and told it had 5250. The
echo is now read off the priced `SnowballTerms`, which also carries
`lockup_months`, `ki_strike_mode` and `basis_pa` into a response for the first
time — all three had been priced since Phase 1 and were invisible.

---

## What is verified, and against what

| Claim | Evidence |
|---|---|
| Price | PDE against Monte Carlo, agreeing inside a budget built from the simulation's measured error and the grid's own refinement change |
| Unreachable barriers | Reduces to a risk-free bond, `coupon_pa · T · exp(−rT)` |
| Terminal-only knock-in | Reduces to a European short put, closed form |
| Affinity in the coupon | Collinear to 7.8e-16 across six coupons; enforced per call |
| Par coupon | Monte Carlo reprices to zero at the solved coupon, 0.54σ |
| Knocked-in surface | Both engines, 0.13% relative — first cross-check of that state |
| Local volatility | Constant-surface degenerate reduces to the flat engine to 1e-9 |
| Carry | Futures board, [`futures_carry.md`](../data/futures_carry.md) |

The hardest gate is `tests/test_snowball_local_vol.py`'s
`FLAT_PRICE = 0.0683263509055402`, asserted with `==`. Flat volatility is this
project's default pricing basis, and any change that moves that constant has to
explain itself.

---

## Not covered

- **Product variants.** No 降敲 (step-down), 保本 (principal-protected), 早利
  (early-coupon) or 不追保 (no-margin-call) snowball. A deliberate decision:
  the classic note was made impeccable first.
- **Heston.** A Heston snowball prices, and it is the one number in this
  project with a single engine. Its second engine would be a two-dimensional
  PDE with a cross-derivative and an ADI solver, which is not built — and the
  response says so rather than leaving it to be assumed.
- **Bucketed vega.** Vega on the surface path is a parallel shift of the
  at-the-money level. Sensitivity to each region of the surface separately is
  the meaningful question once a surface is fitted, and is not answered here.
