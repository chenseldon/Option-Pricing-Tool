# The shark fin: two engines, and three defects they found

Until this phase the shark fin was priced by two unconnected pieces of code
that disagreed about what the contract **was**:

- `/api/shark_fin` evaluated the Rubinstein–Reiner closed form **inline in the
  Flask route**, touching no engine, and so modelled a *continuously* monitored
  barrier.
- `cn_fdm.py` solved a grid truncated at the barrier, and so modelled something
  closer to a *discretely* monitored one, at whatever frequency the grid
  implied.

Neither was wrong in isolation. The problem is that they are prices of
**different contracts**, and nothing said so. On the legacy test case the daily
note is worth **43.02** against **35.17** continuous — a 22% difference.

So `monitoring` is now a required field with no default. Choosing one silently
is how the project came to hold two disagreeing prices.

---

## Defect 1: the analytic formula was wrong for half its cases

Against QuantLib's `AnalyticBarrierEngine`:

| kind | K | H | before | QuantLib | error |
|---|---|---|---|---|---|
| up-and-out call | 100 | 120 | 1.12969334 | 1.12969334 | **2e-14** |
| up-and-out call | 100 | 130 | 3.09770631 | 3.09770631 | 2e-14 |
| down-and-out put | 100 | 80 | 4.87975297 | 1.82112396 | **3.06** |
| down-and-out put | 100 | 70 | 6.46143223 | 4.72554238 | **1.74** |
| down-and-out put | 110 | 90 | 6.58113901 | 1.09472507 | **5.49** |

The cause is Haug's `η`: the reflection terms carry **−1 for an up barrier and
+1 for a down one**, and the shipped code applied the up sign unconditionally.

It had only ever been checked on an up-and-out call — while carrying the comment
*"validated against CN-FDM PDE (error < 0.01%)"*. That comment was true of the
case it had been run on and false of the formula.

All sixteen cases — both directions, both knock senses, two volatilities — now
agree to **≤3.5e-14**.

> This is what the two-engine discipline is for. The defect was invisible to
> any amount of internal consistency checking, because both of the project's
> own shark fin implementations shared the assumption.

---

## Defect 2: solving across the barrier costs an order of convergence

Carrying the grid past the barrier and re-imposing the dead zone at each step
*looks* equivalent to imposing a Dirichlet boundary at the barrier. It is not:
the operator couples nodes **through** the barrier within each step, so the
barrier is effectively smeared over one `dx`.

| nodes | error, projected | error, truncated at barrier |
|---|---|---|
| 400 | 0.135 | 8.9e-04 |
| 800 | 0.065 | 2.1e-04 |
| 1600 | 0.040 | 5.0e-05 |
| 3200 | 0.023 | **8.8e-06** |

The left column merely halves — first order — and is still **3% of price at
3200 nodes**. The right column quarters, which is what Crank–Nicolson should
do. For a continuously monitored barrier the grid now stops **at** the barrier
and its Dirichlet boundary *is* the barrier.

---

## Defect 3: Rannacher damping has to follow every observation

Two fully implicit steps suppress the oscillation Crank–Nicolson shows against
a discontinuous initial condition. The usual placement is at the start of the
backward solve — and for a continuous barrier that is enough, since the solution
stays smooth thereafter.

A **discretely** monitored barrier re-creates a discontinuity at the barrier
node at *every* observation. Damping only the first two steps of the whole
solve leaves 251 of 252 undamped.

Caught by the PDE sitting **8.5 standard errors** from the Monte Carlo on a
daily down-and-out put. Checked across eight seeds, that was the PDE's error
and not the simulation's — the Monte Carlo is *exact* for this contract, since
the log-Euler step is exact for GBM and the observations land on step
boundaries.

### Two mistakes of mine in the same work

**I made every step an "observation" for the continuous case**, which made the
new damping rule fire every step, left the scheme fully implicit throughout, and
destroyed the second-order convergence I had just achieved.

**I dropped the `H`-versus-`K` domain guards** while rewriting the analytic, so
an up-and-out call with `H ≤ K` raised an error instead of returning the zero it
is worth. (The old route *also* raised there, which is wrong in a different
way: it is a fact about the contract, not an unsupported input.)

---

## What discrete monitoring actually costs

| monitoring | price | premium over continuous |
|---|---|---|
| 4 / year | 1.6337 | +0.958 |
| 12 / year | 1.2432 | +0.568 |
| 52 / year | 0.9515 | +0.276 |
| 252 / year | 0.8065 | +0.131 |
| 1008 / year | 0.7506 | +0.075 |
| 4032 / year | 0.7153 | +0.040 |
| continuous | **0.6754** | — (analytic) |

The premium roughly **halves as the frequency quadruples**, which is the
`1/√m` rate Broadie, Glasserman and Kou give for the discrete barrier
correction. So the engine reproduces a known analytic result rather than merely
trending the right way.

Quarterly observation is worth **2.4×** the continuous price; daily is worth
**1.19×**. Conflating them is not a rounding matter.

### A convergence order stated honestly

Discrete monitoring converges only **first order** in `dx` — expected, since
every observation re-creates a discontinuity. Measured on a daily down-and-out
put: 1.3381, 1.3529, 1.3609, 1.3649, 1.3669 as nodes doubled.

Richardson-extrapolating gives **1.3690** against a Monte Carlo mean of
**1.3711 ± 0.0012** — a 1.7σ gap, i.e. consistent. The test tolerance is set
from that measurement rather than from optimism.

---

## Knock-ins come from parity, not a second solver

A knock-in needs to know whether the barrier has already been touched — a second
state variable, and a second grid dimension. Instead:

```
knock-in + knock-out = vanilla
```

exact for any monitoring frequency when the rebate is zero, since every path
ends in precisely one of the two. A knock-in **with** a rebate is refused rather
than silently mis-split, because the rebate belongs to the knock-out leg only.

The PDE parity test is marked **tautological in its own docstring** — the
knock-in is computed as the residual, so the identity holds whatever the
knock-out solve did. The Monte Carlo version is not tautological: each leg is
computed from its own paths, so it genuinely verifies that the two branches
partition the sample.

> That test also had to be fixed. It first compared the simulated sum against
> the **analytic** vanilla, which measures sampling error — 0.0317, about 1–2
> standard errors. It now asserts exactness against the vanilla *on the same
> paths*, and compares to the closed form separately with a tolerance from its
> own standard error.

---

## On the local volatility surface

This is where the smile matters most in the whole project, for a structural
reason: **a barrier payoff is concentrated at one spot level**, so it depends on
the volatility *there* rather than on an average. A flat number cannot express
that, and its error is one-sided in a direction set by which side the barrier is
on.

Measured on the 510300 surface against a flat volatility at the surface's own
at-the-money level, so the comparison isolates the smile rather than the choice
of level:

| contract | flat @ ATM | local vol | difference |
|---|---|---|---|
| 1y up-and-out call, 110% | 0.00689 | 0.01335 | **+93.7%** |
| 1y up-and-out call, 120% | 0.05652 | 0.06542 | **+15.8%** |
| 1y down-and-out put, 90% | 0.00916 | 0.00952 | +3.9% |
| 1y down-and-out put, 80% | 0.08602 | 0.05457 | **−36.6%** |

The **signs** are what the tests assert. On a downward-skewed equity surface the
volatility at an up barrier is below the at-the-money, so an up-and-out knocks
out less often and is worth more; at a distant down barrier it is above, so a
down-and-out knocks out more often and is worth less. A price-level check would
not catch the smile being applied backwards. A sign check does.

There is no closed form for a barrier under a local volatility, so the two
engines are the only cross-check available, and they agree within three standard
errors. The constant-local-volatility degenerate reduces to the flat engine to
1e-9, which localises any future failure to how the surface is sampled onto the
grid rather than to the discretisation.

---

## The legacy path

`/api/cn_fdm` keeps working. Shark fin routes to the new engine at **252
observations a year**, which is what the legacy grid priced — now stated instead
of implied.

The legacy three-method Greeks are **removed** rather than carried forward. They
came from the same code whose snowball deltas were wrong in sign, and were never
validated for a barrier at all, so returning them beside a corrected price would
lend them credibility they never had. The response says so.

`core/types.BarrierKind` finally has a caller, having been declared since
Phase 1.
