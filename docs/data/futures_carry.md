# Carry from the futures board: what is measured, and what is not

A snowball coupon is paid for by volatility, the futures discount and cash
management. Until Phase 4 `basis_pa` was a user input defaulting to **zero**, so
the second of those three was absent from every price this project produced —
while 725 days of CFFEX index futures sat in the panel being used only to
validate forwards.

This file is about that measurement. The product it feeds is documented in
[`../models/snowball.md`](../models/snowball.md); the panel itself in
[`wind_panel.md`](wind_panel.md).

---

## The board shows a drift, not a basis

A future's settlement is the forward, so

```
log(F / S) / tau  =  drift  =  r − q − basis
```

is **one observable against three unknowns**. The same forward is consistent
with `q = 3%, basis = 0` and with `q = 0, basis = 3%`. Nothing on the board
separates the dividend yield from the financing spread from the discount.

So what is measured is the drift, which is observed, and what is reported is

```
carry = r − drift = q + basis
```

which is the single quantity both engines actually read — `dV/dq` and
`dV/dbasis_pa` are the same derivative, and the two bumped prices come back
**bit-identical**. It is fed in as `q=0, basis_pa=carry`. Splitting it further
would be inventing the split.

### The rate is a label, not an input to the price

I got this wrong on the first pass: I measured `−log(F/S)/tau` and called it the
carry, which is the carry at a **zero rate**. The offset is the whole rate — 2%
a year, against measured carries of 2% to 10%.

What keeps it from mattering is that the rate cancels. Feed `q=0` and
`basis_pa = r − drift` into an `EquityMarket` carrying that same `r`, and its
drift comes back as `r − (r − drift)` — the measured drift exactly, whatever `r`
was. So under `TailCarry.HOLD_LAST` every forward produced is invariant to the
rate, and there is a test asserting that across three rates 100bp apart: the
carry label moves, the forward does not.

The exception is `TailCarry.ZERO`, which is a claim about *carry* rather than
about the board, and so is the one tail rule whose forwards do move with the
rate. That is not a flaw in it — "no carry" has no meaning without a rate to be
no carry against.

The error was caught by the constant-carry degenerate: build the board from
`F = S·exp((r−c)·tau)` for one `c` and every accessor must return exactly `c`.
It failed for every non-zero carry.

---

## The front contract cannot be annualised

Annualising a small basis over a small tenor divides noise by a small number.
Front-contract drift across all 725 panel days, binned by its own tenor:

| front contract tenor | n | sd | range |
|---|---|---|---|
| `tau ≤ 0.03y` | 264 | **0.473** | −5.63 to +1.76 |
| `0.03 < tau ≤ 0.06y` | 222 | 0.109 | −0.97 to +0.52 |
| `0.06 < tau ≤ 0.10y` | 198 | 0.060 | −0.18 to +0.11 |
| spot-to-farthest | 672 | **0.026** | — |

A −563% annualised drift is one index point of basis divided by seven days. The
stable aggregate is **eighteen times tighter** than the raw front contract.

`MIN_FUTURES_TENOR = 0.05` rejects those contracts and counts them. Under a
constant carry the filter provably changes nothing, which is what separates
"discarding noise" from "discarding signal" — and no panel day loses its curve,
because four contracts list at a time and the rejected one is replaced by the
next.

---

## Half the note's life is not measured at all

The listed contracts reach **0.4984 years on average** (min 0.019, max 0.663).
Against the products this project prices:

| snowball tenor | fraction measured |
|---|---|
| 1 year | **48.5%** |
| 2 years | 24.2% |
| 3 years | **16.2%** |

So for a three-year note, **84% of the carry is an assumption** — and the carry
is the largest single input in the quote, larger than the volatility at that
tenor.

`TailCarry` is therefore a **required choice with no default**, for the same
reason `SharkFinTerms.monitoring` is: picking one silently is how a model comes
to hold a number nobody chose. The measurement that justifies it is blunt — on
the CSI 1000 board the three-year average carry is

```
10.154% a year under hold_last
 1.641% a year under zero
```

**A factor of six.** Not a refinement; a different trade.

The three rules:

| rule | meaning | when it is right |
|---|---|---|
| `ZERO` | no carry past the board; the forward grows at `r` | the project's old default, and what a desk assumes when it has no view |
| `HOLD_LAST` | hold the farthest measured carry flat | the natural reading of the screen — but it extrapolates six months out to three years |
| `EXPLICIT` | the caller supplies the number | when there is a house view, and it should be owned |

Passing `tail_carry` alongside a rule that computes its own is **refused**. Two
sources for one number is the defect pattern this phase kept finding.

`measured_fraction` is what the report leads with, before the level.

---

## The carry term structure is not flat

`|carry(far) − carry(near)|` exceeds 2% a year on **59.8%** of days for IF,
**54.9%** for IH and **70.1%** for IM. So a single scalar carry is a real
approximation rather than a formality.

The shape is kept as a `ForwardCurve` — reused, not reimplemented. That class
already holds a piecewise-constant drift with exact integration, an explicit
tail and a reported kink, and it was validated against the option-implied
forwards in Phase 2. `average_carry_to(T)` integrates it over the note's own
life, which is what weights measurement against tail assumption by how long
each applies.

One detail that is reported rather than reconciled: `ForwardCurve` infers its
own `F(0)` from the first listed *segment's* forward-forward drift, and the
index close is a different number. The curve here is anchored at the index close
— so the measured forwards come back exactly — and `anchor_gap` publishes the
difference, which measures how badly a forward-forward carry stands in for a
spot-to-futures one. On the real IM board it is −0.14%.

---

## `basis_pa ≥ 0` was refusing measured data

Spot-to-farthest carry over the panel, at the panel's own measured zero rate
(mean 1.57%):

| root | index | mean | sd | range | carry negative |
|---|---|---|---|---|---|
| IF | CSI 300 | +3.70% | 2.48% | −4.68% to +12.73% | 8.3% of days |
| IH | SSE 50 | +2.34% | 2.10% | −6.91% to +12.15% | **10.3%** |
| IM | CSI 1000 | **+10.35%** | 3.70% | +0.19% to +19.08% | **0.0%** |

`core/market.py` required `basis_pa ≥ 0`, which reads the market view — "the
A-share index futures persistently trade at a discount" — as a law rather than
as a tendency. It refused roughly **one day in ten** of measured IF and IH data.

The underlying market fact is larger than the carry statistic: **IH futures
trade above their index on 40.9% of panel days.** It is only the rate that lifts
most of those into a positive carry. Premium is a real market state — the SSE
50's dividend structure differs from the CSI 1000's — and clamping it to zero
would be editing the data to fit the assumption rather than the reverse.

The check is now on magnitude alone, `|basis_pa| < 1.0`. What is actually worth
refusing is a **unit error**: a basis handed over in percent (`6.0` meaning 6%)
is the realistic mistake, and a sign check never caught it. The measured range
is −6.9% to +19.1%, so that bound is two decades of headroom.

### The CSI 1000 is the exception, and it is the interesting one

Positive carry on **every single panel day**, averaging 10.35%. That is why
snowballs are written on it: at that carry the quotable coupon on a one-year
103%/75% note is **15.04%** against 6.53% at a zero carry.

---

## Using it

```
POST /api/carry
{ "underlying": "MO", "as_of": "2024-12-25", "tail": "hold_last", "rate": 0.02 }
```

`underlying` is the **option** code — `IO`, `HO` or `MO`. The futures root and
settling index come from the contract registry via `resolve_underlying`, not
from the request, so the pairing has one source. An ETF underlying is refused
with the reason: no index future shares its expiry calendar, which is what makes
this measurement interpolation-free for the index options and unavailable
elsewhere.

The response's `feed_as` block spells out `q=0, basis_pa=carry, r=<same rate>`,
because the rate cancels only if the same one is used on both sides.

The panel is institutional data and is **not committed to this repository**, so
a checkout without it gets a 400 naming `OTCPRICER_WIND_DIR` rather than a 500.
An absent data file is a configuration fact, not a solver failure.

---

## Not covered

- **Dividends separately.** By construction: the board cannot identify them.
  Anyone needing `q` alone needs a dividend forecast, which is a different data
  source and a different kind of claim.
- **The futures-versus-parity gap.** `validation/futures_check.py` measures it
  and deliberately does not feed prices. Phase 3 established that the gap does
  *not* identify the rate: closing the measured 0.08%–0.46% gaps would need
  2,508–35,654 basis points, and `dF/dr` even flips sign between expiries.
- **Intraday or term-structure-aware carry.** One number per root per day, read
  off settlements. Settlement is synchronous by construction, which is why it
  is used rather than the close — measured 5.7× better parity residuals.
