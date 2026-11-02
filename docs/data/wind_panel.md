# The Wind panel: 725 days of real market data

Phase 1 made the snowball **price** real, Phase 1.5 the **Greeks**, Phase 2a
the **surface**, Phase 2b the **local volatility**. All of it stood on two
things: a pair of single-day akshare captures, and a hardcoded 2% rate.

This phase replaces both. The captures stay — as the live path, and as the
in-repo offline fixtures — but they can only ever describe one afternoon.

---

## What arrived

| File | Rows | Carries |
|---|---|---|
| 期权基本信息数据 | 18,692 | Wind code, underlying, call/put, strike, maturity, contract unit, `ADJ_SIGN` |
| 期权日行情数据 | 1,332,440 | OHLC, **settlement**, volume, open interest |
| 指数日行情数据 | 2,900 | 000016 / 000300 / 000852 / 000905 |
| 期货基本信息数据 | 144 | IC / IF / IH / IM plus delist dates |
| 期货日行情数据 | 11,600 | Close, settlement, open interest |
| 国债收益率 | 749 | 1-year and 10-year CGB yields |

**725 trading days, 2023-06-30 to 2026-06-29.**

---

## Why the CFFEX index options became primary

|  | akshare ETF captures | CFFEX index (IO/HO/MO) |
|---|---|---|
| Usable expiries | 3 | **6** |
| Longest tenor | 0.41 y | **1.008 y** |
| Adjusted contracts | 1,812 of 11,372 | **0** |
| Parity residual | 0.08–0.69% of F | **0.02–0.08%** |
| Matching index future | none | **IF / IH / IM** |

The last row changes what can be *verified* rather than merely what can be
fitted. An index future shares its expiry date with the option — confirmed on
all 36 expiry tags, zero exceptions — so the futures settlement **is** the
forward to the option's own expiry. No interpolation, no basis adjustment.

The ETF path is kept but secondary, and the measured reason is blunt: on
2024-12-25, **140 of 510050's 220 rows are adjusted contracts** — 64% of the
chain — against zero for IO.

> **Equity only, and the data says so.** All 18,692 contracts have
> `S_INFO_EXERCISINGEND == S_INFO_MATURITYDATE`, i.e. European. The only
> non-equity products in the file are 1,308 treasury-futures options
> (TF/TS/TL/TT roots) which carry **zero quote rows**. There was never an
> American option here to price.

---

## Two conventions the data settled

### The price is the settlement, not the close

Put-call parity residuals on twelve IO slices across two dates, as a fraction
of the forward:

```
close    0.105  0.202  0.436  0.237  0.220  0.116
         0.211  0.290  0.318  0.267  0.241  0.161    mean 0.234%
settle   0.063  0.022  0.084  0.056  0.043  0.033
         0.021  0.031  0.016  0.044  0.045  0.031    mean 0.041%
```

The settlement wins **12 of 12, by 5.7×**. A settlement is one exchange-wide
reference struck on one convention, so it is synchronous *by construction*,
while closes are last trades at different instants — the exact defect
`quotes.py` names as dominant.

### An untraded contract must not look traded

**97,884 of the panel's 1,332,440 rows have zero volume and a non-zero close.**
Every single one. `OptionQuote.has_traded` asks whether the last price is
positive, so taken naively the whole untraded population reads as live quotes.

They are not stale carry-forwards either: only **2.7%** match the previous
day's settlement, and the median absolute daily move is **3.25%**. CFFEX is
marking untraded strikes *theoretically*. Fitting SSVI to those would fit the
exchange's own interpolation and call the result a market surface.

So `last` comes from the settlement only when volume is positive, and
`previous_settlement` is a genuine lagged join. Nothing in `quotes.py` changed,
and its promise of "a price from yesterday" became literally true.

---

## The rate curve, and what it is not

The panel supplies the 1-year and 10-year CGB yield. **That is two points, so
it is an interpolated curve and not a bootstrap** — there is no instrument
stack to solve, and the word is avoided throughout.

Linear in the zero rate between the knots, flat outside. The flat-below-1-year
half is the one that is tempting to get wrong: extrapolating the +6 bp/yr slope
down to zero *feels* principled but moves the short rate in the wrong
direction, since the very short CNY end sits below the 1-year.

Since every listed option expires inside 1.008 years, for surface construction
**this curve is "the 1-year yield, flat, refreshed daily"**. The 10-year point
only does work in a snowball's tail, where it is worth about 7 bp at two years.

**Why bother:** the 1-year yield ranged **0.83% to 2.40%**, so the old flat 2%
was off by up to 1.17 percentage points and sat more than 50 bp away on 60.6%
of days.

**But calibrate the expectation:** swapping the flat 2% for the real 0.894%
curve moves an IO fit's rms by **0.0034 vol points** and its at-the-money vols
by at most 0.23. This is a correctness fix, not an accuracy headline. The
accuracy came from the better option data.

Conventions: the file quotes percentages of an *annually compounded* bond
yield, so `r = log(1 + y/100)`. Worth 0.6–2.8 bp — free to get right,
expensive to find later.

---

## The futures check, and the premise it disproved

This was designed to check the **rate**, on the reasoning that a forward gap
growing with tenor is the signature of a mis-set rate. Measuring it says
otherwise.

The rate's leverage on the parity forward, IO at 2024-12-25:

| tau | dF / 100bp | as a fraction of F | gap vs the future |
|---|---|---|---|
| 0.063 | −0.013 | −0.0003% | +0.103% |
| 0.159 | −0.018 | −0.0005% | +0.162% |
| 0.236 | +0.274 | +0.0069% | +0.298% |
| 0.485 | +0.724 | +0.0182% | +0.457% |

Closing those gaps with the rate needs **+2,508, +4,346, −30,940 and −35,654
bp**; solving numerically returns −44.9%, +40.1%, or no root in ±50%. Worse,
the **sign of `dF/dr` flips between expiries**, so no rate adjustment moves
every expiry the same way.

The reason is structural: with `D` fixed, `F = mean[(C−P)/D + K]`, so
`dF/dr = T·mean(C−P)/D`. Near the money `mean(C−P)` is small and its sign
depends on whether the traded strikes sit above or below the forward.

**This reconfirms and extends Phase 2a.** Phase 2a found that implying a rate
from options alone is ill-posed (12.48%) and made the rate an input. Options
*plus futures* do not rescue it. Over the full panel, a plausible (sub-200 bp)
rate error accounts for at least one expiry's gap on only **12.4% of rows** —
and those are the rows with the smallest gaps.

What the gap *does* measure is total forward disagreement: 0.26–0.36% median,
2.5% worst. That is the error bar on the moneyness coordinate the surface is
fitted in, which nothing could measure before.

**Coverage is stated, not implied:** only four futures list against six option
expiries, so the two longest are never checked — including the tenor that
matters most for a one-to-three-year snowball.

---

## The 725-day anchor

`scripts/build_wind_anchor.py` sweeps the panel in about eleven minutes and
commits `data/wind/anchor.csv`, `manifest.json`, and five full chains.

| | median | p90 | worst |
|---|---|---|---|
| IO | **0.476** | 0.716 | 3.155 |
| HO | **0.565** | 0.873 | 2.806 |
| MO | **0.588** | 1.025 | 2.178 |

Volatility points, against the akshare captures' 0.587 and 0.654 — comparable
or better on ~3× the points and 2.4× the tenor. **All 2,070 fits converged and
all 2,070 are arbitrage-free.**

The denominator is 690 of 725 per root, not 725: the static contract file stops
before the quote panel does (79,078 rows across 1,704 contracts, all from
2025-09-22), so 35 dates fall below the two-slice minimum. 516 days carry all
six expiries.

### Two findings a single date cannot produce

**`gamma` sits at its upper bound on 46% of fits** (956 of 2,070). The SSVI
power law makes the short-maturity wing diverge like `t^(−gamma/2)`, so on
nearly half the panel that rate is set by the bound rather than by the quotes.
It bears directly on local volatility, where `gamma` governs the short-dated
wing. **This is an open limitation**, pinned as a test so relaxing the bound
has to be deliberate.

**The three indices have persistently different smile orientations**: HO quotes
a positive `rho` on 90% of days, IO on 69%, MO on only 32%. That is why a
positive `rho` on a single IO fit is data, not a sign error.

### How the artefact is kept honest

The manifest records **every screening threshold the sweep ran under**, and a
test asserts the live constants still match. The realistic failure is not code
rot — it is someone tuning `MAX_MONEYNESS_SD` and the committed distribution
silently ceasing to describe the code. The committed snapshots let the spot
checks refit and reconcile without the Wind drive, which no other machine has.

---

## What is still open

- **Extrapolation, improved but not solved.** 1.008 years of quotes against a
  three-year snowball is still **66% extrapolated**, down from 86.5%.
- **`gamma` at its bound on 46% of fits**, above.
- **Two of six expiries are never futures-checked**, and they are the long ones.
- **A government curve is not a funding curve.** An OTC desk discounts higher.
  The futures gap cannot measure the spread, for the reasons above, so the
  curve is labelled for what it is and no spread is invented.
- **The Wind files are not committed** (34 MB, institutional). Every panel test
  skips when they are absent; the akshare snapshots remain the in-repo fixtures.
