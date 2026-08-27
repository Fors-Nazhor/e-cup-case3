# My independent approach — written from the raw data, before reading README.md or src/

Anchor indexing used throughout: `d0 = 2025-01-01 = day 0`. Data covers days 0..408.
Target window `2026-02-14..2026-03-15` = days **409..438** (30 days). I call the first
target day the anchor `a`, so the test anchor is `a = 409` and features use days `< a`.
(The author's file naming uses the last *history* day, so their `2026-02-13` == my `a=409`,
their holdout `2026-01-14` == my `a=379`.)

## 1. What the signal is

All 250,000 users appear in the log, so there is no cold-start problem — every test user
has history.

**Variance decomposition** (13 disjoint 30-day windows, `log1p(gmv)`, window means removed
so macro trend does not contaminate):

| quantity | value |
|---|---|
| total var of detrended log1p target | 5.267 |
| between-user (stable latent level) | 2.829 |
| within-user (idiosyncratic) | 2.438 |
| **ICC** | **0.537** |

So slightly over half the variance is a persistent user-level propensity and the rest is
irreducible month-to-month noise. Consequences:

- **RMSLE floor if the user's true latent level were known exactly: 1.5615.**
- RMSLE of the best constant: 2.295.
- Leave-one-out (predict a window from the mean of the *other 12*, i.e. using the future —
  not achievable): **1.6866**.
- Causal OLS on the 12 disjoint monthly lags: **1.7063**.
- Causal ridge on my 143 hand-built features at `a=379`: **1.6727**.

The lag coefficients from that OLS are `0.230, 0.159, 0.109, 0.080, 0.054, 0.039, 0.032,
0.032, 0.025, 0.037, 0.026, 0.040` — a **slowly decaying long memory**. This is the classic
latent-rate / Gamma-Poisson signature: recent months matter most but the tail out to a year
still carries real weight, and the sum (~0.86) is well below 1, i.e. strong mean reversion.

46% of targets are exactly zero. Nonzero rate in a 30-day window has risen monotonically
from 0.37 (Jan 2025) to 0.54 (Feb 2026) — the platform is growing hard.

## 2. Target and loss

`log1p(gmv)` with plain **L2**. RMSLE *is* RMSE in log1p space, so this makes the training
loss literally the metric. Predict, clip at 0, `expm1`.

I would **not** use Tweedie, Poisson, or a hurdle/two-part model. `E[log1p(Y)] =
P(Y>0)·E[log1p(Y)|Y>0]` decomposes exactly, and direct L2 already estimates the left-hand
side without bias. A two-part model can only win through variance reduction, never through
correcting a bias — so I would expect it to be worth ~0 and would spend the time elsewhere.

I would expect (correctly) that the sum of predictions is far *below* the true total GMV,
because L2-on-log targets a conditional geometric mean. Any urge to rescale predictions so
the totals match is a direct loss.

## 3. Validation

**Rolling origin, multiple test anchors.** Train on anchors `≤ a_test − 30` so no training
target window overlaps the evaluation window. Evaluate at `a ∈ {289, 319, 349, 379}`.

The reason for *multiple* anchors is the number that matters most:

| comparison | SE |
|---|---|
| unpaired SE of RMSLE on 250k users | **0.0025** |
| paired SE between two models with residual ρ=0.996 | **0.0003** |
| **between-window spread of a model-difference** | must be measured, not assumed |

A paired test on one holdout window resolves 0.0003 — but it only tells you the effect *on
that window*. If an effect is worth +0.0006 on one window and −0.0004 on another, a
single-window paired t-test of 3.0 is meaningless for the test anchor. I would never accept
a sub-0.001 effect on one holdout.

## 4. Features I would build

Multi-scale windows (1/3/7/14/21/30/45/60/90/120/180/270/365) of gmv, `to_ord`, `to_cart`,
`searches`, active days, order days; **disjoint** monthly lags 1..12 and **disjoint** weekly
lags 1..8; recency of last activity / last order; tenure; lifetime aggregates *normalised by
available history days* (critical — history depth varies 183→409 days across anchors and a
tree splits on absolute thresholds); AOV; conversion ratios; short/long trend ratios;
inter-order-gap regularity (CV of gaps, median gap, days-since-last ÷ own median gap);
weekend share; concentration of GMV across days.

**Explicitly disjoint lags matter and are not substitutable by cumulative windows.** An
axis-aligned tree cannot compute `gmv_s60 − gmv_s30`; it can only threshold each. The
OLS evidence above (1.7063 with 12 disjoint lags vs 1.7397 with the best cumulative mean)
says the disjoint basis carries ~0.033 of linear signal that cumulative windows do not
express directly.

## 5. The macro / anchor-seasonality problem — the biggest single term

The 30-day window level moves a lot: mean `log1p(y)` runs 1.54 (Jan 2025) → 2.51 (Dec 2025)
→ 2.24 (Jan 2026). A model trained on past anchors and applied forward is **biased by a
window-specific amount**, and I measured it directly (LightGBM, 7 training anchors, 100k
user subsample):

| test anchor | RMSLE | mean residual (y−p) | RMSLE after optimal global shift | gain |
|---|---|---|---|---|
| 289 | 1.70503 | −0.0674 | 1.70370 | 0.00133 |
| 319 | 1.73821 | **+0.0453** | 1.73762 | 0.00059 |
| 349 | 1.74656 | −0.1289 | 1.74179 | 0.00476 |
| 379 | 1.69044 | **−0.2319** | 1.67446 | **0.01598** |

Two things follow:

1. This term dominates everything else on the list. At `a=379` it is worth 0.016 — five
   times the entire gap between rank 61 and rank 15.
2. **Its sign is not stable.** It is +0.045 at one anchor and −0.232 at the next. So it is
   not a systematic bias you can learn away; it is the realisation of a macro shock. The
   right move is to remove it from the *training labels* (so it stops injecting noise into
   the fit) and then set the test-time level to the most defensible central estimate — but
   you cannot expect to predict it, and you should not bet much on a seasonal extrapolation.

I would normalise each anchor's target by that anchor's macro coefficient and predict at the
"average anchor" level at test time, exactly because the direction of the shock is
unforecastable. (Feb 14 – Mar 15 in Russia contains Feb 23 and Mar 8, so a naive read says
"high season" and pushes the coefficient above 1 — but the daily data shows the effect is
mild: a pre-Mar-8 bump of ~1.10–1.18 for four days and a *dip* on the holidays themselves,
`rel=0.94` on Feb 23 and `0.92` on Mar 8. I would not bet 12% on that.)

## 6. What I would *not* pursue

- **Cross-user / graph structure.** There is no item, category, or query content in the
  schema — `search` and `cat` are per-day *channel* flags, not categories. The only
  cross-user coupling available is co-timing, which is exactly the macro term above. Nothing
  to mine.
- **Post-hoc calibration beyond a single constant.** The residual bias is a level shock, not
  a shape distortion.
- **Heavy hyperparameter search.** With ICC 0.5 and 250k effective users, the ceiling is set
  by information, not by fitting.

## 7. Achievable score, my estimate

The achievable band is narrow. Predicting the LOO oracle (1.687 on my detrended scale) is
already beaten by a good tabular model; the 1.5615 "known latent level" floor is
unreachable because it presumes infinitely many observations of a level that itself drifts.
My prior before reading their work: a good GBDT lands ~1.66–1.68 on a holdout window of this
type, and the spread between a competent solution and the best possible one is on the order
of **0.005–0.01**, most of which is the macro-level term in §5 rather than per-user
modelling.
