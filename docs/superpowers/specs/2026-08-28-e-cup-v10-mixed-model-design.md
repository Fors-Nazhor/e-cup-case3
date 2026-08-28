# E-CUP 2026 Task 3: v10 Features and ROCm Mixed Model

## Goal

Improve the public RMSLE of `submission_v9_opt.csv` (1.649315) by changing the
per-user prediction shape, not by continuing to tune a single global multiplier.
The first-place reference supplied by the user is 1.6446514942, so a useful
candidate must close a measurable part of the roughly 0.00466 gap without
overfitting the 20% public split.

The deliverable is a reproducible training and submission pipeline with two
final candidates:

1. a conservative blend based on v9 plus components that improve several
   rolling holdouts;
2. a larger mixed-model blend with a higher weight on the new ROCm model when
   its out-of-time evidence supports that weight.

## Fixed Baseline and Hardware

- Baseline file: `submission_v9_opt.csv`.
- Baseline public RMSLE: 1.649315.
- Baseline provenance: it is exactly `submission_v9.csv * 0.95109674` and
  therefore changes only the v9 global scale.
- Training GPU: AMD Radeon RX 9070 XT (`gfx1201`), 17.1 GB VRAM, ROCm.
- System RAM on the training machine: 32 GB.
- Local Mac runs are limited to short CPU smoke tests and bounded analytical
  checks. Full training runs happen on the ROCm machine.

`submission_v9_opt.csv` remains immutable. New predictions and manifests use
new, explicit names and never overwrite a file that may already have been
submitted.

## Design Principles

1. Optimize the actual metric: the main prediction target remains
   `log1p(GMV over the next 30 days)` with an L2 loss.
2. Separate shape from scale. A model can improve user ordering while an
   unpredictable calendar-level shift changes the best global multiplier.
3. Prefer rolling evidence over a single paired test. Small improvements on one
   250,000-user window can be precisely measured and still fail to transfer to
   the next month.
4. Add information before adding capacity. The current model lacks explicit
   disjoint weekly and monthly histories even though an independent audit found
   a slowly decaying year-long memory.
5. Keep every experiment reproducible and resumable within the 32 GB RAM and
   17.1 GB VRAM limits.

## v10 Anchor Data

The existing per-anchor Parquet layout stays in place. Each anchor describes
history through day `t` and targets future days `[t+1, t+30]`. Files are built
one anchor at a time and may be resumed by skipping validated artifacts that
already exist.

### Disjoint lag features

Add non-overlapping histories alongside the existing cumulative windows:

- weekly lags 1 through 8: `[0,6]`, `[7,13]`, ..., `[49,55]` days before the
  anchor;
- monthly lags 1 through 12: `[0,29]`, `[30,59]`, ..., `[330,359]` days before
  the anchor.

For every lag, aggregate the core signals that have direct behavioral meaning:

- GMV;
- ordered items;
- cart additions;
- searches;
- active days;
- ordering days.

Channel-level Search/Catalog splits are added for the most recent four weekly
lags and six monthly lags. This bounds width and build time while retaining the
periods most likely to carry channel-mix changes.

### Long-memory summaries

Derive compact summaries from the disjoint lag matrix:

- exponentially weighted weekly levels at 2-, 4-, and 8-week half-lives and
  monthly levels at 1-, 3-, and 6-month half-lives;
- weighted and unweighted linear slopes;
- recent-versus-historical deviations;
- variability, nonzero fraction, maximum-period concentration, and entropy;
- ordering incidence and GMV conditional on an ordering period;
- user-level shrinkage estimates that pull sparse users toward the population
  mean according to their observed number of active periods.

All coverage-sensitive features include observed-period counts. Missing history
is not silently treated as a full zero period for early anchors.

### Auxiliary targets

In addition to the unchanged 30-day target, labeled anchors store four future
weekly blocks (days 1-7, 8-14, 15-21, and 22-30):

- weekly GMV;
- whether weekly GMV is positive;
- ordered items.

These labels are training-only and are excluded from every feature matrix. They
support auxiliary GPU losses but never replace the main RMSLE-aligned head.

## Candidate A: Controlled v10 Upgrade

Candidate A isolates the value of v10 information. It keeps the current model
families and comparable settings:

- direct LightGBM log regression;
- the existing two-part LightGBM model;
- the existing temporal CNN with the wider v10 tabular block.

CatBoost may be included in a final blend when its already-generated prediction
is available, but repeated CPU CatBoost training is not required for every
feature ablation. The controlled comparison uses the same anchor sampling,
seeds, stopping rule, and calibration protocol on v9 and v10.

## Candidate B: ROCm Mixed Challenger

The mixed network has three encoders whose outputs are concatenated before the
prediction heads.

### Local daily encoder

A pooling-based temporal CNN reads the most recent 180 daily steps. It
captures short bursts, recency, funnel events, and event order. Ordinary
convolutions plus downsampling are retained because dilated convolutions are
known to be very slow on the current `gfx1201` ROCm stack.

### Long-history token encoder

The full available history is converted into 59 fixed weekly slots and 14 fixed
30-day slots, with uncovered early-history slots masked out.
Each token contains the corresponding disjoint aggregates, a coverage mask, and
calendar-position features. The initial `mixed_base` profile uses width 160 and
three Transformer blocks. The explicitly larger `mixed_large` challenger uses
width 192 and four blocks. Both capture interactions across periods without
applying attention to all 409 individual days.

### Static tabular encoder

An MLP receives the v10 tabular features after signed `log1p`, train-fold-only
standardization, clipping, and explicit NaN handling. It supplies stable RFM,
conversion, tenure, and long-memory context.

### Heads and losses

The main head predicts expected `log1p(GMV30)` and uses MSE. Auxiliary heads
predict the four weekly GMV log-targets, weekly purchase incidence, and weekly
ordered items. The first run assigns total weights of 0.15, 0.05, and 0.05 to
those three auxiliary groups respectively, with loss averaged inside each
group. The auxiliary losses are disabled as a group in an ablation. Inference
uses only the main 30-day head.

The initial mixed model is deliberately modest. Width, depth, and auxiliary
weights are increased only after the end-to-end data and validation paths pass.
This avoids confusing a capacity change with a pipeline defect.

## Memory and Runtime Boundaries

- Anchor artifacts are generated and validated one at a time.
- Large arrays use memory mapping or chunked reads; stacking every wide anchor
  into multiple in-memory copies is prohibited.
- Training uses approximately 14-day anchor spacing rather than all overlapping
  offsets unless a measured experiment justifies the additional rows.
- GPU training uses mixed precision and configurable batch size.
- An out-of-memory error retries once with a smaller batch; other errors stop
  the run and preserve completed artifacts and logs.
- The ROCm preflight records torch/ROCm versions, GPU name, free VRAM, and a
  forward/backward smoke test before full training.
- CPU commands expose thread-count limits. Local Mac smoke tests use a user
  subset and never launch an unbounded full-data fit.

## Rolling Validation

The outer holdout anchors are:

- 2025-09-19;
- 2025-11-14;
- 2026-01-14.

For an outer holdout at `t`, training uses only anchors whose 30-day target
windows finish before the outer evaluation target begins. Near-duplicate
anchors are sampled at approximately 14-day spacing.

For each outer holdout `t`, define a calibration anchor `c = t - 30 days`.
The honest run trains through `c - 30 days`, predicts both `c` and `t`, fits one
global multiplier on `c`, and applies that multiplier unchanged to `t`. Thus
neither early stopping nor calibration observes the outer target. A separate
refit through `t - 30 days` may be used for the shape-only diagnostic, because
that diagnostic explicitly removes the outer level and does not claim a
deployable score.

Every candidate is reported in two views:

1. honest RMSLE using a scale derived only from earlier labeled windows;
2. shape RMSLE after allowing baseline and candidate the same one-parameter
   oracle calibration on the outer window. This second view is diagnostic and
   is never described as a deployable score.

For paired model comparisons, report the mean difference in squared log error,
its standard error, a confidence interval, and per-fold RMSLE. A change is
accepted only when:

- its average effect is positive;
- it improves at least two of the three outer holdouts;
- its equal-fold-weighted mean shape RMSLE improves by at least 0.0003;
- it does not degrade the newest January shape RMSLE by more than 0.0002;
- a user-block bootstrap of the equal-fold mean squared-log-error difference
  has a positive 95% lower confidence bound;
- its gain remains after weights and scale are optimized together.

Blend weights are learned on older validation predictions and evaluated on the
newest holdout. Weights fitted directly on January are not used to claim an
honest January improvement.

## Final Training and Submission Assembly

After model selection, refit accepted components on every allowed labeled
anchor with iteration counts fixed from validation. Assemble two named outputs:

- `submission_v10_conservative.csv`;
- `submission_v10_mixed.csv`.

The conservative candidate uses cross-window-stable weights and stays close to
v9. The mixed candidate may give the new ROCm model more weight, but only within
the range supported by validation. Both submissions inherit the known v9-opt
scale as a baseline reference; any relative scale adjustment must be derived
from out-of-time folds and written to the manifest. Specifically, compute the
candidate-to-v9 ratio of honest optimal scales on each outer fold, take the
median of those three ratios, and apply it relative to the fixed v9-opt scale.

## Reproducibility and Experiment Log

Create `docs/experiments.md` as an append-only human-readable ledger. Each run
records:

- hypothesis and exact command;
- git commit and feature schema hash;
- data/anchor set, model parameters, seed, and duration;
- peak host RAM and peak VRAM when available;
- per-fold metrics, paired statistics, and decision;
- paths and checksums for predictions and manifests.

Machine-readable JSON manifests accompany every validation prediction and CSV.
The README contains only the current best result, reproduction commands, and a
short table pointing to detailed experiment entries.

## Safety Checks and Tests

The pipeline must fail early on:

- a feature computed from a date after its anchor;
- an auxiliary target included in the model feature list;
- duplicate or missing users;
- any output with other than 250,000 unique required user IDs;
- negative, infinite, or NaN predictions;
- mismatched validation/test variants in a blend;
- a schema hash mismatch when resuming an anchor build.

Unit tests cover disjoint interval boundaries, coverage masks, auxiliary target
windows, scale fitting, paired-statistic calculations, and manifest validation.
A small end-to-end smoke test builds two anchors for a user subset, trains a
tiny CPU model, runs one mixed-network batch when torch is available, and
assembles a schema-valid submission fragment.

## Success Criteria

The implementation is complete when:

1. v10 artifacts and both candidate families can be reproduced from commands
   documented in the README;
2. all leakage, boundary, manifest, and smoke tests pass;
3. validation reports all three outer windows rather than only the January
   holdout;
4. final conservative and mixed CSV files pass submission validation and have
   complete manifests;
5. every accepted and rejected material experiment is recorded in
   `docs/experiments.md`.
