"""
Train the LTV model for E-CUP 2026 task 3.

Metric is RMSLE, so every model is fitted with plain L2 loss on log1p(target).
That makes the training objective identical to the metric, and predictions just
go back through expm1 with a clip at zero -- no retransformation correction.

Two things beyond a plain GBDT matter here:

1. Time holdout. The anchor 2026-01-14 (target 2026-01-15..2026-02-13) is never
   used for fitting, and training anchors stop at 2025-12-15 so that no training
   target window overlaps the holdout window.

2. Anchor-level seasonality. The ratio (next 30d GMV)/(previous 30d GMV) swings
   between 0.76 and 1.13 across anchors, and nothing in the per-user features can
   predict it. Training labels are divided by that anchor factor so the trees fit
   user-level signal instead of macro noise, and predictions are multiplied back
   by the factor expected for the target window.
"""

from __future__ import annotations

import argparse
import os
import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / os.environ.get("CASE3_WORK", "work")
OUT = ROOT / ("out" if os.environ.get("CASE3_WORK", "work") == "work"
               else "out_" + os.environ["CASE3_WORK"])

DATA_END = date(2026, 2, 13)
TEST_ANCHOR = DATA_END
VAL_ANCHOR = date(2026, 1, 14)
HORIZON = 30
MAX_TRAIN_ANCHOR_FOR_VAL = VAL_ANCHOR - timedelta(days=HORIZON)  # 2025-12-15

# Expected (next 30d)/(previous 30d) GMV ratio for the test window.
# Feb 14 - Mar 15 is a high season (Feb 23 / Mar 8); the same calendar transition
# one year earlier ran at 16,731,754 / 14,389,481 = 1.1628. A year-over-year
# check agrees: YoY growth is a steady ~1.43-1.50, and 1.46 x 16.73M = 24.4M
# against a 21.0M previous window, i.e. 1.163 again.
# Calibration applied to test-window predictions, as a direct multiplier on
# expm1(model output) -- NOT divided by the anchor base, because it was measured
# end-to-end rather than derived.
#
# The a-priori seasonal estimate was 1.163 (Feb 14 - Mar 15 is a high season, and
# the same calendar transition a year earlier ran at that ratio), giving a scale
# of 1.143. Submitting both settings showed it wrong: 1.143 scored 1.65486 on the
# public board, 1.0 scored 1.65059. A parabola through those two points, with the
# curvature measured on the holdout, puts the optimum at 0.989.
#
# The seasonal part was fine -- the ratio between holdout and test optima (1.41)
# matched the predicted 1.47. What it missed is a further ~12% overprediction,
# largely the truncated-window bias described in the README. The two nearly
# cancel, which is why a plain 1.0 wins.
TEST_SCALE = 1.0

DROP = {"user_id", "target", "anchor_ord"}

# From a random search over 26 configs on the holdout. The whole search space
# spanned only 0.0017 RMSLE, so these are a mild improvement over a hand-picked
# config rather than a decisive one -- the ceiling here is features, not fitting.
LGB_PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.02,
    num_leaves=255,
    min_data_in_leaf=20,
    feature_fraction=0.25,
    bagging_fraction=0.7,
    bagging_freq=1,
    lambda_l1=0.0,
    lambda_l2=5.0,
    path_smooth=1.0,
    max_bin=255,
    num_threads=16,
    verbosity=-1,
)


def rmsle(y_true, y_pred) -> float:
    a = np.log1p(np.clip(y_true, 0, None))
    b = np.log1p(np.clip(y_pred, 0, None))
    return float(np.sqrt(np.mean((a - b) ** 2)))


def anchor_files() -> dict[date, Path]:
    return {date.fromisoformat(p.stem.split("_")[1]): p for p in sorted(WORK.glob("anchor_*.parquet"))}


def season_ratios(files: dict[date, Path]) -> dict[date, float]:
    """Realised (next 30d)/(previous 30d) total-GMV ratio for every labelled anchor."""
    out = {}
    for a, p in files.items():
        d = pl.read_parquet(p, columns=["gmv_s30"]) if a == TEST_ANCHOR else pl.read_parquet(p, columns=["gmv_s30", "target"])
        if "target" not in d.columns:
            continue
        out[a] = float(d["target"].sum() / d["gmv_s30"].sum())
    return out


# Constant within an anchor, so they identify the anchor rather than the user.
# `anchor_ord` was already in DROP for exactly this reason; `hist_days` is
# anchor_ord + 1 by construction and was missed. The ya_* family is 100% null
# for every anchor before 2025-12-03 -- about three quarters of the training
# rows -- because the window a year earlier falls before the data starts, yet
# it is fully populated at the test anchor.
DRIFT = ["hist_days", "ya_cov", "ya_gmv", "ya_ord", "ya_cart", "ya_days",
         "ya_ord_days", "ya_gmv_share", "ya_vs_recent"]


def load_stack(anchors: list[date], files: dict[date, Path], ratios, base, deseason, tau: float = 0.0,
               top: int = 0, drop_drift: bool = False, drop_re: str = "", rule: bool = False):
    """Stack anchors into one float32 matrix, one anchor at a time.

    Concatenating the polars frames first would double peak memory, which at 40+
    anchors is the difference between fitting in RAM and not.
    """
    feats = [c for c in pl.read_parquet_schema(files[anchors[0]]) if c not in DROP]
    if drop_drift:
        n0 = len(feats)
        feats = [c for c in feats if c not in set(DRIFT)]
        print(f"dropped {n0 - len(feats)} drifting columns")
    if drop_re:
        # An audit flagged that the 365-day family is degenerate early on: data
        # starts 2025-01-01, so at the first anchor a "365-day" sum covers the
        # same 174 days as the 180-day one and the two columns are identical for
        # every user. The ratio climbs to 1.75 by January and the test anchor
        # sits at 1.80, so training does cover almost the whole range -- this
        # flag exists to measure whether the remaining mismatch actually costs
        # anything, rather than to assume it does.
        import re as _re
        pat = _re.compile(drop_re)
        n0 = len(feats)
        feats = [c for c in feats if not pat.search(c)]
        print(f"dropped {n0 - len(feats)} columns matching /{drop_re}/")
    if top:
        # Cutting the net's inputs from 417 to the 150 highest-gain columns was
        # the single largest gain of the project, so the same question is worth
        # asking of the trees: feature_fraction=0.25 means every split search
        # samples ~104 columns, and if 267 of the 417 are near-noise then most
        # candidates at every node are junk.
        n_all = len(feats)
        ranked = [c for c in json.load(open(OUT / "lgb_importance.json")) if c in feats]
        assert len(ranked) >= top, f"importance file ranks {len(ranked)} of {n_all} columns"
        feats = ranked[:top]
        print(f"restricted to top {top} of {n_all} features by gain")
    # The organisers picked the 250k users for being active in three consecutive
    # 30-day blocks ending at the data cutoff, so at the TEST anchor every user
    # satisfies that rule by construction. At a historical anchor only 72-93% do,
    # and the quarter that does not has a mean log1p target lower by ~0.4. Training
    # on them means a quarter of every batch is a population that cannot appear at
    # test. `rule` keeps only the rows that would have qualified at their anchor.
    keep = {}
    if rule:
        for a_ in anchors:
            keep[a_] = np.load(ROOT / os.environ.get("CASE3_WORK", "work") / "rule" / f"{a_}.npy")
        n = int(sum(keep[a_].sum() for a_ in anchors))
        print(f"rule filter: {n:,} of {250_000 * len(anchors):,} rows kept "
              f"({n / (250_000 * len(anchors)):.1%})")
    else:
        n = sum(250_000 for _ in anchors)
    x = np.empty((n, len(feats)), dtype=np.float32)
    y = np.empty(n, dtype=np.float64)
    w = np.empty(n, dtype=np.float32)
    aid = np.empty(n, dtype=np.int32)      # which anchor each row came from
    newest = max(anchors)
    i = 0
    for a in anchors:
        d = pl.read_parquet(files[a])
        if rule:
            d = d.filter(pl.Series(keep[a]))
        k = d.height
        x[i:i + k] = d.select(feats).to_numpy()
        ya = d["target"].to_numpy().astype(np.float64)
        y[i:i + k] = ya / (ratios[a] / base) if deseason else ya
        # anchors closer to the prediction date describe a marketplace closer to
        # the one we predict on; tau=0 disables the weighting
        w[i:i + k] = 1.0 if tau <= 0 else np.exp(-((newest - a).days) / tau)
        aid[i:i + k] = anchors.index(a)
        i += k
        del d
    assert i == n, (i, n)
    return x, y, w, aid, feats


def fit_lgb(xtr, ytr_log, xva, yva_log, num_round, seed, log_every=200, wtr=None):
    import lightgbm as lgb

    p = dict(LGB_PARAMS, seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2,
             data_random_seed=seed + 3)
    dtr = lgb.Dataset(xtr, label=ytr_log, weight=wtr)
    cbs = [lgb.log_evaluation(log_every)]
    valid = []
    if xva is not None:
        valid = [lgb.Dataset(xva, label=yva_log, reference=dtr)]
        cbs.append(lgb.early_stopping(200, verbose=True))
    return lgb.train(p, dtr, num_boost_round=num_round, valid_sets=valid, callbacks=cbs)


def fit_cat(xtr, ytr_log, xva, yva_log, num_round, seed, wtr=None):
    from catboost import CatBoostRegressor, Pool

    m = CatBoostRegressor(
        loss_function="RMSE", iterations=num_round, learning_rate=0.06, depth=8,
        l2_leaf_reg=6.0, random_seed=seed, od_type="Iter", od_wait=200,
        thread_count=16, verbose=200, allow_writing_files=False, border_count=128,
    )
    tr = Pool(np.nan_to_num(xtr, nan=-999.0), ytr_log, weight=wtr)
    va = Pool(np.nan_to_num(xva, nan=-999.0), yva_log) if xva is not None else None
    m.fit(tr, eval_set=va, use_best_model=va is not None)
    return m


def fit_twopart(xtr, ytr_used, xva, yva, sval, num_round, seed, wtr=None, anchor_id=None):
    """P(y>0) x E[log1p(y) | y>0].

    Algebraically E[log1p(y)] = P(y>0) * E[log1p(y)|y>0], so this targets the
    same quantity as the direct regressor but lets a dedicated classifier carry
    the zero-inflation. Measured on the holdout it beats the direct model by
    0.0004 (paired t = 2.4), and blending the two beats either.
    """
    import lightgbm as lgb

    p = dict(LGB_PARAMS, seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
    nz = ytr_used > 0

    # Dividing the target by a positive scalar cannot change which targets are
    # zero, so the de-seasonalisation that fixes the *level* leaves the
    # *incidence* untouched. But P(y>0) relative to the recent ordering rate is
    # 1.04-1.07 across training anchors and 0.96 at the holdout, so a classifier
    # pooled over anchors learns "incidence rises ~5%" and applies it where it
    # falls. Feeding each anchor's own log-odds as an init_score makes the model
    # learn how users rank *within* an anchor, free of its base rate; scoring
    # then happens at the pooled rate, which the level calibration handles.
    init = None
    if anchor_id is not None:
        init = np.zeros(len(nz), dtype=np.float64)
        pooled = float(nz.mean())
        lo_pooled = np.log(pooled / (1 - pooled))
        for a in np.unique(anchor_id):
            m = anchor_id == a
            pa = float(np.clip(nz[m].mean(), 1e-6, 1 - 1e-6))
            init[m] = np.log(pa / (1 - pa)) - lo_pooled
    va_sets = {}
    if xva is not None:
        va_sets["clf"] = lgb.Dataset(xva, label=(yva > 0).astype(np.int8))
        m = yva > 0
        va_sets["reg"] = lgb.Dataset(xva[m], label=np.log1p(yva[m] / sval))

    dclf = lgb.Dataset(xtr, label=nz.astype(np.int8), weight=wtr, init_score=init)
    n_clf, n_reg = (num_round if isinstance(num_round, (list, tuple))
                    else (num_round, num_round))
    clf = lgb.train(dict(p, objective="binary", metric="binary_logloss"),
                    dclf,
                    num_boost_round=n_clf,
                    valid_sets=[va_sets["clf"]] if xva is not None else [],
                    callbacks=[lgb.early_stopping(200, verbose=False)] if xva is not None else [])
    reg = lgb.train(p, lgb.Dataset(xtr[nz], label=np.log1p(ytr_used[nz]),
                                   weight=None if wtr is None else wtr[nz]),
                    num_boost_round=n_reg,
                    valid_sets=[va_sets["reg"]] if xva is not None else [],
                    callbacks=[lgb.early_stopping(200, verbose=False)] if xva is not None else [])
    return clf, reg


def raw_predict(model, x, kind: str) -> np.ndarray:
    """Model output in log1p space."""
    if kind == "cat":
        return model.predict(np.nan_to_num(x, nan=-999.0))
    if kind == "two":
        clf, reg = model
        return clf.predict(x) * reg.predict(x)
    return model.predict(x)


def to_level(log_pred: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.expm1(log_pred), 0, None) * scale


def run_val(args, files, ratios):
    train_anchors = sorted(a for a in files if a <= MAX_TRAIN_ANCHOR_FOR_VAL)[::-1]
    train_anchors = sorted(train_anchors[::args.anchor_step])
    base = float(np.mean([ratios[a] for a in train_anchors]))
    print(f"train anchors ({len(train_anchors)}):", [str(a) for a in train_anchors])
    print(f"mean season ratio over train anchors = {base:.4f}")
    print(f"val anchor {VAL_ANCHOR} realised ratio = {ratios[VAL_ANCHOR]:.4f} "
          f"-> relative {ratios[VAL_ANCHOR]/base:.4f}", flush=True)

    t0 = time.time()
    xtr, ytr_used, wtr, atr, feats = load_stack(train_anchors, files, ratios, base, args.deseason, args.tau,
                                                 args.top_feats, args.drop_drift, args.drop_re,
                                                 args.rule)
    print(f"X train {xtr.shape} ({xtr.nbytes/1e9:.2f} GB) [{time.time()-t0:.0f}s]", flush=True)

    va = pl.read_parquet(files[VAL_ANCHOR])
    if args.rule_val:
        # score on the same population the test anchor has, not on all 250k
        va = va.filter(pl.Series(np.load(ROOT / os.environ.get("CASE3_WORK", "work")
                                         / "rule" / f"{VAL_ANCHOR}.npy")))
        print(f"validation restricted to {va.height:,} rule-satisfying users")
    xva = va.select(feats).to_numpy().astype(np.float32)
    yva = va["target"].to_numpy().astype(np.float64)
    sval = ratios[VAL_ANCHOR] / base

    print(f"\nRMSLE zeros         : {rmsle(yva, np.zeros_like(yva)):.5f}")
    print(f"RMSLE carry gmv_s30 : {rmsle(yva, va['gmv_s30'].to_numpy()):.5f}", flush=True)

    ytr_log = np.log1p(ytr_used)
    # The model predicts on the neutral seasonal scale, so early stopping has to
    # compare against a holdout label put on that same scale. Against the raw
    # label the constant seasonal offset inflates the error floor and stops
    # training far too early.
    yva_log = np.log1p(yva / sval) if args.deseason else np.log1p(yva)

    # accept either inline JSON or a path to a JSON file; PowerShell 5.1 strips
    # quotes when passing JSON to a native exe, so the file form is the safe one
    if not args.fixed_rounds:
        fixed = {}
    elif args.fixed_rounds.lstrip().startswith("{"):
        fixed = json.loads(args.fixed_rounds)
    else:
        fixed = json.load(open(args.fixed_rounds))
    if fixed:
        print(f"frozen round counts (no early stopping): {fixed}", flush=True)

    logs, iters, scores = {}, {}, {}
    for kind in args.models.split(","):
        s = time.time()
        print(f"\n=== {kind} ===", flush=True)
        # With a frozen round count the fit helpers get no validation set, which
        # switches early stopping off. Choosing the number of rounds on the same
        # anchor the score is then reported on is selection on the measurement
        # set, and it inflates every effect measured that way.
        f = fixed.get(kind)
        va_x, va_l = (None, None) if f else (xva, yva_log)
        va_y = None if f else yva
        if kind == "two":
            m = fit_twopart(xtr, ytr_used, va_x, va_y, sval, f or args.rounds, 42,
                            wtr=wtr, anchor_id=atr if args.incidence else None)
            iters[kind] = f or [int(m[0].best_iteration), int(m[1].best_iteration)]
        elif kind == "lgb":
            m = fit_lgb(xtr, ytr_log, va_x, va_l, f or args.rounds, 42, wtr=wtr)
            iters[kind] = f or int(m.best_iteration)
            imp = sorted(zip(feats, m.feature_importance("gain")), key=lambda z: -z[1])
            json.dump({n: float(v) for n, v in imp}, open(OUT / "lgb_importance.json", "w"))
            print("top 30:", ", ".join(f"{n}" for n, _ in imp[:30]))
        else:
            m = fit_cat(xtr, ytr_log, va_x, va_l, f or args.rounds, 42, wtr=wtr)
            iters[kind] = f or int(m.get_best_iteration())
        lp = raw_predict(m, xva, kind)
        logs[kind] = lp
        np.save(OUT / f"val_log_{kind}{args.tag}.npy", lp)
        np.save(OUT / f"val_log_{kind}{args.tag}__{VAL_ANCHOR}.npy", lp)
        # headline number: predictions put back on the holdout's seasonal scale,
        # which is what the equivalent test-time pipeline will do
        scores[kind] = rmsle(yva, to_level(lp, sval))
        scores[f"{kind}_noscale"] = rmsle(yva, to_level(lp, 1.0))
        grid = {round(g, 3): rmsle(yva, to_level(lp, g)) for g in np.arange(0.5, 1.45, 0.02)}
        bg = min(grid, key=grid.get)
        print(f"{kind}: RMSLE={scores[kind]:.5f} (scale={sval:.3f}) | "
              f"no-scale={scores[f'{kind}_noscale']:.5f} | "
              f"best empirical scale={bg} -> {grid[bg]:.5f} | iters={iters[kind]} [{time.time()-s:.0f}s]",
              flush=True)

    if len(logs) > 1:
        # pairwise over whatever was actually trained -- hardcoding lgb/cat here
        # crashed two runs when other model kinds were requested. blend.py does
        # the full weight search later; this is only a sanity read.
        ks = sorted(logs)
        cand = [(a, b, w, rmsle(yva, to_level(w * logs[a] + (1 - w) * logs[b], sval)))
                for i, a in enumerate(ks) for b in ks[i + 1:]
                for w in np.arange(0, 1.001, 0.05)]
        a, b, w, sc = min(cand, key=lambda z: z[3])
        print()
        print(f"blend {a}={w:.2f} / {b}={1-w:.2f} RMSLE={sc:.5f}")
        scores["blend"], iters["blend_w"] = sc, float(w)

    np.save(OUT / "val_y.npy", yva)
    np.save(OUT / f"val_y__{VAL_ANCHOR}.npy", yva)

    OUT.mkdir(exist_ok=True, parents=True)
    json.dump({"scores": scores, "iters": iters, "season_base": base,
               "val_ratio": ratios[VAL_ANCHOR], "deseason": args.deseason},
              open(OUT / "val_report.json", "w"), indent=2)
    print("\n" + json.dumps(scores, indent=2))


def run_final(args, files, ratios):
    train_anchors = sorted(a for a in files if a != TEST_ANCHOR)[::-1]
    train_anchors = sorted(train_anchors[::args.anchor_step])
    base = float(np.mean([ratios[a] for a in train_anchors]))
    scale = TEST_SCALE if args.deseason else 1.0
    print(f"train anchors ({len(train_anchors)}):", [str(a) for a in train_anchors])
    print(f"season base={base:.4f} -> prediction scale={scale:.4f}", flush=True)
    OUT.mkdir(exist_ok=True, parents=True)
    # blend.py must reuse *this* scale: the final models see a different anchor
    # set than the validation run, so their neutral scale differs too
    json.dump({"season_base": base, "scale": scale, "test_scale": TEST_SCALE,
               "train_anchors": [str(a) for a in train_anchors]},
              open(OUT / "final_meta.json", "w"), indent=2)

    xtr, ytr_used, wtr, atr, feats = load_stack(train_anchors, files, ratios, base, args.deseason, args.tau,
                                                 args.top_feats, args.drop_drift, args.drop_re,
                                                 args.rule)
    ytr_log = np.log1p(ytr_used)
    print(f"X train {xtr.shape} ({xtr.nbytes/1e9:.2f} GB)", flush=True)

    te = pl.read_parquet(files[TEST_ANCHOR])
    xte = te.select(feats).to_numpy().astype(np.float32)
    uid = te["user_id"].to_numpy()

    if args.final_rounds == "auto":
        # reuse the iteration counts the holdout run settled on, with a little
        # headroom because the final fit sees more anchors
        it = json.load(open(OUT / "val_report.json"))["iters"]
        # a list means the two-part model: keep each component's own count,
        # sharing max() trained the regressor ~50% past its optimum
        rounds_by_kind = {k: ([int(x * 1.15) for x in v] if isinstance(v, list)
                              else int(v * 1.15))
                          for k, v in it.items() if k in ("lgb", "cat", "two")}
        print(f"final rounds (auto, from val_report): {rounds_by_kind}", flush=True)
    elif args.final_rounds.strip().isdigit():
        # plain integer: same round count for every requested model. Avoids
        # passing JSON on the command line, which PowerShell 5.1 mangles.
        rounds_by_kind = {k: int(args.final_rounds) for k in args.models.split(",")}
    else:
        rounds_by_kind = json.loads(args.final_rounds)

    acc, n = np.zeros(len(uid)), 0
    for kind, rounds in rounds_by_kind.items():
        per_kind, k = np.zeros(len(uid)), 0
        for seed in range(args.seeds):
            s = time.time()
            print(f"\n=== final {kind} seed={seed} rounds={rounds} ===", flush=True)
            if kind == "two":
                m = fit_twopart(xtr, ytr_used, None, None, 1.0, rounds, 42 + 17 * seed,
                                wtr=wtr, anchor_id=atr if args.incidence else None)
            elif kind == "lgb":
                m = fit_lgb(xtr, ytr_log, None, None, rounds, 42 + 17 * seed, wtr=wtr)
            else:
                m = fit_cat(xtr, ytr_log, None, None, rounds, 42 + 17 * seed, wtr=wtr)
            lp = raw_predict(m, xte, kind)
            per_kind += lp
            acc += lp
            n += 1
            k += 1
            print(f"done [{time.time()-s:.0f}s]", flush=True)
        # keep each model's log-space test prediction so blend.py can reweight
        # them later without refitting
        np.save(OUT / f"test_log_{kind}{args.tag}.npy", per_kind / k)
    np.save(OUT / "test_user_id.npy", uid)

    pred = to_level(acc / n, scale)
    OUT.mkdir(exist_ok=True, parents=True)
    sub = pl.DataFrame({"user_id": uid, "predict": pred})
    # Never clobber a submission that is already on the leaderboard: this
    # overwrote the live file three times in one day. Candidates get their own
    # name; promoting one is a deliberate step (see make_submission.py).
    out_path = ROOT / (f"submission_{os.environ.get('CASE3_WORK', 'work')}_raw.csv"
                       if (ROOT / "submission.csv").exists() else "submission.csv")
    sub.write_csv(out_path)
    # name the path it ACTUALLY wrote: this said submission.csv unconditionally,
    # which reads like the guard failed and sent me chasing a phantom overwrite
    print(f"\nwrote {out_path} rows={sub.height}")
    print(f"pred: mean={pred.mean():.3f} median={np.median(pred):.3f} "
          f"zeros={(pred<1e-6).mean():.4f} sum={pred.sum():,.0f} max={pred.max():,.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["val", "final"], default="val")
    ap.add_argument("--rounds", type=int, default=8000)
    ap.add_argument("--final-rounds", default="auto",
                    help='"auto" (reuse holdout iteration counts), a plain integer, or JSON')
    ap.add_argument("--models", default="lgb")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--deseason", type=int, default=1)
    ap.add_argument("--fixed-rounds", default="",
                    help="JSON like {\"lgb\":360,\"cat\":240,\"two\":[450,340]}. "
                         "Disables early stopping on the evaluation anchor -- "
                         "picking the round count there and then reporting the "
                         "score there is selection on the measurement set")
    ap.add_argument("--incidence", type=int, default=1,
                    help="per-anchor logit offset for the two-part classifier")
    ap.add_argument("--tag", default="",
                    help="suffix for saved predictions, so variants of the same "
                         "model do not overwrite each other")
    ap.add_argument("--val-anchor", default="",
                    help="override the holdout anchor, to check that gains are "
                         "not specific to 2026-01-14")
    ap.add_argument("--tau", type=float, default=0.0,
                    help="exponential half-life in days for anchor recency weighting; 0 = off")
    # Separated deliberately: to tell whether matching the TRAINING population to
    # the test one helps, both arms must be scored on the same evaluation
    # population -- otherwise the two changes are confounded.
    ap.add_argument("--rule", type=int, default=0,
                    help="filter TRAINING rows to users who satisfied the "
                         "organisers' 3-block activity rule at their anchor")
    ap.add_argument("--rule-val", type=int, default=0,
                    help="restrict the VALIDATION set to rule-satisfying users, "
                         "which is the population the test anchor actually has")
    ap.add_argument("--drop-re", default="",
                    help="regex; drop every feature whose name matches")
    ap.add_argument("--drop-drift", type=int, default=0,
                    help="drop anchor-identifier and year-ago columns (see DRIFT)")
    ap.add_argument("--top-feats", type=int, default=0,
                    help="keep only the N highest-gain columns (needs lgb_importance.json)")
    ap.add_argument("--anchor-step", type=int, default=1,
                    help="use every k-th anchor; anchors sit ~4-5 days apart, so"
                         " k=3 restores a ~14-day stride and drops near-duplicates")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True, parents=True)
    files = anchor_files()
    ratios = season_ratios(files)
    if args.val_anchor:
        global VAL_ANCHOR, MAX_TRAIN_ANCHOR_FOR_VAL
        VAL_ANCHOR = date.fromisoformat(args.val_anchor)
        MAX_TRAIN_ANCHOR_FOR_VAL = VAL_ANCHOR - timedelta(days=HORIZON)
        print(f"holdout overridden to {VAL_ANCHOR}", flush=True)
    print("anchors:", {str(k): round(v, 4) for k, v in sorted(ratios.items())}, flush=True)

    (run_val if args.mode == "val" else run_final)(args, files, ratios)


if __name__ == "__main__":
    main()
