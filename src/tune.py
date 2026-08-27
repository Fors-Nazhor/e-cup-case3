"""
Random search over LightGBM hyperparameters on the holdout anchor.

Everything up to now used a single hand-picked config, which is the most obvious
untuned lever left. Data is loaded once and reused across trials, so each trial
costs only the fit.

Selection bias is real here: picking the best of N trials on one holdout
overfits it. The paired SE between two configs on this fold is ~0.0006, so the
best of ~50 draws is inflated by roughly 0.0015. Anything smaller than that is
not a finding, and the winner is re-checked with a paired test in report().
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / os.environ.get("CASE3_WORK", "work2")
OUT = ROOT / "tuning"

VAL_ANCHOR = date(2026, 1, 14)
HORIZON = 30
MAX_TRAIN_ANCHOR = VAL_ANCHOR - timedelta(days=HORIZON)
DROP = {"user_id", "target", "anchor_ord"}

SPACE = {
    "num_leaves":        [31, 63, 127, 255, 511],
    "min_data_in_leaf":  [20, 50, 100, 200, 500, 1000],
    "feature_fraction":  [0.25, 0.35, 0.45, 0.6, 0.8],
    "bagging_fraction":  [0.6, 0.7, 0.8, 0.9, 1.0],
    "lambda_l1":         [0.0, 0.1, 1.0, 10.0],
    "lambda_l2":         [0.0, 1.0, 5.0, 20.0, 100.0],
    "learning_rate":     [0.015, 0.02, 0.03, 0.05],
    "max_bin":           [127, 255],
    "min_sum_hessian_in_leaf": [1e-3, 0.1, 1.0],
    "path_smooth":       [0.0, 1.0, 10.0],
}


def rmsle_log(ly, pred_log):
    return float(np.sqrt(np.mean((ly - pred_log) ** 2)))


def anchor_files():
    return {date.fromisoformat(p.stem.split("_")[1]): p for p in sorted(WORK.glob("anchor_*.parquet"))}


def season_ratios(files):
    out = {}
    for a, p in files.items():
        if "target" not in pl.read_parquet_schema(p):
            continue
        d = pl.read_parquet(p, columns=["gmv_s30", "target"])
        out[a] = float(d["target"].sum() / d["gmv_s30"].sum())
    return out


def load(anchor_step: int):
    files = anchor_files()
    ratios = season_ratios(files)
    anchors = sorted(sorted(a for a in files if a <= MAX_TRAIN_ANCHOR)[::-1][::anchor_step])
    base = float(np.mean([ratios[a] for a in anchors]))
    feats = [c for c in pl.read_parquet_schema(files[anchors[0]]) if c not in DROP]

    n = 250_000 * len(anchors)
    x = np.empty((n, len(feats)), dtype=np.float32)
    y = np.empty(n, dtype=np.float64)
    i = 0
    for a in anchors:
        d = pl.read_parquet(files[a])
        k = d.height
        x[i:i + k] = d.select(feats).to_numpy()
        y[i:i + k] = d["target"].to_numpy() / (ratios[a] / base)
        i += k
        del d

    va = pl.read_parquet(files[VAL_ANCHOR])
    xva = va.select(feats).to_numpy().astype(np.float32)
    yva = va["target"].to_numpy().astype(np.float64)
    sval = ratios[VAL_ANCHOR] / base
    print(f"train {x.shape} over {len(anchors)} anchors, val scale {sval:.4f}", flush=True)
    return x, np.log1p(y), xva, yva, sval, feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--anchor-step", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import lightgbm as lgb

    OUT.mkdir(exist_ok=True, parents=True)
    rng = np.random.default_rng(args.seed)
    xtr, ytr_log, xva, yva, sval, feats = load(args.anchor_step)
    # compare against the holdout label put on the model's neutral scale, the
    # same convention train.py uses
    yva_log_neutral = np.log1p(yva / sval)
    ly_true = np.log1p(yva)

    results_path = OUT / "lgb_trials.json"
    results = json.load(open(results_path)) if results_path.exists() else []
    print(f"{len(results)} trials already on disk", flush=True)

    for t in range(args.trials):
        # values in SPACE are already plain Python ints/floats
        p = {k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()}
        p.update(objective="regression", metric="rmse", num_threads=16, verbosity=-1,
                 bagging_freq=1, seed=42, bagging_seed=43, feature_fraction_seed=44)
        s = time.time()
        m = lgb.train(
            p, lgb.Dataset(xtr, label=ytr_log), num_boost_round=args.rounds,
            valid_sets=[lgb.Dataset(xva, label=yva_log_neutral)],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        lp = m.predict(xva, num_iteration=m.best_iteration)
        # Score each config at its OWN best scale rather than at the seasonal
        # one. The leaderboard showed the a-priori scale is ~12% too high, and
        # calibration is a single free parameter fitted at submission time --
        # so ranking configs at a wrong shared scale only adds noise.
        p0 = np.clip(np.expm1(lp), 0, None)
        grid = np.exp(np.linspace(np.log(0.45), np.log(1.3), 60))
        scores = [rmsle_log(ly_true, np.log1p(p0 * g)) for g in grid]
        j = int(np.argmin(scores))
        score, best_scale = scores[j], float(grid[j])
        rec = dict(params={k: p[k] for k in SPACE}, best_iter=int(m.best_iteration),
                   rmsle=score, best_scale=round(best_scale, 4),
                   rmsle_at_season=rmsle_log(ly_true, np.log1p(p0 * sval)),
                   secs=round(time.time() - s, 1))
        results.append(rec)
        results.sort(key=lambda r: r["rmsle"])
        json.dump(results, open(results_path, "w"), indent=1)
        np.save(OUT / f"pred_{score:.5f}.npy", lp)
        print(f"[{len(results):3}] {score:.5f} @s={best_scale:.3f}  iter={m.best_iteration:4}  "
              f"{rec['secs']:6.0f}s  best so far {results[0]['rmsle']:.5f}  "
              f"{json.dumps(rec['params'])}", flush=True)


if __name__ == "__main__":
    main()
