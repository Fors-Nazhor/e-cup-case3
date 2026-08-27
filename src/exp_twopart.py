"""
Does splitting the target into "will they buy" x "how much" beat direct L2?

L2 on log1p(target) estimates E[log1p(y)] directly, which is the RMSLE optimum,
so a two-part model is not needed *in principle*. But 46% of targets are exactly
zero, and the identity

    E[log1p(y)] = P(y>0) * E[log1p(y) | y>0]

decomposes exactly, so a dedicated classifier may estimate the first factor
better than a single regressor does implicitly. That is an empirical question,
and this settles it on the holdout instead of assuming.
"""

from __future__ import annotations

import os
import time
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / os.environ.get("CASE3_WORK", "work3")
VAL_ANCHOR = date(2026, 1, 14)
MAX_TRAIN = VAL_ANCHOR - timedelta(days=30)
DROP = {"user_id", "target", "anchor_ord"}

BASE = dict(num_threads=16, verbosity=-1, num_leaves=255, min_data_in_leaf=100,
            feature_fraction=0.25, bagging_fraction=0.7, bagging_freq=1,
            lambda_l2=5.0, learning_rate=0.02, max_bin=255, path_smooth=1.0,
            seed=42, bagging_seed=43, feature_fraction_seed=44)


def best_scaled(ly, lp):
    """RMSLE at the calibration constant that suits this prediction best."""
    p0 = np.clip(np.expm1(lp), 0, None)
    g = np.exp(np.linspace(np.log(0.45), np.log(1.35), 80))
    sq = [(ly - np.log1p(p0 * x)) ** 2 for x in g]
    j = int(np.argmin([s.mean() for s in sq]))
    return float(np.sqrt(sq[j].mean())), float(g[j]), sq[j]


def main() -> None:
    files = {date.fromisoformat(p.stem.split("_")[1]): p for p in sorted(WORK.glob("anchor_*.parquet"))}
    ratios = {}
    for a, p in files.items():
        if "target" not in pl.read_parquet_schema(p):
            continue
        d = pl.read_parquet(p, columns=["gmv_s30", "target"])
        ratios[a] = float(d["target"].sum() / d["gmv_s30"].sum())

    anchors = sorted(sorted(a for a in files if a <= MAX_TRAIN)[::-1][::3])
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
    ly = np.log1p(yva)
    print(f"train {x.shape} over {len(anchors)} anchors", flush=True)

    # --- reference: one regressor on every row ---
    t = time.time()
    m = lgb.train(dict(BASE, objective="regression", metric="rmse"),
                  lgb.Dataset(x, label=np.log1p(y)), num_boost_round=6000,
                  valid_sets=[lgb.Dataset(xva, label=np.log1p(yva / (ratios[VAL_ANCHOR] / base)))],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    lp_direct = m.predict(xva, num_iteration=m.best_iteration)
    r, g, sq_direct = best_scaled(ly, lp_direct)
    print(f"direct      {r:.5f} @ scale {g:.3f}  iter={m.best_iteration}  [{time.time()-t:.0f}s]", flush=True)

    # --- part 1: probability of any purchase ---
    t = time.time()
    clf = lgb.train(dict(BASE, objective="binary", metric="binary_logloss"),
                    lgb.Dataset(x, label=(y > 0).astype(np.int8)), num_boost_round=6000,
                    valid_sets=[lgb.Dataset(xva, label=(yva > 0).astype(np.int8))],
                    callbacks=[lgb.early_stopping(200, verbose=False)])
    p_buy = clf.predict(xva, num_iteration=clf.best_iteration)
    print(f"classifier  iter={clf.best_iteration}  [{time.time()-t:.0f}s]", flush=True)

    # --- part 2: size given a purchase ---
    t = time.time()
    nz = y > 0
    nz_va = yva > 0
    reg = lgb.train(dict(BASE, objective="regression", metric="rmse"),
                    lgb.Dataset(x[nz], label=np.log1p(y[nz])), num_boost_round=6000,
                    valid_sets=[lgb.Dataset(xva[nz_va],
                                            label=np.log1p(yva[nz_va] / (ratios[VAL_ANCHOR] / base)))],
                    callbacks=[lgb.early_stopping(200, verbose=False)])
    lp_size = reg.predict(xva, num_iteration=reg.best_iteration)
    print(f"size model  iter={reg.best_iteration}  [{time.time()-t:.0f}s]", flush=True)

    lp_two = p_buy * lp_size
    r2, g2, sq_two = best_scaled(ly, lp_two)
    print(f"two-part    {r2:.5f} @ scale {g2:.3f}")

    # blending the two formulations, in the space the metric lives in
    best = min(((w, best_scaled(ly, w * lp_direct + (1 - w) * lp_two)[0])
                for w in np.arange(0, 1.01, 0.05)), key=lambda z: z[1])
    print(f"blend       {best[1]:.5f} at w_direct={best[0]:.2f}")

    d = sq_direct - sq_two
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"\npaired direct vs two-part: {d.mean():+.6f} SE {se:.6f} t={d.mean()/se:+.2f}")
    np.save(ROOT / "out_work3" / "val_log_twopart.npy", lp_two)


if __name__ == "__main__":
    main()
