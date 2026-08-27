"""
Does what a user did in the same calendar window a year ago add signal?

The appeal is obvious: the target window covers 23 February and 8 March, and a
user who bought gifts then last year plausibly will again. The catch is that the
feature is only computable for anchors whose target window starts after
2026-01-01, which excludes every training anchor -- so it cannot be validated
the usual way.

This tests the prior question instead, using only the holdout anchor: split its
users in half, fit on one half with and without the year-ago columns, score on
the other. It cannot tell us how much the feature is worth in the real training
setup, but it can tell us whether the signal exists at all -- and if it does not,
there is nothing to engineer around.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work3"
VAL_ANCHOR = dt.date(2026, 1, 14)
DROP = {"user_id", "target", "anchor_ord"}

PARAMS = dict(objective="regression", metric="rmse", num_threads=16, verbosity=-1,
              num_leaves=127, min_data_in_leaf=100, feature_fraction=0.3,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              learning_rate=0.03, seed=42)


def per_user(raw, users, a: str, b: str, col="gmv"):
    w = (raw.filter(pl.col("event_date").is_between(dt.date.fromisoformat(a), dt.date.fromisoformat(b)))
            .group_by("user_id").agg(pl.col(col).sum().alias("v")))
    return users.join(w, on="user_id", how="left").with_columns(pl.col("v").fill_null(0.0))["v"].to_numpy()


def main() -> None:
    va = pl.read_parquet(WORK / f"anchor_{VAL_ANCHOR}.parquet")
    feats = [c for c in va.columns if c not in DROP]
    users = va.select("user_id")
    x = va.select(feats).to_numpy().astype(np.float32)
    y = va["target"].to_numpy().astype(np.float64)

    raw = pl.read_parquet(ROOT / "train.parquet", columns=["event_date", "user_id", "gmv", "to_ord"])

    # the holdout's target window is 2026-01-15..2026-02-13; its year-ago twin
    # is 2025-01-15..2025-02-13, which the data does cover
    extra = {
        "ya_gmv":      per_user(raw, users, "2025-01-15", "2025-02-13", "gmv"),
        "ya_ord":      per_user(raw, users, "2025-01-15", "2025-02-13", "to_ord"),
        "ya_gmv_pre":  per_user(raw, users, "2024-12-16", "2025-01-14", "gmv"),   # empty, kept for symmetry
        "ya_gmv_wide": per_user(raw, users, "2025-01-01", "2025-02-28", "gmv"),
    }
    # ratio of the year-ago window to that user's whole-2025 activity: expresses
    # "this user is unusually active in this part of the year"
    tot25 = per_user(raw, users, "2025-01-01", "2025-12-31", "gmv")
    extra["ya_share"] = extra["ya_gmv"] / (tot25 + 1e-6)
    extra["ya_share_wide"] = extra["ya_gmv_wide"] / (tot25 + 1e-6)

    names = [k for k in extra if extra[k].any()]
    print(f"year-ago columns with any signal: {names}")
    xe = np.column_stack([x] + [extra[k].astype(np.float32) for k in names])

    rng = np.random.default_rng(0)
    half = rng.random(len(y)) < 0.5
    ly = np.log1p(y)

    def run(mat, tag, show_importance=False):
        d = lgb.train(PARAMS, lgb.Dataset(mat[half], label=np.log1p(y[half])),
                      num_boost_round=4000,
                      valid_sets=[lgb.Dataset(mat[~half], label=np.log1p(y[~half]))],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        lp = d.predict(mat[~half], num_iteration=d.best_iteration)
        p0 = np.clip(np.expm1(lp), 0, None)
        g = np.exp(np.linspace(np.log(0.5), np.log(1.5), 60))
        sq = [(ly[~half] - np.log1p(p0 * v)) ** 2 for v in g]
        j = int(np.argmin([s.mean() for s in sq]))
        print(f"  {tag:22} {np.sqrt(sq[j].mean()):.5f}  iter={d.best_iteration}")
        if show_importance:
            imp = sorted(zip(feats + names, d.feature_importance("gain")), key=lambda z: -z[1])
            rank = {n: i for i, (n, _) in enumerate(imp)}
            print("  ранг новых признаков по gain:",
                  {n: rank[n] for n in names}, f"из {len(imp)}")
        return sq[j]

    print("\nfit on half the holdout users, score on the other half:")
    a = run(x, "without year-ago")
    b = run(xe, "with year-ago", show_importance=True)
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"\npaired: {d.mean():+.6f}  SE {se:.6f}  t = {d.mean()/se:+.2f}")
    print("вывод:", "сигнал есть" if d.mean() / se > 2 else "сигнала не видно")


if __name__ == "__main__":
    main()
