"""
Can deliberately different models decorrelate the ensemble?

Diagnosis so far: every model sits at ~1.669 and their residuals correlate at
0.996-0.999, so each 0.001 gained on a component turns into ~0.0006 on the
blend. Polishing components is therefore near-useless; what the blend needs is
members that fail on *different* users.

Two sources of difference are tried here, both cheap:
  - feature families -- fit on disjoint slices of the feature set
  - loss functions -- Huber and quantile regression weight users differently
    than L2 does, so they misfit a different subset

Each is individually worse than the tuned L2 model by construction. The question
is whether the blend beats the tuned model anyway.
"""

from __future__ import annotations

import itertools
import os
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

BASE = dict(num_threads=16, verbosity=-1, num_leaves=255, min_data_in_leaf=20,
            feature_fraction=0.25, bagging_fraction=0.7, bagging_freq=1,
            lambda_l2=5.0, learning_rate=0.02, max_bin=255, path_smooth=1.0,
            seed=42, bagging_seed=43, feature_fraction_seed=44)

G = np.exp(np.linspace(np.log(0.45), np.log(1.35), 60))


def family(name: str) -> callable:
    """Disjoint-ish slices of the feature set."""
    if name == "recency":
        return lambda c: c.startswith(("rec_", "ord_gap", "ord_overdue", "gap_",
                                       "active_weeks", "active_months", "wknd_"))
    if name == "volume":
        return lambda c: ("_s7" in c or "_s14" in c or "_s30" in c or "_s60" in c
                          or "_s90" in c or "_s180" in c or "_s365" in c
                          or c.startswith("life_") or "_r180" in c or "_r365" in c)
    if name == "rates":
        return lambda c: c.startswith(("act_rate", "ord_rate", "aov", "cart2ord",
                                       "srch2cart", "srch_per_day", "gmv_per_day",
                                       "trend_", "share_", "recent_gmv"))
    raise ValueError(name)


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
    sval = ratios[VAL_ANCHOR] / base
    print(f"train {x.shape}", flush=True)

    def score(lp):
        p0 = np.clip(np.expm1(lp), 0, None)
        ms = [np.mean((ly - np.log1p(p0 * v)) ** 2) for v in G]
        j = int(np.argmin(ms))
        return (ly - np.log1p(p0 * G[j])) ** 2

    def fit(params, cols=None, label=None, tag=""):
        idx = [i for i, c in enumerate(feats) if cols is None or cols(c)]
        xt, xv = (x, xva) if cols is None else (x[:, idx], xva[:, idx])
        lab = np.log1p(y) if label is None else label(y)
        vlab = np.log1p(yva / sval) if label is None else label(yva / sval)
        m = lgb.train(dict(BASE, **params), lgb.Dataset(xt, label=lab),
                      num_boost_round=4000,
                      valid_sets=[lgb.Dataset(xv, label=vlab)],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        lp = m.predict(xv, num_iteration=m.best_iteration)
        sq = score(lp)
        print(f"  {tag:26} {np.sqrt(sq.mean()):.5f}  ({len(idx)} признаков, iter={m.best_iteration})", flush=True)
        return lp, sq

    out = {}
    print("\nбазовая модель:")
    out["l2"] = fit(dict(objective="regression", metric="rmse"), tag="L2, все признаки")

    print("\nсемейства признаков:")
    for f in ("recency", "volume", "rates"):
        out[f] = fit(dict(objective="regression", metric="rmse"), cols=family(f), tag=f"L2, только {f}")

    print("\nдругие функции потерь:")
    out["huber"] = fit(dict(objective="huber", alpha=2.0, metric="rmse"), tag="Huber")
    out["q60"] = fit(dict(objective="quantile", alpha=0.6, metric="quantile"), tag="квантиль 0.6")

    print("\nкорреляция остатков с базовой моделью:")
    r0 = ly - np.log1p(np.clip(np.expm1(out["l2"][0]), 0, None) * sval)
    for k, (lp, _) in out.items():
        if k == "l2":
            continue
        r = ly - np.log1p(np.clip(np.expm1(lp), 0, None) * sval)
        print(f"  l2-{k:8} {np.corrcoef(r0, r)[0, 1]:.4f}")

    ks = list(out)
    best = (None, 9.0)
    for w in itertools.product(np.arange(0, 1.001, 0.1), repeat=len(ks)):
        if abs(sum(w) - 1) > 1e-9:
            continue
        s = np.sqrt(score(sum(wi * out[k][0] for wi, k in zip(w, ks))).mean())
        if s < best[1]:
            best = (w, s)
    print(f"\nлучшая смесь: {dict(zip(ks, [round(v,2) for v in best[0]]))} -> {best[1]:.5f}")
    print(f"базовая одна: {np.sqrt(out['l2'][1].mean()):.5f}   выигрыш {np.sqrt(out['l2'][1].mean())-best[1]:+.5f}")
    for k, (lp, _) in out.items():
        np.save(ROOT / f"out_{WORK.name}" / f"val_log_div_{k}.npy", lp)


if __name__ == "__main__":
    main()
