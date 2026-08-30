"""
A distinct estimator of E[log1p(y)]: multiclass over binned log1p, not L2.

Our four models disagree with each other almost nowhere -- residual correlations
run 0.9968 to 0.9989 on a clean anchor -- so blending them buys almost nothing.
They vary the algorithm (boosting, convolutions) but not the way the quantity is
estimated: all of them fit a conditional mean with squared loss.

This one goes the other way round. Discretise log1p(y) into K bins, fit a
multiclass model, and read the target back off the predicted distribution as
sum_k p_k * centre_k. It estimates the same quantity through the full conditional
distribution instead of its first moment, and it handles the atom at zero as a
class of its own rather than as a value the regressor has to average through.

What matters here is not whether it beats the others outright -- it probably will
not -- but whether it errs differently. A model at 0.98 residual correlation
would be the first real diversity in the ensemble.
"""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta

import lightgbm as lgb
import numpy as np
import polars as pl

import train as T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-anchor", required=True)
    ap.add_argument("--predict-test", type=int, default=0,
                    help="also write predictions for the test anchor")
    ap.add_argument("--bins", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=120)
    ap.add_argument("--anchor-step", type=int, default=3)
    ap.add_argument("--tag", default="_binned")
    args = ap.parse_args()

    va_date = date.fromisoformat(args.val_anchor)
    files = T.anchor_files()
    ratios = T.season_ratios(files)
    train_anchors = sorted(a for a in files if a <= va_date - timedelta(days=T.HORIZON))
    train_anchors = sorted(train_anchors[::-1][::args.anchor_step])
    base = float(np.mean([ratios[a] for a in train_anchors]))
    print(f"train anchors ({len(train_anchors)}): {train_anchors[0]} .. {train_anchors[-1]}", flush=True)

    x, y, _, _, feats = T.load_stack(train_anchors, files, ratios, base, deseason=1)
    ly = np.log1p(np.clip(y, 0, None))

    # Bin edges from the positive part only; zero gets class 0 to itself, which
    # is the point -- roughly half the mass sits exactly there.
    pos = ly[ly > 0]
    edges = np.quantile(pos, np.linspace(0, 1, args.bins)[1:-1])
    cls = np.digitize(ly, edges) * (ly > 0)
    centres = np.array([ly[cls == k].mean() if (cls == k).any() else 0.0
                        for k in range(args.bins)])
    print(f"{args.bins} classes, zero class holds {(cls == 0).mean():.1%} of rows")

    s = time.time()
    m = lgb.train(
        dict(objective="multiclass", num_class=args.bins, learning_rate=0.1,
             num_leaves=63, min_data_in_leaf=100, feature_fraction=0.3,
             bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0,
             num_threads=16, verbosity=-1),
        lgb.Dataset(x, cls), num_boost_round=args.rounds)
    print(f"fitted in {time.time() - s:.0f}s", flush=True)

    va = pl.read_parquet(files[va_date])
    xva = va.select(feats).to_numpy().astype(np.float32)
    yva = va["target"].to_numpy().astype(np.float64)
    pred = m.predict(xva) @ centres            # E[log1p(y) | x]

    out = T.OUT / f"val_log_binned{args.tag}__{va_date}.npy"
    np.save(out, pred)
    lyv = np.log1p(yva)
    g = np.exp(np.polyfit([0], [0], 0)[0]) if False else 1.0
    # calibrate the level the same way every other model here is calibrated
    from scipy.optimize import minimize_scalar
    f = lambda u: float(np.mean((lyv - np.log1p(np.clip(np.expm1(pred), 0, None) * np.exp(u))) ** 2))
    u = minimize_scalar(f, bounds=(-1, 1), method="bounded", options={"xatol": 1e-9}).x
    print(f"binned: RMSLE={f(u) ** 0.5:.5f} (scale={np.exp(u):.3f})  -> {out.name}")

    if args.predict_test:
        te = pl.read_parquet(files[T.TEST_ANCHOR])
        xte = te.select(feats).to_numpy().astype(np.float32)
        pt = m.predict(xte) @ centres
        tout = T.OUT / f"test_log_binned{args.tag}.npy"
        np.save(tout, pt)
        print(f"test predictions -> {tout.name}  (mean log1p {pt.mean():.4f})")


if __name__ == "__main__":
    main()
