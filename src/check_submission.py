"""
Sanity-check a submission before it is trusted.

A file that loads but is subtly wrong -- missing users, NaNs, a scale that drifted
by an order of magnitude -- costs a whole submission slot, so check the things
that would actually invalidate it rather than eyeballing the head of the file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "submission.csv"
    sub = pl.read_csv(path)
    ref = pl.read_csv(ROOT / "sample_submit.csv")
    # work9 is the live feature set; the original work/ holds only daily.npy now
    prev = pl.read_parquet(ROOT / os.environ.get("CASE3_WORK", "work9") / "anchor_2026-02-13.parquet",
                           columns=["user_id", "gmv_s30"])

    fails, warns = [], []

    if sub.columns != ["user_id", "predict"]:
        fails.append(f"columns are {sub.columns}, expected ['user_id', 'predict']")
    if sub.height != 250_000:
        fails.append(f"{sub.height} rows, expected 250000")

    ids = sub["user_id"].to_numpy()
    if len(np.unique(ids)) != len(ids):
        fails.append("duplicate user_id values")
    if not np.array_equal(np.sort(ids), np.sort(ref["user_id"].to_numpy())):
        fails.append("user_id set does not match sample_submit.csv")

    p = sub["predict"].to_numpy().astype(np.float64)
    if not np.isfinite(p).all():
        fails.append(f"{(~np.isfinite(p)).sum()} non-finite predictions")
    if (p < 0).any():
        fails.append(f"{(p < 0).sum()} negative predictions (they get zeroed by the scorer)")

    # An RMSLE-optimal model deliberately sums to far less than the truth: L2 on
    # log1p targets the conditional geometric mean, not the arithmetic one. On
    # the holdout the fitted model summed to 0.435x the realised total, and
    # rescaling it up to match that total made RMSLE *worse* (1.673 -> 1.846).
    # So the total is checked against what the validated model does, not against
    # the realised GMV. Neutral-scale reference: predicted/carry = 0.462.
    carry = float(prev["gmv_s30"].sum())
    ratio = p.sum() / carry
    scale = 1.0
    w = os.environ.get("CASE3_WORK", "work9")
    meta = ROOT / ("out" if w == "work" else "out_" + w) / "final_meta.json"
    if meta.exists():
        import json
        scale = json.load(open(meta))["scale"]
    expected = 0.462 * scale
    if not 0.5 * expected <= ratio <= 1.7 * expected:
        fails.append(f"total predicted GMV is {ratio:.3f}x the previous 30 days, "
                     f"expected about {expected:.3f}x -- the seasonal scale or the "
                     f"target transform is likely wrong")
    elif not 0.75 * expected <= ratio <= 1.35 * expected:
        warns.append(f"total predicted GMV is {ratio:.3f}x the previous 30 days, "
                     f"expected about {expected:.3f}x")

    zero_frac = float((p < 1e-6).mean())
    if zero_frac > 0.30:
        warns.append(f"{zero_frac:.1%} of predictions are exactly zero -- unusual for a "
                     "regression on log1p, which normally predicts small positives")

    print(f"file          : {path}")
    print(f"rows          : {sub.height}")
    print(f"predict       : mean={p.mean():.3f} median={np.median(p):.3f} "
          f"max={p.max():,.1f} zeros={zero_frac:.4f}")
    print(f"total GMV     : {p.sum():,.0f}  ({ratio:.3f}x previous 30 days = {carry:,.0f})")
    for w in warns:
        print(f"WARN  {w}")
    for f in fails:
        print(f"FAIL  {f}")
    print("VERDICT: " + ("PASS" if not fails else "INVALID"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
