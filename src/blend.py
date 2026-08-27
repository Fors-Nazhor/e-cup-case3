"""
Blend the model families and write the submission.

Weights are fitted in log1p space on the holdout anchor (2026-01-14), because
that is where the metric lives, then applied to the stored test predictions.
Everything is combined as a weighted average of log1p predictions; the seasonal
scale for the test window is applied once at the end.
"""

from __future__ import annotations

import argparse
import os
import itertools
import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / ("out" if os.environ.get("CASE3_WORK", "work") == "work"
               else "out_" + os.environ["CASE3_WORK"])
SEASON_TEST = 1.163


def rmsle(y, p) -> float:
    return float(np.sqrt(np.mean(
        (np.log1p(np.clip(y, 0, None)) - np.log1p(np.clip(p, 0, None))) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=0.0, help="0 = read from val_report.json")
    ap.add_argument("--step", type=float, default=0.05)
    args = ap.parse_args()

    rep = json.load(open(OUT / "val_report.json"))
    sval = rep["val_ratio"] / rep["season_base"]
    # the final models were fitted over a different anchor set, so their neutral
    # scale comes from final_meta.json rather than the validation report
    fm = OUT / "final_meta.json"
    stest = args.scale or (json.load(open(fm))["scale"] if fm.exists()
                           else SEASON_TEST / rep["season_base"])

    y = np.load(OUT / "val_y.npy")
    val, test = {}, {}
    for p in sorted(OUT.glob("val_log_*.npy")):
        k = p.stem.replace("val_log_", "")
        tp = OUT / f"test_log_{k}.npy"
        if not tp.exists():
            print(f"skip {k}: no test predictions")
            continue
        val[k], test[k] = np.load(p), np.load(tp)
    if (OUT / "nn_log_val.npy").exists() and (OUT / "nn_log_test.npy").exists():
        val["nn"], test["nn"] = np.load(OUT / "nn_log_val.npy"), np.load(OUT / "nn_log_test.npy")

    keys = sorted(val)
    print(f"models: {keys}")
    for k in keys:
        print(f"  {k}: val RMSLE = {rmsle(y, np.clip(np.expm1(val[k]),0,None)*sval):.5f}")

    grid = np.arange(0, 1 + 1e-9, args.step)
    best = (None, 1e9)
    for w in itertools.product(grid, repeat=len(keys)):
        s = sum(w)
        if abs(s - 1) > 1e-9:
            continue
        mix = sum(wi * val[k] for wi, k in zip(w, keys))
        sc = rmsle(y, np.clip(np.expm1(mix), 0, None) * sval)
        if sc < best[1]:
            best = (w, sc)
    w = best[0]
    print(f"\nbest weights: {dict(zip(keys, [round(x,3) for x in w]))} -> val RMSLE {best[1]:.5f}")

    mix_test = sum(wi * test[k] for wi, k in zip(w, keys))
    pred = np.clip(np.expm1(mix_test), 0, None) * stest
    uid = np.load(OUT / "test_user_id.npy")
    sub = pl.DataFrame({"user_id": uid, "predict": pred})
    sub.write_csv(ROOT / "submission.csv")
    print(f"\nwrote {ROOT/'submission.csv'} rows={sub.height} scale={stest:.4f}")
    print(f"pred: mean={pred.mean():.3f} zeros={(pred<1e-6).mean():.4f} sum={pred.sum():,.0f}")
    json.dump({"weights": dict(zip(keys, map(float, w))), "val_rmsle": best[1],
               "test_scale": stest}, open(OUT / "blend_report.json", "w"), indent=2)


if __name__ == "__main__":
    main()
