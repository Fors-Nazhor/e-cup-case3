"""
Assemble a submission from stored per-model predictions.

Written after an audit found that the file on the leaderboard could not be
regenerated from the repository -- the weights in blend_report.json, the ones in
the README and the ones actually in the CSV all disagreed, because it had been
built by an ad-hoc inline computation. Everything here is explicit and a
manifest is written next to the CSV.

Two things this gets right that the earlier ad-hoc blending did not:

1. Weights and calibration are fitted TOGETHER. Pinning the scale first does not
   select the best models, it selects whichever model's own optimum happens to
   sit nearest that constant -- which produced a degenerate all-weight-on-one
   blend when the scale was pinned at 1.0.

2. The holdout scale is not the test scale. The holdout window (Jan 15 - Feb 13)
   and the target window (Feb 14 - Mar 15) sit in different parts of the year.
   The one real measurement available is the earlier pair of submissions: an
   ensemble whose holdout optimum was 0.7049 turned out to be optimal at ~1.0 on
   the public board. That RATIO transfers; the constant does not.

Model sources are given as `kind` or `kind:val_tag`, because variants of the same
model (e.g. the two-part model with and without the per-anchor incidence offset)
are otherwise easy to mix up between the holdout and test halves -- which
happened, and produced weights fitted on a model that was not the one predicting.
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
VAL_ANCHOR = date(2026, 1, 14)

# holdout optimum 0.7049 -> test optimum ~1.0, measured over two submissions
TRANSFER = 1.419

LO, HI = np.log(0.45), np.log(1.60)


def mse_at(ly, p0, g):
    return float(np.mean((ly - np.log1p(p0 * g)) ** 2))


def best_scale(ly, lp, tol=1e-4):
    """Optimal calibration constant for one prediction vector.

    MSE is smooth and unimodal in log(scale), so a golden-section search finds
    the optimum in ~12 evaluations where a grid needed 80. With ~1800 weight
    combinations to score that is the difference between two minutes and twenty.
    """
    p0 = np.clip(np.expm1(lp), 0, None)
    invphi = (np.sqrt(5) - 1) / 2
    a, b = LO, HI
    c, d = b - invphi * (b - a), a + invphi * (b - a)
    fc, fd = mse_at(ly, p0, np.exp(c)), mse_at(ly, p0, np.exp(d))
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = mse_at(ly, p0, np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = mse_at(ly, p0, np.exp(d))
    g = float(np.exp((a + b) / 2))
    return g, mse_at(ly, p0, g)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="feature-set dir, e.g. work9")
    ap.add_argument("--models", default="lgb,cat,two,nn",
                    help="comma list; use kind:val_tag to pick a variant")
    ap.add_argument("--nn-from", default="",
                    help="take the net's predictions from another out_ dir")
    ap.add_argument("--shrink", type=float, default=0.0,
                    help="pull fitted weights this far toward equal (0..1). The "
                         "residual matrix is ill-conditioned (cond ~3000), so a "
                         "fitted corner solution such as w_lgb=0 is unstable "
                         "across resamples rather than evidence about the model")
    ap.add_argument("--scale", type=float, default=0.0, help="0 = derive via TRANSFER")
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = ROOT / f"out_{args.work}"
    yva = np.load(out_dir / f"val_y__{VAL_ANCHOR}.npy")
    ly = np.log1p(yva)

    val, test, used, own_scale = {}, {}, {}, {}
    for spec in args.models.split(","):
        # kind[:val_tag[:test_tag]] -- the two halves are tagged separately
        # because a holdout run and a final run of the same variant are done at
        # different times and do not always carry the same suffix
        parts = spec.split(":")
        kind = parts[0]
        vtag = parts[1] if len(parts) > 1 else ""
        ttag = parts[2] if len(parts) > 2 else ""
        d = ROOT / f"out_{args.nn_from}" if (kind == "nn" and args.nn_from) else out_dir
        vp = d / (f"nn_log_val{vtag}.npy" if kind == "nn"
                  else f"val_log_{kind}{vtag}__{VAL_ANCHOR}.npy")
        tp = d / (f"nn_log_test{ttag}.npy" if kind == "nn" else f"test_log_{kind}{ttag}.npy")
        if not vp.exists() or not tp.exists():
            print(f"SKIP {spec}: missing {vp.name if not vp.exists() else tp.name}")
            continue
        v, t = np.load(vp), np.load(tp)
        assert len(v) == len(ly), f"{vp.name}: {len(v)} rows vs {len(ly)} holdout"
        val[kind], test[kind] = v, t
        used[kind] = {"val": vp.name, "test": tp.name, "dir": d.name}
        g, m = best_scale(ly, v)
        own_scale[kind] = g
        print(f"{kind:4} holdout {m ** 0.5:.5f} at its own scale {g:.3f}   "
              f"[{vp.name} / {tp.name}]")

    assert val, "no models found"

    # A model whose own optimum sits far from its peers is the signature of a
    # holdout/test mismatch -- it is being scored against a different variant.
    gs = np.array(list(own_scale.values()))
    if gs.max() / gs.min() > 1.12:
        print(f"\nWARNING: own-optimum scales span {gs.min():.3f}..{gs.max():.3f}. "
              f"That usually means one model's holdout and test predictions come "
              f"from different variants. Check the file names above.")

    ks = sorted(val)
    grid = np.arange(0, 1 + 1e-9, args.step)
    best = None
    for w in itertools.product(grid, repeat=len(ks)):
        if abs(sum(w) - 1) > 1e-9:
            continue
        mix = sum(wi * val[k] for wi, k in zip(w, ks))
        g, m = best_scale(ly, mix)
        if best is None or m < best[2]:
            best = (w, g, m)
    w_fit = np.array(best[0], dtype=float)
    if args.shrink > 0:
        w_fit = (1 - args.shrink) * w_fit + args.shrink / len(ks)
        # the scale has to be refitted for the new mix, not carried over
        mix = sum(wi * val[k] for wi, k in zip(w_fit, ks))
        g, m = best_scale(ly, mix)
        best = (tuple(w_fit), g, m)
        print(f"\nshrunk {args.shrink:.0%} toward equal")
    weights = dict(zip(ks, (round(float(x), 3) for x in w_fit)))
    g_hold = best[1]
    print(f"\nweights {weights}")
    print(f"holdout RMSLE {best[2] ** 0.5:.5f} at holdout scale {g_hold:.3f}")

    scale = args.scale if args.scale > 0 else g_hold * TRANSFER
    print(f"test scale {scale:.3f}" +
          ("" if args.scale > 0 else f" = {g_hold:.3f} x {TRANSFER} (transferred)"))

    lp = sum(w_fit[i] * test[k] for i, k in enumerate(ks))
    pred = np.clip(np.expm1(lp), 0, None) * scale
    uid = np.load(out_dir / "test_user_id.npy")
    assert len(uid) == len(pred) == 250_000, (len(uid), len(pred))

    path = ROOT / args.out
    pl.DataFrame({"user_id": uid, "predict": pred}).write_csv(path)
    json.dump({"work": args.work, "models": args.models, "weights": weights,
               "holdout_scale": g_hold, "transfer": TRANSFER, "test_scale": scale,
               "holdout_rmsle": best[2] ** 0.5, "own_scales": own_scale,
               "val_anchor": str(VAL_ANCHOR), "sources": used,
               "n_rows": int(len(uid)), "pred_sum": float(pred.sum())},
              open(path.with_suffix(".json"), "w"), indent=2)
    print(f"\nwrote {path.name} (sum {pred.sum():,.0f}) + {path.with_suffix('.json').name}")


if __name__ == "__main__":
    main()
