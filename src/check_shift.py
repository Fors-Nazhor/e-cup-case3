"""Flag a model whose level moves differently from its peers between the
holdout anchor and the test anchor.

Written after submission v10 scored 1.65167 against v9's 1.64932 while being
0.00083 BETTER on the holdout. Decomposing the regression through
    MSE_new - MSE_base = <D,D> - 2<D,e>
gave <D,e> = -0.000025, i.e. the change carried exactly no information about
the true errors and the whole loss was the self-cost of moving predictions.

The cause is visible without the leaderboard. Every established model drops its
mean log prediction by about 0.14 going from the holdout anchor to the test
anchor, 30 days further out. The net trained on the top 150 features by gain
dropped only 0.056, because the pruning had removed the long-window and
history columns (life_days, active_months365, the 365-day sums) that tell a
model where it sits on the timeline. It could not adapt, and predicted high.

So: compare each model's holdout->test shift against the median of the others.
A model that stands apart is not necessarily wrong, but it is making a claim
the rest of the ensemble disagrees with, and that claim should be justified
before it is worth a submission attempt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# name -> (holdout predictions, test predictions), both log1p scale
DEFAULT = {
    "lgb":       ("val_log_lgb__2026-01-14.npy",         "test_log_lgb.npy"),
    "cat":       ("val_log_cat__2026-01-14.npy",         "test_log_cat.npy"),
    "two":       ("val_log_two_noincid__2026-01-14.npy", "test_log_two.npy"),
    "nn_work3":  ("nn_log_val.npy",                      "nn_log_test.npy"),
    "nn_top150": ("nn_log_val_top150.npy",               "nn_log_test_top150.npy"),
}

# v10 sat 0.089 from the consensus and cost 0.0024 on the board
TOL = 0.04


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work9")
    ap.add_argument("--tol", type=float, default=TOL,
                    help="how far from the peer median counts as suspect")
    args = ap.parse_args()
    out = ROOT / f"out_{args.work}"

    rows = []
    for name, (v, t) in DEFAULT.items():
        vp, tp = out / v, out / t
        if not vp.exists() or not tp.exists():
            print(f"skip {name}: missing {vp.name if not vp.exists() else tp.name}")
            continue
        a, b = np.load(vp), np.load(tp)
        rows.append((name, a.mean(), b.mean(), b.mean() - a.mean(), a.std(), b.std()))

    assert len(rows) >= 3, "need at least three models to have a consensus"
    shifts = np.array([r[3] for r in rows])

    print(f"{'model':12}{'val mean':>10}{'test mean':>11}{'shift':>9}"
          f"{'vs peers':>10}{'val sd':>9}{'test sd':>9}")
    flagged = []
    for i, (name, vm, tm, sh, vs, ts) in enumerate(rows):
        # median of the OTHERS, so a suspect model cannot vote for itself
        peer = float(np.median(np.delete(shifts, i)))
        d = sh - peer
        mark = "  <-- SUSPECT" if abs(d) > args.tol else ""
        if mark:
            flagged.append(name)
        print(f"{name:12}{vm:10.4f}{tm:11.4f}{sh:+9.4f}{d:+10.4f}{vs:9.4f}{ts:9.4f}{mark}")

    print()
    if flagged:
        print(f"SUSPECT: {', '.join(flagged)} -- level moves unlike the rest of the "
              f"ensemble between the two anchors. Do not spend a submission on a "
              f"blend these dominate without explaining why they disagree.")
    else:
        print("all models move together between the anchors")


if __name__ == "__main__":
    main()
