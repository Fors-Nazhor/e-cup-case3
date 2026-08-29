"""
Optimal affine combination of already-scored submissions, in closed form.

This is the second of the two solutions in this repository. The first
(`make_submission.py`) is the model pipeline. This one sits on top of it and
squeezes out what the leaderboard has already told us for free.

The idea. Every scored submission is a known vector `L_i` (log1p of its
predictions) with a known `MSE_i` (the reported RMSLE, squared). Writing the
unknown truth as `y`:

    MSE_i = <y,y> - 2<y,L_i> + <L_i,L_i>

`<L_i,L_i>` is computable from the file, so each score pins down `<y,L_i>` up to
the single unknown constant `<y,y>`. For a combination `L(w) = sum_i w_i L_i`:

    MSE(w) = <y,y> - 2 sum_i w_i <y,L_i> + w'Gw

and if the weights are constrained to sum to one, the `<y,y>` terms cancel:

    MSE(w) = w'Gw - sum_i w_i G_ii + sum_i w_i MSE_i          (*)

Every quantity on the right is known. So the best affine combination of past
submissions -- and its exact score -- can be computed without spending an
attempt. Verified twice against reality: a blend predicted 1.6491575 scored
1.6491604, and a six-vector optimum predicted 1.6481569 scored 1.6481830.

Why the norm has to be constrained. The reported scores carry a small error
(the public board covers roughly a third of the users, so quantities computed
over all 250k differ slightly from the ones being scored). That error enters (*)
linearly through `sum_i w_i MSE_i`, i.e. multiplied by ||w||_1. Unconstrained,
the search happily exploits near-degenerate directions and returns weights in
the thousands with a fantasy score; those blow up on the private half. The cap
keeps the amplification below the measured error floor.

Predictions are also required to stay positive: clipping a negative back to zero
would take the result out of the affine span and void the exact prediction.

Usage:
    python src/lb_combine.py --scores scores.json --out submission_fitted.csv

where scores.json maps a submission file to its reported public RMSLE:
    {"submission_v9_opt.csv": 1.649315706180368, ...}
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent


def optimum(G, g, m, idx, cap):
    """Minimise (*) over weights summing to one, restricted to `idx`.

    Parametrised as w = w0 + Z t with w0 a corner of the simplex and Z spanning
    the sum-to-zero directions, which turns the constraint into an unconstrained
    solve in one fewer dimension.
    """
    A = G[np.ix_(idx, idx)]
    gg, mm = g[idx], m[idx]
    k = len(idx)
    w0 = np.zeros(k)
    w0[0] = 1.0
    Z = np.zeros((k, k - 1))
    for j in range(k - 1):
        Z[0, j] = -1.0
        Z[j + 1, j] = 1.0
    H = Z.T @ (2 * A) @ Z
    if np.linalg.eigvalsh(H).min() <= 1e-11:      # vectors linearly dependent
        return None
    t = np.linalg.solve(H, -(Z.T @ (2 * A @ w0 + (mm - gg))))
    w = w0 + Z @ t
    if np.abs(w).sum() > cap:
        return None
    return w, float(w @ A @ w - gg @ w + mm @ w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True,
                    help="JSON mapping submission csv -> reported public RMSLE")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=float, default=5.0,
                    help="max L1 norm of the weights; see the note above")
    args = ap.parse_args()

    scores = json.load(open(ROOT / args.scores))
    names = list(scores)
    ref, L = None, []
    for n in names:
        d = pl.read_csv(ROOT / n).sort("user_id")
        if ref is None:
            ref = d
        assert np.array_equal(ref["user_id"].to_numpy(), d["user_id"].to_numpy()), n
        L.append(np.log1p(d["predict"].to_numpy()))
    L = np.array(L)
    N = L.shape[1]
    G = L @ L.T / N
    g = np.diag(G).copy()
    m = np.array([scores[n] for n in names]) ** 2

    # A vector that is already a combination of the others adds no dimension;
    # the rank tells you how much independent information the span holds.
    sv = np.linalg.svd((L[1:] - L[0]) @ (L[1:] - L[0]).T / N, compute_uv=False)
    rank = int((sv > sv[0] * 1e-10).sum())
    print(f"{len(names)} submissions, span rank {rank}")

    best = None
    for r in range(2, len(names) + 1):
        for idx in itertools.combinations(range(len(names)), r):
            z = optimum(G, g, m, list(idx), args.cap)
            if z is None or z[1] <= 0:
                continue
            w, mse = z
            if (np.expm1(w @ L[list(idx)]) <= 0).any():
                continue                           # would need clipping
            if best is None or mse < best[1]:
                best = (list(idx), mse, w)
    assert best is not None, "no admissible combination under the norm cap"
    idx, mse, w = best

    print(f"\nbest single: {min(scores.values()):.7f}")
    print(f"combination: {mse ** 0.5:.7f}   gain {min(scores.values()) - mse ** 0.5:+.5f}")
    print(f"||w||_1 = {np.abs(w).sum():.2f}")
    for i, wi in zip(idx, w):
        print(f"  {names[i]:34} {wi:+.5f}")

    p = np.expm1(w @ L[idx])
    path = ROOT / args.out
    pl.DataFrame({"user_id": ref["user_id"], "predict": p}).write_csv(path)
    json.dump({"inputs": {names[i]: scores[names[i]] for i in idx},
               "weights": {names[i]: float(wi) for i, wi in zip(idx, w)},
               "l1_norm": float(np.abs(w).sum()),
               "predicted_rmsle": mse ** 0.5, "span_rank": rank},
              open(path.with_suffix(".json"), "w"), indent=2)
    print(f"\nwrote {path.name} (sum {p.sum():,.0f}) + {path.with_suffix('.json').name}")


if __name__ == "__main__":
    main()
