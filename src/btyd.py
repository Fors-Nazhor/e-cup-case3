"""
BG/NBD + Gamma-Gamma features, per anchor.

The task description suggests BTYD methods and I had skipped them. They matter
here for a specific reason: every model in the ensemble so far is a tree or a
net reading the same engineered columns, and their residuals correlate at
0.996-0.999, so the blend gains almost nothing. BG/NBD has a genuinely different
inductive bias -- a parametric story about purchase timing rather than
thresholds on aggregates -- which is what an over-correlated ensemble needs.

The model (Fader, Hardie & Lee 2005): each customer buys at a Poisson rate
lambda ~ Gamma(r, alpha), and after each purchase drops out with probability
p ~ Beta(a, b). Fitting the four population parameters by maximum likelihood
gives, per customer, P(still active) and the expected number of purchases in
the next 30 days in closed form. Gamma-Gamma then models spend per purchase,
assumed independent of frequency.

Written directly rather than via `lifetimes`, which is unmaintained since 2020
and does not survive current numpy/pandas.
"""

from __future__ import annotations

import argparse
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, hyp2f1

ROOT = Path(__file__).resolve().parent.parent
DATA_START = date(2025, 1, 1)
HORIZON = 30
WEEK = 7.0          # fit in weeks; days make alpha tiny and the optimiser unhappy


def rfm_at(df: pl.DataFrame, users: pl.DataFrame, anchor: date) -> pl.DataFrame:
    """frequency / recency / T per user, counting days with an order as purchases."""
    d = df.filter((pl.col("event_date") <= anchor) & (pl.col("to_ord") > 0))
    g = d.group_by("user_id").agg(
        n_tx=pl.len().cast(pl.Float64),
        first_tx=pl.col("event_date").min(),
        last_tx=pl.col("event_date").max(),
        gmv_sum=pl.col("gmv").sum().cast(pl.Float64),
    )
    out = users.join(g, on="user_id", how="left")
    return out.with_columns(
        # x: repeat purchases; t_x: age at last purchase; T: age at the anchor
        x=(pl.col("n_tx").fill_null(0.0) - 1).clip(0.0),
        t_x=((pl.col("last_tx") - pl.col("first_tx")).dt.total_days() / WEEK).fill_null(0.0),
        T=((pl.lit(anchor) - pl.col("first_tx")).dt.total_days() / WEEK).fill_null(0.0),
        mval=(pl.col("gmv_sum") / pl.col("n_tx")).fill_null(0.0),
    ).select(["user_id", "x", "t_x", "T", "n_tx", "mval"])


def _nll(params, x, t_x, T):
    """Negative log-likelihood, Fader-Hardie-Lee (2005) eq. 3.

    L = A1 * A2 * (A3 + 1{x>0} * A4), so the two survival branches are combined
    with a log-sum-exp and the customer-independent factors added outside it.
    """
    r, alpha, a, b = np.exp(params)          # positivity by construction
    ln_a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    ln_a2 = betaln(a, b + x) - betaln(a, b)
    ln_a3 = -(r + x) * np.log(alpha + T)
    has_repeat = x > 0
    ln_a4 = np.where(has_repeat,
                     np.log(a) - np.log(np.where(has_repeat, b + x - 1.0, 1.0))
                     - (r + x) * np.log(alpha + t_x),
                     -np.inf)
    branch = np.where(has_repeat, np.logaddexp(ln_a3, ln_a4), ln_a3)
    ll = ln_a1 + ln_a2 + branch
    if not np.all(np.isfinite(ll)):
        return 1e12
    return -float(np.sum(ll))


def fit_bgnbd(x, t_x, T, seed=(1.0, 1.0, 1.0, 1.0)):
    res = minimize(_nll, np.log(seed), args=(x, t_x, T), method="Nelder-Mead",
                   options=dict(maxiter=4000, xatol=1e-4, fatol=1e-4))
    return np.exp(res.x), res.fun


def bgnbd_predict(params, x, t_x, T, horizon_weeks):
    """P(alive) and expected purchases over the next `horizon_weeks`."""
    r, alpha, a, b = params
    # P(alive | x, t_x, T)
    denom = 1.0 + np.where(x > 0, (a / (b + x - 1.0)) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0.0)
    p_alive = 1.0 / denom
    # E[Y(t) | x, t_x, T]
    z = horizon_weeks / (alpha + T + horizon_weeks)
    hyp = hyp2f1(r + x, b + x, a + b + x - 1.0, z)
    first = ((a + b + x - 1.0) / (a - 1.0)) * (1.0 - ((alpha + T) / (alpha + T + horizon_weeks)) ** (r + x) * hyp)
    exp_tx = first / denom
    return p_alive, np.clip(exp_tx, 0, None)


def fit_gamma_gamma(mval, n_tx):
    """Spend per purchase: m_bar ~ Gamma, itself Gamma-distributed across users."""
    m = (n_tx > 0) & (mval > 0)
    mb, nx = mval[m], n_tx[m]

    def nll(par):
        p, q, v = np.exp(par)
        return -np.sum(gammaln(p * nx + q) - gammaln(p * nx) - gammaln(q)
                       + q * np.log(v) + (p * nx - 1) * np.log(mb)
                       + (p * nx) * np.log(nx) - (p * nx + q) * np.log(v + nx * mb))

    res = minimize(nll, np.log([1.0, 1.0, np.mean(mb)]), method="Nelder-Mead",
                   options=dict(maxiter=3000))
    return np.exp(res.x)


def gg_predict(par, mval, n_tx):
    p, q, v = par
    return np.where(n_tx > 0, (p * v + p * n_tx * mval) / (p * n_tx + q - 1.0), 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="work3")
    ap.add_argument("--dst", default="work5")
    args = ap.parse_args()

    src, dst = ROOT / args.src, ROOT / args.dst
    dst.mkdir(exist_ok=True, parents=True)

    df = pl.read_parquet(ROOT / "train.parquet",
                         columns=["event_date", "user_id", "to_ord", "gmv"])
    anchors = sorted(date.fromisoformat(p.stem.split("_")[1]) for p in src.glob("anchor_*.parquet"))
    print(f"{len(anchors)} anchors", flush=True)

    params = gg = None
    for a in anchors:
        out_fp = dst / f"anchor_{a}.parquet"
        if out_fp.exists():
            continue
        base = pl.read_parquet(src / f"anchor_{a}.parquet")
        users = base.select("user_id")
        rfm = rfm_at(df, users, a)
        x = rfm["x"].to_numpy()
        t_x = rfm["t_x"].to_numpy()
        T = rfm["T"].to_numpy()

        # BG/NBD is undefined for customers with no first purchase, so the fit
        # uses only those who bought at least once; the rest still get scored
        # and simply fall back to the population prior
        buyers = rfm["n_tx"].to_numpy() > 0
        if params is None:
            params, nll = fit_bgnbd(x[buyers], t_x[buyers], T[buyers])
            gg = fit_gamma_gamma(rfm["mval"].to_numpy(), rfm["n_tx"].to_numpy())
            print(f"BG/NBD r={params[0]:.4f} alpha={params[1]:.4f} "
                  f"a={params[2]:.4f} b={params[3]:.4f}  nll={nll:,.0f}", flush=True)
            print(f"Gamma-Gamma p={gg[0]:.4f} q={gg[1]:.4f} v={gg[2]:.4f}", flush=True)

        p_alive, exp_tx = bgnbd_predict(params, x, t_x, T, HORIZON / WEEK)
        p_alive = np.where(buyers, p_alive, 0.0)
        exp_tx = np.where(buyers, exp_tx, 0.0)
        exp_m = gg_predict(gg, rfm["mval"].to_numpy(), rfm["n_tx"].to_numpy())

        base = base.with_columns(
            btyd_p_alive=pl.Series(p_alive).cast(pl.Float32),
            btyd_exp_tx=pl.Series(exp_tx).cast(pl.Float32),
            btyd_exp_m=pl.Series(exp_m).cast(pl.Float32),
            btyd_clv=pl.Series(exp_tx * exp_m).cast(pl.Float32),
            btyd_x=pl.Series(x).cast(pl.Float32),
            btyd_T=pl.Series(T).cast(pl.Float32),
            btyd_recency=pl.Series(T - t_x).cast(pl.Float32),
        )
        base.write_parquet(out_fp, compression="zstd")
        print(f"{a}: p_alive mean={p_alive.mean():.4f} exp_tx mean={exp_tx.mean():.4f} "
              f"clv mean={(exp_tx*exp_m).mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
