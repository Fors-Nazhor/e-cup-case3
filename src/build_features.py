"""
E-CUP 2026, task 3 -- Search user LTV.

Builds anchor-based user features from the raw daily activity log.

For an anchor date t, features describe user behaviour over windows ending at t
(inclusive), and the target is total GMV over [t+1, t+30].
The test anchor is 2026-02-13, so the target window is [2026-02-14, 2026-03-15].
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

DATA = Path(__file__).resolve().parent.parent
RAW = DATA / "train.parquet"
WORK = DATA / os.environ.get("CASE3_WORK", "work")

HORIZON = 30
DATA_START = date(2025, 1, 1)
DATA_END = date(2026, 2, 13)
TEST_ANCHOR = DATA_END

# windows are [t - (w-1), t], i.e. w days ending on the anchor
CORE_WINDOWS = [7, 14, 30, 60, 90, 180, 365]
CHAN_WINDOWS = [30, 90, 365]
STAT_WINDOWS = [30, 90, 365]

CORE_COLS = ["gmv", "to_ord", "to_cart", "searches", "is_ord_day", "is_active"]
CHAN_COLS = [
    "gmv_search", "gmv_cat",
    "search_to_ord", "cat_to_ord",
    "search_to_cart", "cat_to_cart",
    "search", "cat",
    "has_search_to_ord", "has_cat_to_ord",
    "has_search_to_cart", "has_cat_to_cart",
]
STAT_COLS = ["gmv", "searches", "to_ord", "to_cart"]

# (feature_name_suffix, boolean condition column) -> days since that event last happened
RECENCY = {
    "any": pl.lit(True),
    "ord": pl.col("to_ord") > 0,
    "cart": pl.col("to_cart") > 0,
    "gmv": pl.col("gmv") > 0,
    "search": pl.col("search") > 0,
    "cat": pl.col("cat") > 0,
    "sord": pl.col("search_to_ord") > 0,
    "cord": pl.col("cat_to_ord") > 0,
}

NEVER = 9999  # sentinel for "event never observed in the lookback"


def load_raw() -> pl.DataFrame:
    """Load the activity log, downcast, and add per-row helper indicators."""
    df = pl.read_parquet(RAW)
    df = df.with_columns(
        [pl.col(c).cast(pl.Int32) for c in df.columns if df.schema[c] == pl.Int64 and c != "user_id"]
        + [pl.col("user_id").cast(pl.Int32)]
        + [pl.col(c).cast(pl.Float32) for c in ("gmv", "gmv_search", "gmv_cat")]
    )
    df = df.with_columns(
        is_ord_day=(pl.col("to_ord") > 0).cast(pl.Int32),
        is_active=pl.lit(1, dtype=pl.Int32),
    )
    # within-group row order drives the gap features, so pin it down once
    return df.sort(["user_id", "event_date"])


def _window_exprs() -> list[pl.Expr]:
    """Conditional aggregations over every (column, window) pair."""
    out: list[pl.Expr] = []
    for w in CORE_WINDOWS:
        m = pl.col("age") < w
        for c in CORE_COLS:
            out.append(pl.when(m).then(pl.col(c)).otherwise(0).sum().alias(f"{c}_s{w}"))
    for w in CHAN_WINDOWS:
        m = pl.col("age") < w
        for c in CHAN_COLS:
            out.append(pl.when(m).then(pl.col(c)).otherwise(0).sum().alias(f"{c}_s{w}"))
    for w in STAT_WINDOWS:
        m = pl.col("age") < w
        for c in STAT_COLS:
            out.append(pl.when(m).then(pl.col(c)).otherwise(None).max().alias(f"{c}_mx{w}"))
            out.append(pl.when(m).then(pl.col(c)).otherwise(None).std().alias(f"{c}_sd{w}"))
    for name, cond in RECENCY.items():
        out.append(pl.when(cond).then(pl.col("age")).otherwise(None).min().alias(f"rec_{name}"))
    # spacing between active days over the last 90 days
    gap = pl.col("age").filter(pl.col("age") < 90)
    out.append(gap.diff().abs().mean().alias("gap_mean90"))
    out.append(gap.diff().abs().std().alias("gap_std90"))

    # --- purchase-timing and purchase-size detail (BTYD-flavoured) ---
    # The window aggregates say how much and how often, but not how regularly a
    # user buys nor how large their most recent baskets were. Residuals across
    # model families correlate at 0.995+, so the ceiling is information rather
    # than modelling; this is the information that was missing.
    oage = pl.col("age").filter(pl.col("to_ord") > 0)      # ages of ordering days
    out.append(oage.diff().abs().mean().alias("ord_gap_mean"))
    out.append(oage.diff().abs().std().alias("ord_gap_std"))
    out.append(oage.diff().abs().max().alias("ord_gap_max"))
    # how overdue the user is: recency measured in their own typical spacing
    out.append((oage.min() / (oage.diff().abs().mean() + 1e-6)).alias("ord_overdue"))
    # 2nd and 3rd most recent order, for a sense of momentum beyond recency
    out.append(oage.sort().slice(1, 1).first().alias("rec_ord2"))
    out.append(oage.sort().slice(2, 1).first().alias("rec_ord3"))

    ogmv = pl.col("gmv").filter(pl.col("gmv") > 0)
    oa = pl.col("age").filter(pl.col("gmv") > 0)
    out.append(ogmv.median().alias("gmv_pos_med"))
    out.append(ogmv.quantile(0.9).alias("gmv_pos_p90"))
    out.append(ogmv.sort_by(oa).first().alias("gmv_last"))          # most recent buying day
    out.append(ogmv.sort_by(oa).head(3).mean().alias("gmv_last3"))
    out.append(ogmv.len().alias("n_gmv_days"))

    # weekend share of activity: some users only shop at weekends
    wknd = (pl.col("dow") >= 5)
    out.append((pl.when(wknd).then(1).otherwise(0).sum() / pl.len()).alias("wknd_share"))
    out.append((pl.when(wknd & (pl.col("to_ord") > 0)).then(1).otherwise(0).sum()
                / (pl.when(pl.col("to_ord") > 0).then(1).otherwise(0).sum() + 1e-6))
               .alias("wknd_ord_share"))

    # breadth of engagement: distinct active weeks / months in the last year
    out.append((pl.col("age") // 7).n_unique().alias("active_weeks365"))
    out.append((pl.col("age") // 30).n_unique().alias("active_months365"))
    out.append(((pl.col("age") // 7).filter(pl.col("age") < 90)).n_unique().alias("active_weeks90"))
    return out


def _safe_div(a: str, b: str, name: str) -> pl.Expr:
    return (pl.col(a) / (pl.col(b) + 1e-6)).alias(name)


def _derived_exprs(anchor: date) -> list[pl.Expr]:
    """Ratios, rates and trends built on top of the raw window aggregates."""
    e: list[pl.Expr] = []

    # activity rate and per-active-day intensity
    for w in CORE_WINDOWS:
        e.append((pl.col(f"is_active_s{w}") / w).alias(f"act_rate{w}"))
        e.append((pl.col(f"is_ord_day_s{w}") / w).alias(f"ord_rate{w}"))
        e.append(_safe_div(f"gmv_s{w}", f"is_active_s{w}", f"gmv_per_day{w}"))
        e.append(_safe_div(f"gmv_s{w}", f"to_ord_s{w}", f"aov{w}"))
        e.append(_safe_div(f"to_ord_s{w}", f"to_cart_s{w}", f"cart2ord{w}"))
        e.append(_safe_div(f"to_cart_s{w}", f"searches_s{w}", f"srch2cart{w}"))
        e.append(_safe_div(f"searches_s{w}", f"is_active_s{w}", f"srch_per_day{w}"))

    # momentum: recent window vs. longer window, normalised per day
    for short, long in [(7, 30), (14, 60), (30, 90), (30, 180), (60, 180), (90, 365)]:
        for c in ("gmv", "to_ord", "searches", "to_cart", "is_active"):
            e.append(
                ((pl.col(f"{c}_s{short}") / short) / (pl.col(f"{c}_s{long}") / long + 1e-6))
                .alias(f"trend_{c}_{short}_{long}")
            )

    # channel mix
    for w in CHAN_WINDOWS:
        e.append(_safe_div(f"gmv_search_s{w}", f"gmv_s{w}", f"share_gmv_search{w}"))
        e.append(_safe_div(f"search_to_ord_s{w}", f"to_ord_s{w}", f"share_ord_search{w}"))
        e.append(_safe_div(f"search_to_cart_s{w}", f"to_cart_s{w}", f"share_cart_search{w}"))
        e.append(_safe_div(f"search_s{w}", f"is_active_s{w}", f"share_days_search{w}"))
        e.append(_safe_div(f"cat_s{w}", f"is_active_s{w}", f"share_days_cat{w}"))
        e.append(_safe_div(f"gmv_search_s{w}", f"search_to_ord_s{w}", f"aov_search{w}"))
        e.append(_safe_div(f"gmv_cat_s{w}", f"cat_to_ord_s{w}", f"aov_cat{w}"))

    # lifetime context and how much history actually exists at this anchor
    hist = (anchor - DATA_START).days + 1
    e.append(pl.lit(hist, dtype=pl.Int32).alias("hist_days"))
    e.append(
        (pl.lit(hist) - (pl.col("first_date") - pl.lit(DATA_START)).dt.total_days())
        .cast(pl.Float32)
        .alias("tenure")
    )
    e.append(_safe_div("life_gmv", "life_days", "life_gmv_per_day"))
    e.append(_safe_div("life_gmv", "life_ord", "life_aov"))
    e.append(_safe_div("life_ord", "life_cart", "life_cart2ord"))
    # is the user's most recent basket unusually large or small for them?
    e.append(_safe_div("gmv_last", "gmv_pos_med", "gmv_last_vs_med"))
    e.append(_safe_div("gmv_last3", "gmv_pos_med", "gmv_last3_vs_med"))
    # spelled out from the raw sums: aov30/aov365 are created in this same
    # with_columns batch and so are not yet visible to each other
    e.append((((pl.col("gmv_s30") / (pl.col("to_ord_s30") + 1e-6))
               / ((pl.col("gmv_s365") / (pl.col("to_ord_s365") + 1e-6)) + 1e-6))
              .alias("aov_trend_30_365")))
    e.append(_safe_div("gmv_s90", "life_gmv", "recent_gmv_share90"))
    e.append(_safe_div("gmv_s365", "life_gmv", "recent_gmv_share365"))
    return e


def build_anchor(
    df: pl.DataFrame,
    users: pl.DataFrame,
    anchor: date,
    with_target: bool,
) -> pl.DataFrame:
    """Feature matrix (one row per user) for a single anchor date."""
    lo = anchor - timedelta(days=364)
    hist = df.filter(pl.col("event_date").is_between(lo, anchor)).with_columns(
        age=(pl.lit(anchor) - pl.col("event_date")).dt.total_days().cast(pl.Int32),
        dow=pl.col("event_date").dt.weekday().cast(pl.Int8) - 1,   # 0 = Monday
    )

    feats = hist.group_by("user_id").agg(_window_exprs())

    # lifetime stats must only look at data up to the anchor, otherwise the
    # target window would leak straight into the features
    g = (
        df.filter(pl.col("event_date") <= anchor)
        .group_by("user_id")
        .agg(
            first_date=pl.col("event_date").min(),
            life_days=pl.len().cast(pl.Int32),
            life_gmv=pl.col("gmv").sum(),
            life_ord=pl.col("to_ord").sum(),
            life_cart=pl.col("to_cart").sum(),
            life_searches=pl.col("searches").sum(),
            life_ord_days=pl.col("is_ord_day").sum(),
            life_gmv_max=pl.col("gmv").max(),
        )
    )

    out = users.join(feats, on="user_id", how="left").join(g, on="user_id", how="left")

    # users with no history at all: zero counts, "never" recency, epoch-less tenure
    zero_cols = [c for c in out.columns if c.endswith(tuple(f"_s{w}" for w in CORE_WINDOWS + CHAN_WINDOWS))]
    out = out.with_columns(
        [pl.col(c).fill_null(0) for c in zero_cols]
        + [pl.col(f"rec_{k}").fill_null(NEVER) for k in RECENCY]
        + [
            pl.col("life_days").fill_null(0),
            pl.col("life_gmv").fill_null(0.0),
            pl.col("life_ord").fill_null(0),
            pl.col("life_cart").fill_null(0),
            pl.col("life_searches").fill_null(0),
            pl.col("life_ord_days").fill_null(0),
            pl.col("first_date").fill_null(anchor),
        ]
    )

    out = out.with_columns(_derived_exprs(anchor))
    out = out.drop("first_date")

    if with_target:
        t0, t1 = anchor + timedelta(days=1), anchor + timedelta(days=HORIZON)
        tgt = (
            df.filter(pl.col("event_date").is_between(t0, t1))
            .group_by("user_id")
            .agg(target=pl.col("gmv").sum().cast(pl.Float64))
        )
        out = out.join(tgt, on="user_id", how="left").with_columns(
            pl.col("target").fill_null(0.0)
        )

    out = out.with_columns(anchor_ord=pl.lit((anchor - DATA_START).days, dtype=pl.Int32))
    # keep everything compact; GBDTs are happy with float32
    out = out.with_columns(
        [
            pl.col(c).cast(pl.Float32)
            for c in out.columns
            if c not in ("user_id", "target") and out.schema[c] != pl.Float32
        ]
    )
    return out


def anchors_for(n_folds: int, stride: int, last: date) -> list[date]:
    return sorted(last - timedelta(days=i * stride) for i in range(n_folds))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=14)
    ap.add_argument("--n-folds", type=int, default=15)
    # extra anchor series shifted back by these many days; they multiply the
    # training data and cover anchor weekdays other than the base series'
    ap.add_argument("--offsets", default="0")
    args = ap.parse_args()
    offsets = [int(o) for o in args.offsets.split(",")]

    WORK.mkdir(exist_ok=True, parents=True)

    t0 = time.time()
    df = load_raw()
    print(f"raw loaded {df.shape} in {time.time()-t0:.1f}s", flush=True)

    users = df.select(pl.col("user_id").unique().sort())
    print(f"{users.height} users", flush=True)

    last_train_anchor = DATA_END - timedelta(days=HORIZON)  # 2026-01-14
    train_anchors = sorted(
        {a
         for off in offsets
         for a in anchors_for(args.n_folds, args.stride, last_train_anchor - timedelta(days=off))}
    )
    print(f"train anchors ({len(train_anchors)}):", [str(a) for a in train_anchors], flush=True)
    print("test anchor:", TEST_ANCHOR, flush=True)


    for a in train_anchors:
        fp = WORK / f"anchor_{a}.parquet"
        if fp.exists():
            print(f"skip {a} (exists)", flush=True)
            continue
        s = time.time()
        out = build_anchor(df, users, a, with_target=True)
        out.write_parquet(fp, compression="zstd")
        print(
            f"{a}: {out.shape} target mean={out['target'].mean():.3f} "
            f"nz={(out['target']>0).mean():.4f} [{time.time()-s:.1f}s]",
            flush=True,
        )

    fp = WORK / f"anchor_{TEST_ANCHOR}.parquet"
    if not fp.exists():
        s = time.time()
        out = build_anchor(df, users, TEST_ANCHOR, with_target=False)
        out.write_parquet(fp, compression="zstd")
        print(f"TEST {TEST_ANCHOR}: {out.shape} [{time.time()-s:.1f}s]", flush=True)

    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
