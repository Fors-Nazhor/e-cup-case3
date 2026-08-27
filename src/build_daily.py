"""
Densify the sparse activity log into one daily tensor.

Shape is (n_users, n_days, n_channels) over the whole 2025-01-01..2026-02-13
timeline, stored as float16 of log1p(value). At ~2.9 GB it stays in RAM, and any
anchor is just a slice: the window ending on anchor `a` is
    daily[:, off(a) - L + 1 : off(a) + 1, :]
so one array serves every anchor instead of materialising a sequence dataset per
anchor.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"   # shared across feature versions
DATA_START = date(2025, 1, 1)
DATA_END = date(2026, 2, 13)
N_DAYS = (DATA_END - DATA_START).days + 1
# zero-pad the front so that a 180-day window ending on the earliest anchor does
# not index before position 0; day d of the timeline lives at column PAD + d
PAD = 200

# log1p-scaled count/value channels, then two raw binary flags
VAL_CH = [
    "gmv", "gmv_search", "gmv_cat",
    "to_ord", "to_cart", "searches",
    "search_to_ord", "cat_to_ord",
    "search_to_cart", "cat_to_cart",
]
BIN_CH = ["search", "cat"]
CHANNELS = VAL_CH + BIN_CH


def main() -> None:
    t0 = time.time()
    df = pl.read_parquet(ROOT / "train.parquet")
    users = df["user_id"].unique().sort()
    n_users = users.len()
    print(f"{n_users} users x {N_DAYS} days x {len(CHANNELS)} ch", flush=True)

    # users is sorted, so searchsorted gives the row index straight away and
    # matches the user ordering used by build_features
    uarr = users.to_numpy()
    row = np.searchsorted(uarr, df["user_id"].to_numpy()).astype(np.int32)
    days = df["event_date"].to_numpy().astype("datetime64[D]")
    col = (days - np.datetime64(DATA_START)).astype(np.int32) + PAD

    out = np.zeros((n_users, N_DAYS + PAD, len(CHANNELS)), dtype=np.float16)
    for k, c in enumerate(CHANNELS):
        v = df[c].to_numpy().astype(np.float32)
        if c in VAL_CH:
            v = np.log1p(np.clip(v, 0, None))
        out[row, col, k] = v.astype(np.float16)
        print(f"  {c} done [{time.time()-t0:.0f}s]", flush=True)

    WORK.mkdir(exist_ok=True, parents=True)
    np.save(WORK / "daily.npy", out)
    np.save(WORK / "daily_users.npy", users.to_numpy().astype(np.int64))
    print(f"saved {out.shape} {out.nbytes/1e9:.2f} GB in {time.time()-t0:.0f}s")
    nz = float((out[:, :, 0] > 0).mean())
    print(f"sanity: frac days with gmv>0 = {nz:.4f}")


if __name__ == "__main__":
    main()
