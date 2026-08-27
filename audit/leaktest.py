import sys, time
from datetime import date, timedelta
sys.path.insert(0,"src")
import polars as pl, numpy as np
import build_features as bf

t0=time.time()
df = bf.load_raw()
users = df.select(pl.col("user_id").unique().sort())
print("loaded", df.shape, f"{time.time()-t0:.0f}s", flush=True)

for anchor in [date(2026,1,14), date(2025,10,8)]:
    full = bf.build_anchor(df, users, anchor, with_target=True)
    trunc_df = df.filter(pl.col("event_date") <= anchor)
    tr = bf.build_anchor(trunc_df, users, anchor, with_target=False)
    cols = [c for c in tr.columns if c not in ("user_id",)]
    bad=[]
    for c in cols:
        a = full[c].to_numpy().astype(np.float64)
        b = tr[c].to_numpy().astype(np.float64)
        m = ~(np.isnan(a)&np.isnan(b))
        if not np.allclose(a[m], b[m], rtol=1e-5, atol=1e-5, equal_nan=True):
            d = np.nanmax(np.abs(a[m]-b[m]))
            bad.append((c, float(d), int(np.sum(~np.isclose(a[m],b[m],rtol=1e-5,atol=1e-5)))))
    print(f"\nANCHOR {anchor}: compared {len(cols)} cols, MISMATCH={len(bad)}")
    for c,d,n in bad[:20]: print(f"   {c}: maxdiff={d:.6g} nrows={n}")
    # also: does hist_days == anchor_ord+1 ?
    h=full["hist_days"].unique().to_list(); ao=full["anchor_ord"].unique().to_list()
    print(f"   hist_days={h} anchor_ord={ao}")
print(f"\ntotal {time.time()-t0:.0f}s")
