"""
Sequence model over daily user behaviour, trained on the Radeon via ROCm.

The task description hints at tokenising daily behaviour and letting a network
read it, which is what this does: a dilated temporal CNN reads the raw
(days x channels) activity strip ending at the anchor, and its embedding is
concatenated with the same engineered tabular features the GBDT sees.

It exists mainly for ensemble diversity -- it looks at the ordering of days,
which the window aggregates throw away.
"""

from __future__ import annotations

import argparse
import os
import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / os.environ.get("CASE3_WORK", "work")
# the daily tensor is just the densified raw log, so it is shared across
# feature versions rather than living in the versioned directory
DAILY = ROOT / "work" / "daily.npy"
OUT = ROOT / ("out" if os.environ.get("CASE3_WORK", "work") == "work"
               else "out_" + os.environ["CASE3_WORK"])

DATA_START = date(2025, 1, 1)
DATA_END = date(2026, 2, 13)
TEST_ANCHOR = DATA_END
VAL_ANCHOR = date(2026, 1, 14)
HORIZON = 30
MAX_TRAIN_ANCHOR_FOR_VAL = VAL_ANCHOR - timedelta(days=HORIZON)
SEASON_TEST = 1.163
SEQ_LEN = 180
PAD = 200          # must match build_daily.PAD
DROP = {"user_id", "target", "anchor_ord"}


def rmsle(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean(
        (np.log1p(np.clip(y_true, 0, None)) - np.log1p(np.clip(y_pred, 0, None))) ** 2)))


def dev() -> torch.device:
    if torch.cuda.is_available():          # ROCm exposes the Radeon as a cuda device
        return torch.device("cuda")
    return torch.device("cpu")


class Net(nn.Module):
    """Temporal CNN over the daily activity strip, plus a tabular branch.

    Receptive field is grown by halving the sequence each stage rather than by
    dilating the kernels. Dilation would be the natural choice, but on this ROCm
    build a dilated Conv1d costs 556 ms fwd+bwd against 34 ms for a plain one --
    16x slower -- which made the dilated version unusable. Downsampling reaches
    the same 180-day context and is cheaper still, since later stages run on
    shorter sequences.
    """

    def __init__(self, n_ch: int, n_tab: int, width: int = 96, tab_width: int = 512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_ch, width, 5, padding=2), nn.GroupNorm(8, width), nn.GELU())
        # 180 -> 90 -> 45 -> 22 -> 11
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(width, width, 5, padding=2), nn.GroupNorm(8, width), nn.GELU(),
                nn.Conv1d(width, width, 5, padding=2), nn.GroupNorm(8, width), nn.GELU(),
                nn.AvgPool1d(2),
            ) for _ in range(4)
        ])
        # mean over the whole window, mean over the most recent stride, and max
        self.seq_out = width * 3
        self.tab = nn.Sequential(
            nn.LayerNorm(n_tab),
            nn.Linear(n_tab, tab_width), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(tab_width, tab_width // 2), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(self.seq_out + tab_width // 2, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 128), nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, seq, tab):
        h = self.stem(seq)
        for st in self.stages:
            h = st(h)
        pooled = torch.cat([h.mean(-1), h[..., -2:].mean(-1), h.amax(-1)], dim=1)
        return self.head(torch.cat([pooled, self.tab(tab)], dim=1)).squeeze(-1)


def season_ratios(files):
    out = {}
    for a, p in files.items():
        cols = pl.read_parquet_schema(p)
        if "target" not in cols:
            continue
        d = pl.read_parquet(p, columns=["gmv_s30", "target"])
        out[a] = float(d["target"].sum() / d["gmv_s30"].sum())
    return out


def anchor_files():
    return {date.fromisoformat(p.stem.split("_")[1]): p for p in sorted(WORK.glob("anchor_*.parquet"))}


def load_tab(anchors, files, ratios, base, feats):
    """Tabular block plus the neutral-scale label, stacked over anchors."""
    n = 250_000 * len(anchors)
    x = np.empty((n, len(feats)), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    off = np.empty(n, dtype=np.int32)   # anchor day index, for slicing the daily tensor
    uix = np.empty(n, dtype=np.int32)
    i = 0
    for a in anchors:
        d = pl.read_parquet(files[a])
        k = d.height
        x[i:i + k] = d.select(feats).to_numpy()
        if "target" in d.columns:
            y[i:i + k] = np.log1p(d["target"].to_numpy() / (ratios[a] / base))
        off[i:i + k] = (a - DATA_START).days + PAD
        uix[i:i + k] = np.arange(k)
        i += k
        del d
    return x, y, off, uix


def make_batch(daily, off, uix, idx, tab, ar):
    """Gather (B, C, L) activity strips ending at each row's anchor.

    Every tensor here already lives in VRAM, so the whole gather runs on the GPU
    and the training loop never touches the CPU or the PCIe bus.
    """
    starts = off[idx] - SEQ_LEN + 1
    cols = starts[:, None] + ar[None, :]          # (B, L) day indices
    seq = daily[uix[idx][:, None], cols]          # (B, L, C) fp16
    return seq.permute(0, 2, 1).float(), tab[idx].float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["val", "final"], default="val")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--anchor-step", type=int, default=1, help="use every k-th anchor")
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()

    device = dev()
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    files = anchor_files()
    ratios = season_ratios(files)
    labelled = sorted(a for a in files if a != TEST_ANCHOR)
    if args.mode == "val":
        train_anchors = [a for a in labelled if a <= MAX_TRAIN_ANCHOR_FOR_VAL]
    else:
        train_anchors = labelled
    train_anchors = train_anchors[::args.anchor_step]
    base = float(np.mean([ratios[a] for a in labelled if a <= MAX_TRAIN_ANCHOR_FOR_VAL])) \
        if args.mode == "val" else float(np.mean([ratios[a] for a in labelled]))
    print(f"train anchors ({len(train_anchors)}), season base={base:.4f}", flush=True)

    daily_np = np.load(DAILY)
    n_ch = daily_np.shape[2]
    print(f"daily {daily_np.shape} {daily_np.nbytes/1e9:.2f} GB -> VRAM", flush=True)
    daily = torch.from_numpy(daily_np).to(device)
    del daily_np

    feats = [c for c in pl.read_parquet_schema(files[train_anchors[0]]) if c not in DROP]
    t0 = time.time()
    xtr, ytr, otr, utr = load_tab(train_anchors, files, ratios, base, feats)
    print(f"tab {xtr.shape} [{time.time()-t0:.0f}s]", flush=True)

    # Squash then standardise: raw window sums span many orders of magnitude.
    # 27 of the 226 columns carry nulls (std/max over windows too short to have
    # them). GBDTs route NaN natively, a network cannot -- and a plain mean over
    # a column holding one NaN poisons the whole column, which is what turned the
    # loss into NaN. Hence nan-aware moments and an explicit fill afterwards;
    # 0 is the column mean once standardised.
    xtr = np.sign(xtr) * np.log1p(np.abs(xtr))
    mu, sd = np.nanmean(xtr, 0), np.nanstd(xtr, 0) + 1e-6
    xtr = (xtr - mu) / sd
    np.clip(xtr, -10, 10, out=xtr)
    xtr = np.nan_to_num(xtr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    assert np.isfinite(xtr).all()

    eval_anchor = VAL_ANCHOR if args.mode == "val" else TEST_ANCHOR
    de = pl.read_parquet(files[eval_anchor])
    xev = de.select(feats).to_numpy().astype(np.float32)
    xev = np.sign(xev) * np.log1p(np.abs(xev))
    xev = np.clip((xev - mu) / sd, -10, 10)
    xev = np.nan_to_num(xev, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    assert np.isfinite(xev).all()
    oev = np.full(de.height, (eval_anchor - DATA_START).days + PAD, dtype=np.int32)
    uev = np.arange(de.height, dtype=np.int32)
    yev = de["target"].to_numpy().astype(np.float64) if "target" in de.columns else None
    sev = (ratios[eval_anchor] / base) if args.mode == "val" else (SEASON_TEST / base)
    print(f"eval anchor {eval_anchor}, scale={sev:.4f}", flush=True)

    # move the whole training set into VRAM (fp16 for the tabular block); at
    # ~3.6 GB + ~1.5 GB this fits the 17 GB card with room to spare
    ar = torch.arange(SEQ_LEN, device=device)
    tab_tr = torch.from_numpy(xtr).half().to(device)
    off_tr = torch.from_numpy(otr).to(device)
    uix_tr = torch.from_numpy(utr).to(device)
    y_tr = torch.from_numpy(ytr).to(device)
    tab_ev = torch.from_numpy(xev).half().to(device)
    off_ev = torch.from_numpy(oev).to(device)
    uix_ev = torch.from_numpy(uev).to(device)
    if device.type == "cuda":
        print(f"VRAM allocated {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    n = len(ytr)
    acc = np.zeros(de.height)
    # in "final" mode there is no label to select on, so use the epoch count the
    # validation run settled on rather than always running to the end
    best_ep_path = OUT / "nn_best_epoch.json"
    target_ep = None
    if args.mode == "final" and best_ep_path.exists():
        target_ep = json.load(open(best_ep_path))["best_epoch"]
        print(f"final mode: stopping at epoch {target_ep} (from validation)", flush=True)
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        best_score, best_pe, best_ep = float("inf"), None, -1
        model = Net(n_ch, len(feats)).to(device)
        nparam = sum(p.numel() for p in model.parameters())
        print(f"\nseed {seed}: {nparam/1e6:.2f}M params", flush=True)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        steps = args.epochs * (n // args.batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                    pct_start=0.15)

        step = 0
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(n, device=device)
            run, cnt, t1 = 0.0, 0, time.time()
            for bi in range(n // args.batch):
                idx = perm[bi * args.batch:(bi + 1) * args.batch]
                seq, tab = make_batch(daily, off_tr, uix_tr, idx, tab_tr, ar)
                tgt = y_tr[idx]
                pred = model(seq, tab)
                loss = F.mse_loss(pred, tgt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                sched.step()
                lv = loss.item()
                if not np.isfinite(lv):
                    raise RuntimeError(f"loss became {lv} at epoch {ep} step {cnt}")
                run += lv; cnt += 1; step += 1
                if cnt % 200 == 0:
                    print(f"  ep{ep} {cnt}/{n//args.batch} loss={run/cnt:.4f} "
                          f"[{time.time()-t1:.0f}s]", flush=True)
            # evaluate
            model.eval()
            pe = np.empty(de.height, dtype=np.float64)
            with torch.no_grad():
                for i in range(0, de.height, 8192):
                    idx = torch.arange(i, min(i + 8192, de.height), device=device)
                    seq, tab = make_batch(daily, off_ev, uix_ev, idx, tab_ev, ar)
                    pe[i:i + len(idx)] = model(seq, tab).float().cpu().numpy()
            msg = f"ep{ep} train_mse={run/max(cnt,1):.4f}"
            if yev is not None:
                lvl = np.clip(np.expm1(pe), 0, None) * sev
                sc = rmsle(yev, lvl)
                msg += f" val_RMSLE={sc:.5f}"
                if sc < best_score:
                    best_score, best_pe, best_ep = sc, pe.copy(), ep
                    msg += "  *best*"
            elif target_ep is not None and ep == target_ep:
                best_pe, best_ep = pe.copy(), ep
            print(msg + f" [{time.time()-t1:.0f}s]", flush=True)
        acc += best_pe if best_pe is not None else pe
        if yev is not None:
            print(f"seed {seed}: best epoch {best_ep} at {best_score:.5f}", flush=True)
            json.dump({"best_epoch": best_ep, "rmsle": best_score}, open(best_ep_path, "w"))

    pe = acc / args.seeds
    OUT.mkdir(exist_ok=True, parents=True)
    tag = "val" if args.mode == "val" else "test"
    np.save(OUT / f"nn_log_{tag}.npy", pe)
    if yev is not None:
        lvl = np.clip(np.expm1(pe), 0, None) * sev
        print(f"\nFINAL nn {tag} RMSLE={rmsle(yev, lvl):.5f}")
    else:
        lvl = np.clip(np.expm1(pe), 0, None) * sev
        pl.DataFrame({"user_id": de["user_id"], "predict": lvl}).write_csv(OUT / "nn_submission.csv")
        print(f"\nwrote {OUT/'nn_submission.csv'} sum={lvl.sum():,.0f}")


if __name__ == "__main__":
    main()
