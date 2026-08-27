"""Where does the sequence model's step time actually go on this ROCm build?"""
import os, time, sys
import torch, torch.nn as nn, torch.nn.functional as F

d = torch.device("cuda")
B, L, C, W = 4096, 180, 12, 96

def timed(name, fn, iters=12, warmup=4):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    dt = (time.time() - t) / iters
    print(f"  {name:<34} {dt*1000:8.1f} ms/step")
    return dt

seq = torch.randn(B, C, L, device=d)
tab = torch.randn(B, 226, device=d)
daily = torch.randn(250000, 609, C, device=d, dtype=torch.float16)
uix = torch.randint(0, 250000, (B,), device=d)
off = torch.full((B,), 400, device=d)
ar = torch.arange(L, device=d)

print(f"MIOPEN_FIND_MODE={os.environ.get('MIOPEN_FIND_MODE','<unset>')}")
print("component timings:")

def gather():
    cols = (off - L + 1)[:, None] + ar[None, :]
    return daily[uix[:, None], cols].permute(0, 2, 1).float()
timed("VRAM gather (B,C,L)", gather)

conv = nn.Conv1d(W, W, 3, padding=1).to(d)
inp  = torch.randn(B, W, L, device=d)
timed("one Conv1d fwd", lambda: conv(inp))

def conv_fwd_bwd():
    o = conv(inp); o.sum().backward()
timed("one Conv1d fwd+bwd", conv_fwd_bwd)

lin = nn.Linear(W, W).to(d)
inp2 = torch.randn(B, L, W, device=d)
timed("one Linear fwd (same shape)", lambda: lin(inp2))

gn = nn.GroupNorm(8, W).to(d)
timed("one GroupNorm fwd", lambda: gn(inp))
