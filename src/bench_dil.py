"""Is dilation the pathological case on MIOpen/gfx1201?"""
import time, torch, torch.nn as nn
d = torch.device("cuda"); B, L, W = 4096, 180, 96
x = torch.randn(B, W, L, device=d)
for dil in (1, 2, 4, 8, 16, 32):
    c = nn.Conv1d(W, W, 3, padding=dil, dilation=dil).to(d)
    def fb():
        o = c(x); o.sum().backward()
    for _ in range(2): fb()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(5): fb()
    torch.cuda.synchronize()
    print(f"  dilation={dil:<3} {(time.time()-t)/5*1000:8.1f} ms  (fwd+bwd)")
print()
# alternative: same receptive field via pooling instead of dilation
for k in (3, 5):
    c = nn.Conv1d(W, W, k, padding=k//2).to(d)
    def fb():
        o = c(x); o.sum().backward()
    for _ in range(2): fb()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(5): fb()
    torch.cuda.synchronize()
    print(f"  plain k={k}     {(time.time()-t)/5*1000:8.1f} ms  (fwd+bwd)")
