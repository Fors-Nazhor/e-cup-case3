"""Steady-state step time of the real Net, and whether MIOpen's tuning persists."""
import time, sys, torch, torch.nn as nn
sys.path.insert(0, "src")
from nn_train import Net

d = torch.device("cuda")
B = 4096
model = Net(12, 226).to(d)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
seq = torch.randn(B, 12, 180, device=d)
tab = torch.randn(B, 226, device=d)
tgt = torch.randn(B, device=d)

def step():
    loss = torch.nn.functional.mse_loss(model(seq, tab), tgt)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

t = time.time(); step(); torch.cuda.synchronize()
print(f"first step (includes any kernel tuning): {time.time()-t:.1f} s")
for _ in range(3): step()
torch.cuda.synchronize()
t = time.time()
for _ in range(10): step()
torch.cuda.synchronize()
dt = (time.time()-t)/10
print(f"steady state: {dt*1000:.0f} ms/step")
print(f"-> {793*dt/60:.1f} min/epoch at 793 steps, {6*793*dt/60:.0f} min for 6 epochs")
