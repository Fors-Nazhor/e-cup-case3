"""Which op families actually work on this ROCm build? MIOpen JIT is broken here."""
import torch, torch.nn as nn, time

d = torch.device("cuda")
def probe(name, fn):
    try:
        out = fn()
        torch.cuda.synchronize()
        print(f"  OK    {name} -> {tuple(out.shape)}")
        return True
    except Exception as e:
        print(f"  FAIL  {name}: {type(e).__name__}: {str(e).splitlines()[0][:110]}")
        return False

x3 = torch.randn(256, 12, 180, device=d)
x2 = torch.randn(256, 226, device=d)
print("op probe:")
probe("Linear",        lambda: nn.Linear(226, 512).to(d)(x2))
probe("LayerNorm",     lambda: nn.LayerNorm(226).to(d)(x2))
probe("GroupNorm",     lambda: nn.GroupNorm(4, 12).to(d)(x3))
probe("BatchNorm1d",   lambda: nn.BatchNorm1d(12).to(d)(x3))
probe("Conv1d",        lambda: nn.Conv1d(12, 96, 3, padding=1).to(d)(x3))
probe("MultiheadAttn", lambda: nn.MultiheadAttention(128, 4, batch_first=True).to(d)(
        torch.randn(256, 30, 128, device=d), torch.randn(256, 30, 128, device=d),
        torch.randn(256, 30, 128, device=d))[0])
probe("sdpa",          lambda: torch.nn.functional.scaled_dot_product_attention(
        *[torch.randn(256, 4, 30, 32, device=d)]*3))
