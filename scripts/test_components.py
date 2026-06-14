#!/usr/bin/env python3
"""
MSA-NeOpt component verification script
Run:     python test_components.py
Requires: pip install torch numpy cvxpy cvxopt
"""

import sys, warnings
warnings.filterwarnings("ignore")
import torch, torch.nn as nn, numpy as np

print("=" * 60)
print("MSA-NeOpt — component verification")
print(f"Python  {sys.version.split()[0]}")
print(f"PyTorch {torch.__version__}")
print(f"NumPy   {np.__version__}")
print("=" * 60)

results = []

def check(name, fn):
    try:
        detail = fn()
        results.append(("PASS", name, detail))
        print(f"  ✓  {name}")
        print(f"       {detail}\n")
    except Exception as e:
        results.append(("FAIL", name, str(e)))
        print(f"  ✗  {name}")
        print(f"       ERROR: {e}\n")

# shared greedy solver
def greedy_peak(y):
    BAT_CAP, BAT_POWER = 0.5, 0.25
    sols = []
    for i in range(len(y)):
        load = y[i].detach().numpy() if hasattr(y[i], "detach") else np.array(y[i])
        soc = 0.0; disp = np.zeros(len(load)); thr = np.percentile(load, 85)
        for t, p in enumerate(load):
            if p > thr and soc > 0:
                d = min(BAT_POWER, soc, p - thr); disp[t] = -d; soc -= d
            elif p < thr * 0.6 and soc < BAT_CAP:
                c = min(BAT_POWER, BAT_CAP - soc); disp[t] = c; soc += c
        sols.append((load + disp).max())
    return torch.tensor(sols, dtype=torch.float32)

B, H = 4, 96
torch.manual_seed(42)
y_true   = torch.rand(B, H) + 0.5
sol_true = greedy_peak(y_true)

#  TEST 1  SPO+ layer 
# Source : PredOpt/predopt-benchmarks · Energy/Trainer/diff_layer.py
# Changes: (1) remove solver arg  (2) scalar peak forward  (3) [B,H] gradient
def test_spo():
    class _SPO(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y_hat, y_true, sol_true):
            sol_hat = greedy_peak(y_hat)
            ctx.save_for_backward(y_hat, y_true, sol_true)
            return (sol_hat - sol_true).clamp(min=0).mean()
        @staticmethod
        def backward(ctx, g):
            y_hat, y_true, sol_true = ctx.saved_tensors
            sol_spo = greedy_peak(2 * y_hat - y_true)
            grad = (sol_true - sol_spo).unsqueeze(1).expand(-1, H) / H
            return grad * g, None, None
    y_hat = torch.rand(B, H, requires_grad=True)
    loss  = _SPO.apply(y_hat, y_true, sol_true); loss.backward()
    assert y_hat.grad.shape == (B, H)
    return f"loss={loss.item():.4f} | grad {list(y_hat.grad.shape)} | norm={y_hat.grad.norm():.4f}"
check("SPO+ layer  (PredOpt diff_layer.py — 3 changes)", test_spo)

# TEST 2  DBB layer
# Source : PredOpt/predopt-benchmarks · Energy/Trainer/diff_layer.py
# Changes: same 3 changes as SPO+
def test_dbb():
    class _DBB(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y_hat, y_true, sol_true):
            sol_hat = greedy_peak(y_hat)
            ctx.save_for_backward(y_hat, sol_hat)
            return sol_hat.mean()
        @staticmethod
        def backward(ctx, g):
            y_hat, sol_hat = ctx.saved_tensors
            sol_p = greedy_peak(y_hat + 1.0 * g)
            grad  = (-(sol_hat - sol_p)).unsqueeze(1).expand(-1, H) / H
            return grad * g, None, None
    y_hat = torch.rand(B, H, requires_grad=True)
    loss  = _DBB.apply(y_hat, y_true, sol_true); loss.backward()
    assert y_hat.grad.shape == (B, H)
    return f"loss={loss.item():.4f} | grad {list(y_hat.grad.shape)} | norm={y_hat.grad.norm():.4f}"
check("DBB layer   (PredOpt diff_layer.py — 3 changes)", test_dbb)

# TEST 3  NCE loss 
# Source : PredOpt/predopt-benchmarks · Energy/Trainer/CacheLosses.py
# Changes: NONE, verbatim from repo
def test_nce():
    import torch.nn.functional as F
    class NCE(nn.Module):
        def __init__(self, minimize=True):
            super().__init__(); self.mm = 1 if minimize else -1
        def forward(self, y_hat, y_true, sol_true, cache):
            loss = 0; mm = self.mm
            for ii in range(len(y_hat)):
                loss += ((mm * (sol_true[ii] - cache) * y_hat[ii]).sum(dim=1)).mean()
            return loss / len(y_hat)
    y_hat = torch.rand(B, H, requires_grad=True)
    loss  = NCE()(y_hat, y_true, torch.rand(B, 1), torch.rand(20, 1))
    loss.backward()
    assert y_hat.grad.shape == (B, H)
    return f"loss={loss.item():.4f} | grad {list(y_hat.grad.shape)} | UNCHANGED from repo"
check("NCE loss    (PredOpt CacheLosses.py — unchanged)", test_nce)

# TEST 4  MultiDeT  (PTO / TFMR baseline
# Source : wangc1073/multidet  (Wang et al. 2022, IEEE Trans. Smart Grid)
# Role   : reference [29] in Kim et al. — single-scale Transformer they beat
# Changes: input dims [B,672,19]; single decoder; output [B,96]
def test_multidet():
    class _Encoder(nn.Module):
        def __init__(self, num_layers, d_model):
            super().__init__()
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
            self.enc  = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.norm = nn.LayerNorm(d_model)
        def forward(self, x): return self.norm(self.enc(x))

    class MultiDeT_adapted(nn.Module):
        """Wang et al. 2022 (wangc1073/multidet) adapted for EirGrid:
        input [B,672,19] → output [B,96] single peak forecast."""
        def __init__(self, in_f=19, seq_len=672, pred_len=96, d=64):
            super().__init__()
            self.proj = nn.Linear(in_f, d)
            self.enc  = _Encoder(2, d)
            self.head = nn.Linear(seq_len * d, pred_len)
        def forward(self, x):
            z = self.enc(self.proj(x))
            return self.head(z.reshape(z.size(0), -1))

    m = MultiDeT_adapted(); x = torch.rand(B, 672, 19)
    out = m(x); out.mean().backward()
    assert out.shape == (B, 96)
    return f"in {list(x.shape)} → out {list(out.shape)} | {sum(p.numel() for p in m.parameters()):,} params | backward OK"
check("MultiDeT    (wangc1073/multidet — PTO baseline, adapted dims)", test_multidet)

# TEST 5  Kim et al. backbone  (NeOpt + DBB baseline) 
# Source : Kim et al. (2025) Section III-A, Equations (3)-(5), Figure 1
# No public repo, implemented directly from paper specification
def test_kim_backbone():
    class KimBackbone(nn.Module):
        """Kim et al. (2025) Sec III-A Eqs (3)-(5).
        Patch embedding → TransformerEncoder → mean + log-var heads."""
        def __init__(self, in_f=19, seq_len=672, patch_len=16,
                     stride=8, d=128, n_heads=8, n_layers=3, pred_len=96):
            super().__init__()
            self.patch_len = patch_len; self.stride = stride
            n_p = (seq_len - patch_len) // stride + 1
            self.proj    = nn.Linear(patch_len * in_f, d)
            self.pos     = nn.Parameter(torch.zeros(1, n_p, d))
            enc_l = nn.TransformerEncoderLayer(d, n_heads, d*2, 0.1, batch_first=True)
            self.tf      = nn.TransformerEncoder(enc_l, n_layers)
            self.h_mean  = nn.Linear(n_p * d, pred_len)
            self.h_lvar  = nn.Linear(n_p * d, pred_len)
        def forward(self, x):
            B, L, F = x.shape
            p = x.unfold(1, self.patch_len, self.stride).contiguous().view(B, -1, self.patch_len * F)
            z = self.tf(self.proj(p) + self.pos).reshape(B, -1)
            return self.h_mean(z), self.h_lvar(z)

    m = KimBackbone(); x = torch.rand(B, 672, 19)
    mu, lv = m(x); (mu.mean() + lv.mean()).backward()
    assert mu.shape == (B, 96)
    return f"in {list(x.shape)} → mean+logvar {list(mu.shape)} | {sum(p.numel() for p in m.parameters()):,} params | backward OK"
check("Kim backbone (Kim 2025 Sec III-A — NeOpt/DBB baseline, paper spec)", test_kim_backbone)

# TEST 6  MSABlock  (my contribution)
# Source : Debnath et al. (2026) Fig.1, notebook CNN spec
# Debnath repo = Keras .h5 only, no reusable PyTorch code
# Adaptations: LSTM removed; 4 parallel CNN branches [4,16,96,192]; cross-scale attention
def test_msa_block():
    class MSABlock(nn.Module):
        """Debnath et al. (2026) multi-scale CNN — MSA-NeOpt backbone.
        Kernels [4,16,96,192] = [1h,4h,24h,48h] at 15-min resolution."""
        def __init__(self, in_f=19, d=64, pred_len=96):
            super().__init__()
            self.branches = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(in_f, d, k, padding=k//2),
                    nn.BatchNorm1d(d), nn.ReLU(),
                    nn.AdaptiveAvgPool1d(pred_len)
                ) for k in [4, 16, 96, 192]
            ])
            fd = d * 4
            self.attn = nn.MultiheadAttention(fd, num_heads=4, batch_first=True)
            self.norm = nn.LayerNorm(fd)
            self.head = nn.Linear(fd, 1)
        def forward(self, x):
            x = x.permute(0, 2, 1)
            z = torch.cat([b(x) for b in self.branches], dim=1).permute(0, 2, 1)
            a, _ = self.attn(z, z, z)
            return self.head(self.norm(z + a)).squeeze(-1)

    m = MSABlock(); x = torch.rand(B, 672, 19)
    out = m(x); out.mean().backward()
    assert out.shape == (B, 96)
    return f"in {list(x.shape)} → out {list(out.shape)} | {sum(p.numel() for p in m.parameters()):,} params | kernels [4,16,96,192]"
check("MSABlock    (Debnath 2026. my contribution, PyTorch from paper spec)", test_msa_block)

# TEST 7  LP battery solver 
# Source : Kim et al. (2025) Equations (7)-(14)
# Separate charge/discharge variables for LP linearity
def test_lp():
    try:
        import cvxpy as cp
    except ImportError:
        return "SKIP — run: pip install cvxpy cvxopt"

    def lp_peak_shave(load, bat_cap=0.5, bat_power=0.25, eta_c=0.95, eta_d=0.95):
        T = len(load)
        u_d  = cp.Variable(T, nonneg=True)
        u_c  = cp.Variable(T, nonneg=True)
        s    = cp.Variable(T + 1)
        peak = cp.Variable(nonneg=True)
        cons = [s[0] == bat_cap / 2]
        for t in range(T):
            cons += [
                load[t] + u_d[t] - u_c[t] <= peak,
                s[t+1] == s[t] - u_d[t]/eta_d + u_c[t]*eta_c,
                s[t+1] >= 0, s[t+1] <= bat_cap,
                u_d[t] <= bat_power, u_c[t] <= bat_power,
            ]
        cp.Problem(cp.Minimize(peak), cons).solve(solver=cp.GLPK)
        return float(peak.value)

    np.random.seed(0)
    load = np.random.rand(96) + 0.3
    result = lp_peak_shave(load); no_bat = float(load.max())
    assert result < no_bat
    return f"no-battery={no_bat:.3f} pu → with-battery={result:.3f} pu | reduction={no_bat-result:.3f} pu"
check("LP solver   (Kim 2025 Eqs 7-14 — cvxpy + GLPK)", test_lp)

# SUMMARY 

passed  = sum(1 for r in results if r[0] == "PASS")
skipped = sum(1 for r in results if "SKIP" in r[2])
failed  = sum(1 for r in results if r[0] == "FAIL" and "SKIP" not in r[2])
print(f"  {passed}/{len(results)} passed   {failed} failed   {skipped} skipped")

if failed:
    print("\nFailed:")
    for s, n, d in results:
        if s == "FAIL" and "SKIP" not in d:
            print(f"  ✗ {n}\n    {d}")