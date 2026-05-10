"""
A deliberately-slow training script. Six anti-patterns built in, all togglable.

Run as-is for the worst baseline:
    python slow_model.py

Toggle the FIX_* flags at the top one at a time, re-run, measure delta.
After 5-6 fixes, you should see ~2× total speedup.

Anti-patterns / fixes:
  1. Eager attention (FIX_USE_SDPA = True dispatches to FA2/FA3)
  2. Synchronous H2D copy (FIX_NON_BLOCKING_H2D)
  3. No num_workers / pin_memory (FIX_DATALOADER)
  4. Unfused AdamW (FIX_FUSED_ADAMW)
  5. .cpu() inside the loop (FIX_REMOVE_CPU_SYNC)
  6. No torch.compile (FIX_COMPILE)
"""

import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ===== TOGGLES =====
FIX_USE_SDPA = False             # Use F.scaled_dot_product_attention (FlashAttention)
FIX_NON_BLOCKING_H2D = False     # Async H2D copy
FIX_DATALOADER = False           # num_workers=4, pin_memory=True
FIX_FUSED_ADAMW = False          # AdamW(fused=True)
FIX_REMOVE_CPU_SYNC = False      # Don't .cpu() the loss inside the loop
FIX_COMPILE = False              # torch.compile the model
# ===================


class FakeData(Dataset):
    def __init__(self, n: int = 500, seq_len: int = 256, vocab: int = 1024):
        self.n = n
        self.seq_len = seq_len
        self.vocab = vocab

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Simulate tokenization work
        time.sleep(0.002)
        x = torch.randint(0, self.vocab, (self.seq_len,))
        y = torch.randint(0, self.vocab, (self.seq_len,))
        return x, y


class TinyAttention(nn.Module):
    """One self-attention block. Switches between eager and SDPA based on flag."""

    def __init__(self, d_model: int = 512, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # each: (B, N, H, D)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if FIX_USE_SDPA:
            # FlashAttention path on Ampere/Hopper
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Eager attention — materializes the (N, N) score matrix in HBM
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            mask = torch.ones(N, N, device=x.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, -float("inf"))
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, N, self.d_model)
        return self.out_proj(out)


class TinyTransformer(nn.Module):
    def __init__(self, vocab: int = 1024, d_model: int = 512, n_layers: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": TinyAttention(d_model),
                "norm1": nn.LayerNorm(d_model),
                "mlp": nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Linear(4 * d_model, d_model),
                ),
                "norm2": nn.LayerNorm(d_model),
            })
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = h + layer["attn"](layer["norm1"](h))
            h = h + layer["mlp"](layer["norm2"](h))
        return self.head(h)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("Needs CUDA.")
    device = torch.device("cuda")

    print(f"Config:")
    print(f"  FIX_USE_SDPA          = {FIX_USE_SDPA}")
    print(f"  FIX_NON_BLOCKING_H2D  = {FIX_NON_BLOCKING_H2D}")
    print(f"  FIX_DATALOADER        = {FIX_DATALOADER}")
    print(f"  FIX_FUSED_ADAMW       = {FIX_FUSED_ADAMW}")
    print(f"  FIX_REMOVE_CPU_SYNC   = {FIX_REMOVE_CPU_SYNC}")
    print(f"  FIX_COMPILE           = {FIX_COMPILE}")
    print()

    model = TinyTransformer().to(device)

    if FIX_COMPILE:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=FIX_FUSED_ADAMW)

    if FIX_DATALOADER:
        loader = DataLoader(FakeData(), batch_size=4, num_workers=4,
                           pin_memory=True, persistent_workers=True)
    else:
        loader = DataLoader(FakeData(), batch_size=4, num_workers=0)

    # Warmup (especially important for torch.compile — first iters include compilation)
    print("Warmup...")
    for warm_step, batch in enumerate(loader):
        x, y = batch
        x = x.to(device, non_blocking=FIX_NON_BLOCKING_H2D)
        y = y.to(device, non_blocking=FIX_NON_BLOCKING_H2D)
        out = model(x)
        loss = F.cross_entropy(out.reshape(-1, out.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if warm_step >= 3:
            break
    torch.cuda.synchronize()

    # Measured loop
    print("Measuring 30 steps...")
    losses = []
    t0 = time.perf_counter()
    n_steps = 0
    for batch in loader:
        if n_steps >= 30:
            break
        x, y = batch
        x = x.to(device, non_blocking=FIX_NON_BLOCKING_H2D)
        y = y.to(device, non_blocking=FIX_NON_BLOCKING_H2D)

        out = model(x)
        loss = F.cross_entropy(out.reshape(-1, out.size(-1)), y.reshape(-1))

        if not FIX_REMOVE_CPU_SYNC:
            # ANTI-PATTERN: forces a GPU sync inside the training loop
            losses.append(loss.cpu().item())
        else:
            # Defer the .item() — accumulate on GPU, sync at the end
            losses.append(loss.detach())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        n_steps += 1

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    if FIX_REMOVE_CPU_SYNC:
        # Now sync everything at once
        losses = [l.item() for l in losses]

    ms_per_step = elapsed / n_steps * 1000
    tokens_per_step = 4 * 256  # batch * seq_len
    tokens_per_sec = tokens_per_step / (ms_per_step / 1000)

    print()
    print(f"Results:")
    print(f"  ms / step:        {ms_per_step:.1f} ms")
    print(f"  tokens / sec:     {tokens_per_sec:.0f}")
    print(f"  final loss:       {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
