"""One LLaMA-shaped decoder block, kept compact so we can compile the whole thing.

No HuggingFace `**kwargs` / Cache plumbing — those are demonstrated as graph
breaks in sub-module 03. This file is intentionally clean so the capstone is
about the piecewise-cuda-graph pattern, not about cleaning up the Transformers
library.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LLaMAConfig:
    hidden: int = 4096
    n_heads: int = 32
    head_dim: int = 128
    intermediate: int = 11008
    rope_theta: float = 10000.0
    max_seqlen: int = 4096
    eps: float = 1e-6


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms * self.weight.float()).to(x.dtype)


def build_rope_cache(max_seqlen: int, head_dim: int, theta: float, device, dtype) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seqlen, device=device).float()
    freqs = torch.outer(t, freqs)  # (S, head_dim/2)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return torch.stack([cos, sin], dim=-1)  # (S, head_dim/2, 2)


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    # x: (B, H, S, Dh). rope: (S, Dh/2, 2)
    B, H, S, Dh = x.shape
    x = x.view(B, H, S, Dh // 2, 2)
    cos = rope[:S, :, 0].view(1, 1, S, Dh // 2)
    sin = rope[:S, :, 1].view(1, 1, S, Dh // 2)
    x0, x1 = x[..., 0], x[..., 1]
    r0 = x0 * cos - x1 * sin
    r1 = x0 * sin + x1 * cos
    return torch.stack([r0, r1], dim=-1).view(B, H, S, Dh)


class LLaMABlock(nn.Module):
    def __init__(self, cfg: LLaMAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.norm1 = RMSNorm(cfg.hidden, cfg.eps)
        self.q = nn.Linear(cfg.hidden, cfg.n_heads * cfg.head_dim, bias=False)
        self.k = nn.Linear(cfg.hidden, cfg.n_heads * cfg.head_dim, bias=False)
        self.v = nn.Linear(cfg.hidden, cfg.n_heads * cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.hidden, bias=False)
        self.norm2 = RMSNorm(cfg.hidden, cfg.eps)
        self.w_gate = nn.Linear(cfg.hidden, cfg.intermediate, bias=False)
        self.w_up = nn.Linear(cfg.hidden, cfg.intermediate, bias=False)
        self.w_down = nn.Linear(cfg.intermediate, cfg.hidden, bias=False)
        self.register_buffer(
            "rope",
            build_rope_cache(cfg.max_seqlen, cfg.head_dim, cfg.rope_theta,
                             device="cuda" if torch.cuda.is_available() else "cpu",
                             dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, hidden)
        B, S, _ = x.shape
        H, Dh = self.cfg.n_heads, self.cfg.head_dim

        h = self.norm1(x)
        q = self.q(h).view(B, S, H, Dh).transpose(1, 2)
        k = self.k(h).view(B, S, H, Dh).transpose(1, 2)
        v = self.v(h).view(B, S, H, Dh).transpose(1, 2)
        q = apply_rope(q, self.rope)
        k = apply_rope(k, self.rope)

        # Attention (eager SDPA). In the piecewise variant this gets wrapped
        # as a custom op so it becomes a partition boundary.
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, S, H * Dh)
        x = x + self.o(attn)

        h = self.norm2(x)
        g = F.silu(self.w_gate(h))
        u = self.w_up(h)
        x = x + self.w_down(g * u)
        return x


def make_block(hidden: int = 4096, dtype: torch.dtype = torch.bfloat16, device: str = "cuda") -> LLaMABlock:
    cfg = LLaMAConfig(hidden=hidden)
    # Match GQA-free LLaMA-7B at hidden=4096; adjust intermediate to match if needed.
    if hidden != 4096:
        cfg.intermediate = 4 * hidden  # keep tractable on small GPUs
    block = LLaMABlock(cfg).to(device=device, dtype=dtype).eval()
    return block
