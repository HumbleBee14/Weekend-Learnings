"""
Pure-PyTorch reference for fused RMSNorm + RoPE.

Used as the correctness oracle for your Triton kernel.

Convention used here (matches LLaMA/Mistral/Qwen):
  - RoPE pairs are (x[2i], x[2i+1]), i.e. interleaved pairs along the last dim.
    (Some implementations use halves: x[:H/2] and x[H/2:]. We use the interleaved form.)
  - cos/sin tables shape: [max_seqlen, head_dim/2]; we broadcast across H pairs.
  - RMS uses fp32 reduction even if inputs are fp16/bf16.
"""

import torch


def build_rope_tables(max_seqlen: int, head_dim: int, base: float = 10000.0, device="cuda"):
    """Build cos/sin tables of shape [max_seqlen, head_dim/2]."""
    assert head_dim % 2 == 0
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    positions = torch.arange(max_seqlen, device=device, dtype=torch.float32)
    angles = positions[:, None] * freqs[None, :]  # [seqlen, half]
    return torch.cos(angles).to(torch.float32), torch.sin(angles).to(torch.float32)


def rmsnorm_rope_reference(
    x: torch.Tensor,           # [B, S, H] or [N, H]
    w: torch.Tensor,           # [H]
    cos_table: torch.Tensor,   # [max_seqlen, H/2]
    sin_table: torch.Tensor,   # [max_seqlen, H/2]
    position_ids: torch.Tensor,  # [B, S] or [N], int64
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference implementation: RMSNorm then RoPE applied to interleaved pairs."""
    orig_dtype = x.dtype
    orig_shape = x.shape
    H = orig_shape[-1]
    half = H // 2

    x = x.reshape(-1, H)
    pos = position_ids.reshape(-1)
    N = x.shape[0]

    # RMSNorm in fp32 for numerical stability
    x32 = x.to(torch.float32)
    rms = torch.sqrt((x32 * x32).mean(dim=-1, keepdim=True) + eps)
    x_normed = (x32 / rms) * w.to(torch.float32)

    # Gather cos/sin for each token's position. Shape: [N, H/2]
    cos = cos_table[pos]  # [N, H/2]
    sin = sin_table[pos]

    # Interleaved-pair RoPE: y[2i] = x[2i]*c - x[2i+1]*s; y[2i+1] = x[2i]*s + x[2i+1]*c
    even = x_normed[:, 0::2]  # [N, H/2]
    odd = x_normed[:, 1::2]   # [N, H/2]
    y_even = even * cos - odd * sin
    y_odd = even * sin + odd * cos

    # Interleave back
    out = torch.empty_like(x_normed)
    out[:, 0::2] = y_even
    out[:, 1::2] = y_odd

    return out.reshape(orig_shape).to(orig_dtype)
