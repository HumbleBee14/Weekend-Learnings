"""
Fused RMSNorm + RoPE in Triton — the capstone kernel.

Three variants are exposed so benchmark.py can compare them:
  - fused_naive       : single-pass, hand-picked config, non-persistent
  - fused_autotuned   : autotuned with early_config_prune
  - fused_persistent  : autotuned + persistent grid

Convention: interleaved-pair RoPE, fp32 reduction, fp16/bf16 input and output.
Forward only. Backward is a stretch goal — see the "if you want backward" note
at the bottom of this file.
"""

import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Naive variant — single config, non-persistent
# -----------------------------------------------------------------------------

@triton.jit
def fused_naive_kernel(
    out_ptr, x_ptr, w_ptr, cos_ptr, sin_ptr, pos_ptr,
    n_rows, n_cols, n_half, cos_sin_stride,
    eps,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x_row_ptr = x_ptr + row * n_cols
    out_row_ptr = out_ptr + row * n_cols

    # Load row + weight, do RMSNorm in fp32
    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    sum_sq = tl.sum(x * x, axis=0)
    inv_rms = 1.0 / tl.sqrt(sum_sq / n_cols + eps)
    x = x * inv_rms * w

    # Now RoPE. We need to separately gather even-indexed and odd-indexed elements,
    # apply rotation with cos/sin, and write back interleaved.
    pos = tl.load(pos_ptr + row)  # scalar int

    half_cols = tl.arange(0, BLOCK_HALF)
    half_mask = half_cols < n_half

    # Even (= cols 0, 2, 4, ...) and odd (= cols 1, 3, 5, ...) elements from x:
    # since `x` is already loaded in register order, we can use tl.where slicing
    # via two strided gathers. Triton's preferred idiom here is two separate loads
    # using strided pointers, but since x is in registers we use a different trick:
    # reshape via index arithmetic and tl.where.
    #
    # The clean approach: do two separate loads of the row, one with stride 2.
    # That requires the row to be loaded twice. To avoid that, we re-read the
    # row at strided offsets. This is fine — the data is already in L1 from the
    # first load.
    even_offsets = half_cols * 2
    odd_offsets = half_cols * 2 + 1

    # Since `x` was already computed in registers above, we need to pick out
    # interleaved pairs. Use tl.load on the same row with stride. (One additional
    # round-trip through register file via masked-gather; the actual HBM was already paid.)
    # In practice, the cleanest way is to redo the normalize after re-loading even/odd.
    # For pedagogical clarity, we reload — the L1 cache makes this cheap.
    x_even_raw = tl.load(x_row_ptr + even_offsets, mask=half_mask, other=0.0).to(tl.float32)
    w_even = tl.load(w_ptr + even_offsets, mask=half_mask, other=0.0).to(tl.float32)
    x_odd_raw = tl.load(x_row_ptr + odd_offsets, mask=half_mask, other=0.0).to(tl.float32)
    w_odd = tl.load(w_ptr + odd_offsets, mask=half_mask, other=0.0).to(tl.float32)

    x_even = x_even_raw * inv_rms * w_even
    x_odd = x_odd_raw * inv_rms * w_odd

    cos = tl.load(cos_ptr + pos * cos_sin_stride + half_cols, mask=half_mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + pos * cos_sin_stride + half_cols, mask=half_mask, other=0.0).to(tl.float32)

    y_even = x_even * cos - x_odd * sin
    y_odd = x_even * sin + x_odd * cos

    tl.store(out_row_ptr + even_offsets, y_even.to(out_ptr.dtype.element_ty), mask=half_mask)
    tl.store(out_row_ptr + odd_offsets, y_odd.to(out_ptr.dtype.element_ty), mask=half_mask)


def fused_naive(x, w, cos, sin, pos, eps=1e-6):
    orig_shape = x.shape
    H = orig_shape[-1]
    x_flat = x.reshape(-1, H).contiguous()
    pos_flat = pos.reshape(-1).contiguous()
    n_rows = x_flat.shape[0]

    out = torch.empty_like(x_flat)
    BLOCK_SIZE = triton.next_power_of_2(H)
    BLOCK_HALF = BLOCK_SIZE // 2

    fused_naive_kernel[(n_rows,)](
        out, x_flat, w, cos, sin, pos_flat,
        n_rows, H, H // 2, cos.stride(0),
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_HALF=BLOCK_HALF,
        num_warps=8,
    )
    return out.reshape(orig_shape)


# -----------------------------------------------------------------------------
# Autotuned variant
# -----------------------------------------------------------------------------

def _prune(configs, named_args, **kwargs):
    n_cols = named_args["n_cols"]
    target = triton.next_power_of_2(n_cols)
    out = []
    for cfg in configs:
        if cfg.kwargs["BLOCK_SIZE"] < target or cfg.kwargs["BLOCK_SIZE"] > 2 * target:
            continue
        if cfg.kwargs["BLOCK_SIZE"] % (cfg.num_warps * 32) != 0:
            continue
        out.append(cfg)
    return out


_CONFIGS = [
    triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=ns)
    for bs in [1024, 2048, 4096, 8192]
    for nw in [4, 8, 16]
    for ns in [2, 3, 4]
]


@triton.autotune(configs=_CONFIGS, key=["n_cols"],
                 prune_configs_by={"early_config_prune": _prune})
@triton.jit
def fused_auto_kernel(
    out_ptr, x_ptr, w_ptr, cos_ptr, sin_ptr, pos_ptr,
    n_rows, n_cols, n_half, cos_sin_stride,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x_row_ptr = x_ptr + row * n_cols
    out_row_ptr = out_ptr + row * n_cols

    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    sum_sq = tl.sum(x * x, axis=0)
    inv_rms = 1.0 / tl.sqrt(sum_sq / n_cols + eps)
    x_normed = x * inv_rms * w  # [BLOCK_SIZE]

    pos = tl.load(pos_ptr + row)

    # Extract even/odd from the already-normed register tile via tl.where.
    # cols[i] is even iff cols[i] & 1 == 0. We build two register vectors
    # by reduction-pair: pair_idx = cols // 2. For even cols, x_normed[c] is
    # the "even of pair c//2". For odd cols, x_normed[c] is the "odd of pair c//2".
    # We then load cos/sin at pair_idx and compute the rotated values directly.
    pair_idx = cols // 2
    cos = tl.load(cos_ptr + pos * cos_sin_stride + pair_idx, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + pos * cos_sin_stride + pair_idx, mask=mask, other=0.0).to(tl.float32)

    # Build the "partner" element for each col:
    #   if col is even (partner is at col+1): we want x[col] * cos - x[col+1] * sin
    #   if col is odd  (partner is at col-1): we want x[col-1] * sin + x[col] * cos
    # Easier: load partner from x_row_ptr at col XOR 1 — that flips between even/odd.
    partner_offsets = cols ^ 1
    x_partner_raw = tl.load(x_row_ptr + partner_offsets, mask=mask, other=0.0).to(tl.float32)
    w_partner = tl.load(w_ptr + partner_offsets, mask=mask, other=0.0).to(tl.float32)
    x_partner = x_partner_raw * inv_rms * w_partner

    is_even = (cols & 1) == 0
    out_val = tl.where(
        is_even,
        x_normed * cos - x_partner * sin,
        x_partner * sin + x_normed * cos,
    )
    tl.store(out_row_ptr + cols, out_val.to(out_ptr.dtype.element_ty), mask=mask)


def fused_autotuned(x, w, cos, sin, pos, eps=1e-6):
    orig_shape = x.shape
    H = orig_shape[-1]
    x_flat = x.reshape(-1, H).contiguous()
    pos_flat = pos.reshape(-1).contiguous()
    n_rows = x_flat.shape[0]
    out = torch.empty_like(x_flat)
    fused_auto_kernel[(n_rows,)](
        out, x_flat, w, cos, sin, pos_flat,
        n_rows, H, H // 2, cos.stride(0),
        eps,
    )
    return out.reshape(orig_shape)


# -----------------------------------------------------------------------------
# Persistent variant
# -----------------------------------------------------------------------------

@triton.autotune(configs=_CONFIGS, key=["n_cols"],
                 prune_configs_by={"early_config_prune": _prune})
@triton.jit
def fused_persistent_kernel(
    out_ptr, x_ptr, w_ptr, cos_ptr, sin_ptr, pos_ptr,
    n_rows, n_cols, n_half, cos_sin_stride,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # Load w ONCE per program (amortized across rows handled by this program).
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    partner_offsets = cols ^ 1
    w_partner = tl.load(w_ptr + partner_offsets, mask=mask, other=0.0).to(tl.float32)
    pair_idx = cols // 2
    is_even = (cols & 1) == 0

    for row in range(pid, n_rows, num_pids):
        x_row_ptr = x_ptr + row * n_cols
        out_row_ptr = out_ptr + row * n_cols

        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq = tl.sum(x * x, axis=0)
        inv_rms = 1.0 / tl.sqrt(sum_sq / n_cols + eps)
        x_normed = x * inv_rms * w

        x_partner_raw = tl.load(x_row_ptr + partner_offsets, mask=mask, other=0.0).to(tl.float32)
        x_partner = x_partner_raw * inv_rms * w_partner

        pos = tl.load(pos_ptr + row)
        cos = tl.load(cos_ptr + pos * cos_sin_stride + pair_idx, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + pos * cos_sin_stride + pair_idx, mask=mask, other=0.0).to(tl.float32)

        out_val = tl.where(
            is_even,
            x_normed * cos - x_partner * sin,
            x_partner * sin + x_normed * cos,
        )
        tl.store(out_row_ptr + cols, out_val.to(out_ptr.dtype.element_ty), mask=mask)


def fused_persistent(x, w, cos, sin, pos, eps=1e-6):
    orig_shape = x.shape
    H = orig_shape[-1]
    x_flat = x.reshape(-1, H).contiguous()
    pos_flat = pos.reshape(-1).contiguous()
    n_rows = x_flat.shape[0]
    out = torch.empty_like(x_flat)
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    grid = (min(num_sms, n_rows),)
    fused_persistent_kernel[grid](
        out, x_flat, w, cos, sin, pos_flat,
        n_rows, H, H // 2, cos.stride(0),
        eps,
    )
    return out.reshape(orig_shape)


# -----------------------------------------------------------------------------
# Backward pass (stretch goal — left as exercise)
# -----------------------------------------------------------------------------
#
# To make the kernel trainable, wrap forward + backward in a torch.autograd.Function.
# The backward needs:
#   - dx given dy: invert the rotation (transpose of the rotation matrix) and then
#     undo the RMSNorm scaling (chain rule through 1/rms and the elementwise w)
#   - dw given dy and saved x_pre_norm: dw[c] = sum over rows of (dy[r,c] * x[r,c] / rms[r])
# Look at Liger-Kernel's rms_norm.py and rope.py for reference implementations.
# This is mechanical once you have forward correct; budget half a day.
