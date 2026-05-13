# GEMM in Triton, from tiles to autotune

## What is a GEMM and why does its shape matter

GEMM is `C = A @ B + bias` for `A` of shape `(M, K)`, `B` of shape `(K, N)`, `C` of shape `(M, N)`. The math is straightforward — for each `(m, n)` output cell, `C[m, n] = sum_k A[m, k] * B[k, n]`. The interesting question is how you compute the `M * N` cells in parallel without wasting memory bandwidth.

The naive answer is "one thread per output cell, each reads `K` elements from `A` and `B` and accumulates." This is wrong on GPUs for the reason every memory-bandwidth-bound discussion in Level 1 has hammered: you'd reload the same `A` row and the same `B` column many times. Specifically, `A[m, :]` is needed by every cell in output row `m` — that's `N` reloads if you don't reuse. With `M = N = K = 4096`, naive needs `M*N*K*2*4 = 4 * 1e11` bytes loaded; the actual minimum is `(M+N)*K*2*4 + M*N*4 ≈ 8.5 * 1e7` bytes — a 4000× reduction is sitting on the table.

The fix is **tiling**: compute `C` in blocks. Each output tile `(BLOCK_M, BLOCK_N)` is responsible for one chunk of `C`. To compute that chunk, you load tile-by-tile chunks of `A` and `B` along the `K` dimension and accumulate. The K-tiles you load can stay in registers and SRAM for the duration of one tile's work — every element is used `min(BLOCK_M, BLOCK_N)` times before being evicted, giving you the reuse the naive version threw away.

## The standard tile loop

```python
@triton.jit
def matmul_kernel(...):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak)
        b = tl.load(b_ptr + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        acc += tl.dot(a, b)

    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc)
```

Read this carefully. Five things to notice:

- **2D grid.** `program_id(0)` and `program_id(1)` divide the output `C` into `(M/BLOCK_M) × (N/BLOCK_N)` tiles. Each program is responsible for one output tile.
- **The K-loop is sequential within each program.** You iterate over the K dimension in chunks of `BLOCK_K`, accumulating into `acc`. This is the loop where you pay for memory bandwidth — every iteration loads one `(BLOCK_M, BLOCK_K)` slice of `A` and one `(BLOCK_K, BLOCK_N)` slice of `B`.
- **`acc` lives in registers.** A `(BLOCK_M, BLOCK_N)` tile of fp32 accumulator at 128×128 is 64 KB — too big for registers on a single warp, but spread across the warps of one program it fits. The compiler handles the spreading.
- **`tl.dot` is the magic.** `tl.dot(a, b)` lowers to a tensor-core instruction (HMMA on pre-Hopper, WGMMA on Hopper, tcgen05 on Blackwell). One `tl.dot` per K iteration does the work that would otherwise be 128×128×32 = 524,288 individual multiply-add operations — in tens of cycles instead of tens of thousands.
- **`acc` is fp32 but inputs are fp16/bf16.** Tensor cores read low-precision inputs and accumulate in fp32 (or fp16 if you ask, but fp32 is the default for numerical stability). This is the standard mixed-precision matmul.

## What `tl.dot` actually does

`tl.dot(a, b)` accepts a `(BLOCK_M, BLOCK_K)` matrix and a `(BLOCK_K, BLOCK_N)` matrix and returns their product. The compiler decomposes the multiplication into tensor-core fragments:

- H100 WGMMA fragments: 64×8×16 (for FP16)
- B200 tcgen05 fragments: various, larger
- Pre-Hopper HMMA fragments: 16×8×8 (for FP16) or 16×8×16

Your `BLOCK_M`, `BLOCK_N`, `BLOCK_K` need to be multiples of these fragment dimensions or the compiler pads with zeros and wastes cycles. Practical rule: pick `BLOCK_M ∈ {64, 128, 256}`, `BLOCK_N ∈ {64, 128, 256}`, `BLOCK_K ∈ {32, 64, 128}`. Then let autotune find the winner.

## Why `tl.make_tensor_descriptor` matters on Hopper+

In `01_tiled_matmul.py`, every iteration of the K loop computes pointer arithmetic and issues a regular `tl.load`. On pre-Hopper hardware this is fine — it lowers to LDG (global load) instructions and the hardware does what it can.

On Hopper, `tl.make_tensor_descriptor` lowers to TMA (Tensor Memory Accelerator) instructions. TMA is dedicated copy hardware that takes a descriptor (tensor base, shape, strides, tile shape) and asynchronously copies a tile from HBM to SRAM. Three benefits:

- **Single instruction for a tile copy.** No per-element pointer arithmetic, no per-lane address calculation.
- **Asynchronous.** The warps can issue a TMA load and immediately go back to computing. The next sub-module exploits this with warp specialization.
- **Coalesced by construction.** TMA loads always move contiguous bytes; no chance of misaligned access killing your bandwidth.

In Triton 3.4+, `tl.make_tensor_descriptor` is the public, Hopper-aware way to load tiles. It falls back to regular loads on pre-Hopper. You should use it everywhere matmul-shaped, even on pre-Hopper, because (a) it doesn't hurt and (b) when you eventually move to H100 your code is already ready.

## Autotune without wasting hours

You write your matmul kernel and the obvious next question is: what `BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages` should you use? The honest answer: it depends on shape and hardware, and the only way to know is to try.

`@triton.autotune` automates the trial. You give it a list of `triton.Config(...)` objects; on the first call with a new shape, Triton compiles each config and benchmarks them, then picks the winner for that shape. Subsequent calls reuse the winner.

Two problems with the naive approach:

1. **The config space is huge.** Sweeping `BLOCK_M ∈ {32, 64, 128, 256} × BLOCK_N ∈ {32, 64, 128, 256} × BLOCK_K ∈ {32, 64, 128} × num_warps ∈ {2, 4, 8, 16} × num_stages ∈ {2, 3, 4, 5}` = 1536 configs. Most won't even compile. Many are nonsense.
2. **The autotuner runs every config.** No timeout, no early-exit when it sees obvious losers. If a config takes 50ms and another takes 0.5ms, you've burned 100× the compile + warmup cycles on the loser.

`early_config_prune` is the fix. You write a Python function that takes the configs and the runtime args and returns a filtered list. We prune:

- `BLOCK_M * BLOCK_K * 2 + BLOCK_K * BLOCK_N * 2 > SMEM_per_SM` — tile won't fit in SRAM. The compiler would error or spill.
- `BLOCK_M * BLOCK_N * 4 > register_file_per_program` — accumulator won't fit in registers without spilling.
- `BLOCK_M < 16 or BLOCK_N < 16` — too small for tensor-core fragments, will pad.
- `num_warps * 32 > BLOCK_M * BLOCK_N / 16` — too many warps for the tile, lanes idle.
- `K % BLOCK_K != 0 and BLOCK_K not in {32, 64, 128}` — K-tail handling has its own cost.

After pruning, ~10-30 configs remain. Autotune time drops from hours to minutes. Same or better winning config.

## What the winning config tells you about the hardware

Once autotune picks a winner, look at it:

```
BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, num_warps=8, num_stages=4
```

You can read the hardware off this:

- `num_stages=4` means the compiler will keep 4 K-tiles in flight: one being computed on, three in various stages of being loaded. This requires `4 * (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * 2 bytes = 256 KB` of SRAM, which fits H100 (228 KB per SM) only because some of it overlaps with the output accumulator allocation. On T4 you'd see `num_stages=2` win — less SRAM, less pipelining.
- `num_warps=8` means 256 active lanes per program, each handling `(128 * 256) / 256 = 128` accumulator elements. With fp32 accumulator that's 512 bytes per lane, well within the register budget.
- `BLOCK_M=128, BLOCK_N=256` means a 32K-element output tile, which is `32768 * (M*N reuse of K) = 32768 * BLOCK_K = 2M` arithmetic operations per tile worth of K work — enough to amortize the load cost.

You're not expected to derive optimal configs by hand. You're expected to recognize why a config won after the fact — that's how you build intuition for the next shape and the next hardware.

## What to expect in the comparison with `torch.compile`

`torch.compile`'s Inductor backend autotunes its Triton kernels with its own (impressive) heuristics. On H100 for `(M=N=K=4096)` fp16, Inductor typically reaches ~85-95% of cuBLAS. Your hand-written kernel from `03_autotuned.py` should be roughly the same — sometimes Inductor wins by 5%, sometimes you win by 5%.

If your kernel is much slower, the most likely cause is one of:
- You missed `tl.make_tensor_descriptor` and are running pre-Hopper loads on Hopper hardware.
- Your config list doesn't include the actual optimum (e.g., you stop `BLOCK_K` at 64 but H100 likes 128 for this shape).
- You're not pruning right and autotune timed out on a bad config.

If Inductor is much slower, that's interesting too — Inductor's heuristics can fail on shapes outside the training distribution. Worth reading the Inductor-emitted Triton to see what it chose.

`torch.matmul` itself dispatches to cuBLAS — heavily hand-tuned C++ libraries. Beating cuBLAS in Triton is occasionally possible at specific shapes; matching it within 5–10% is the realistic ceiling.
