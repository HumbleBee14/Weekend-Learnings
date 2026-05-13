# How a GPU actually executes work

A GPU is a different kind of machine than a CPU and the differences are not cosmetic. If you carry a CPU mental model into kernel writing — "the program runs one step at a time, branches go where I tell them, memory is just memory" — you will spend weeks confused. This document gives you the mental model that makes Triton, FlashAttention, and the rest of this track make sense.

We use NVIDIA terminology throughout because that's what most learning material assumes. AMD's terminology is parallel (we call out the translation each time). The hardware-level concepts are the same.

## 1. The machine has many small cores, not few big ones

A modern CPU has maybe 8–64 cores. Each one is large, has aggressive out-of-order execution, big private caches, branch predictors, and can run a thread mostly independently of the others. They are designed for low-latency single-thread performance.

A GPU has thousands of cores. An H100 has 16,896 FP32 ALUs spread across 132 Streaming Multiprocessors (SMs). A B200 has more. A consumer RTX 5090 has fewer. None of these cores are individually impressive — they're small, in-order, and a single core in isolation is slower than your laptop CPU. The point is not the individual core. The point is that you have *so many* of them that if you can keep them all busy doing useful work, the aggregate throughput is staggering: an H100 hits 989 TFLOPS in FP16 tensor-core matmul. Your laptop does about 1 TFLOP if you push it.

The first rule of GPU programming follows directly: **your work has to split into many independent pieces, or you cannot use the machine.** A workload that has to run sequentially with branches that depend on previous results is the wrong workload for a GPU. Matrix multiply is great because every output element can be computed independently. Recursion is terrible.

(AMD calls SMs "Compute Units" or CUs. An MI300X has 304 CUs. The numbers don't map one-to-one — an AMD CU is structured slightly differently — but for our purposes "SM" and "CU" mean the same thing: the independent worker unit of the chip.)

## 2. Each SM runs warps of 32 lanes that share an instruction pointer

This is the most important fact in this document.

Inside one SM, the smallest unit of execution is a **warp**: 32 lanes. All 32 lanes execute the same instruction at the same time. They share an instruction pointer. They are not 32 independent threads — they are one fused micro-thread with 32-wide vector hardware.

A consequence: if your code branches differently across lanes — `if x > 0: A else: B` where some lanes go one way and others go the other — the hardware does not magically run both branches in parallel. It runs A with the lanes that took A active and the others paused, then runs B with the others active. This is called **warp divergence**, and it is the single most common reason a beginner's GPU code is slow.

```
   no divergence (all 32 lanes do the same):

     lane:  0   1   2   3  ...  29  30  31
     time → A   A   A   A  ...  A   A   A      one instruction, full throughput

   divergence (half take A, half take B):

     lane:  0   1   2   3  ...  29  30  31
     time → A   ░   A   ░  ...  ░   A   ░      step 1: A-lanes active, B-lanes idle
            ░   B   ░   B  ...  B   ░   B      step 2: B-lanes active, A-lanes idle

   total time ≈ 2× the no-divergence case.
   worst case (32 different paths) = 32× slowdown for that section.
```
*Lanes are not threads. They share one instruction pointer; "branching" really means "take turns, idle the rest."*

A second consequence: loads and stores want to be **coalesced**. If lane 0 of a warp asks for `data[0]`, lane 1 asks for `data[1]`, lane 31 asks for `data[31]`, the hardware can issue *one* memory transaction for the whole warp. If lane 0 asks for `data[0]` and lane 1 asks for `data[1024]`, that's two transactions. Strided or scattered access patterns leave bandwidth on the floor.

AMD calls warps "wavefronts" and they're 64 lanes wide. The rules are the same; the number is different.

The instruction-pointer sharing is also the source of the term "SIMT" — single instruction, multiple threads. You'll see it in NVIDIA documentation. It's the same idea written formally.

## 3. The memory hierarchy is steep — five orders of magnitude steep

A useful number to commit to memory: on an H100, registers are about 500× faster than HBM. Shared memory (also called SRAM) sits between them.

| Memory | Where | Size (H100) | Latency | Bandwidth |
|---|---|---|---|---|
| Registers | Per lane | 256 KB per SM total | 1 cycle | unbounded |
| Shared memory (SRAM) | Per SM | 228 KB | ~30 cycles | ~10 TB/s |
| L2 cache | Chip-wide | 50 MB | ~200 cycles | ~5 TB/s |
| HBM (global memory) | Chip-wide | 80 GB | ~500 cycles | 3.4 TB/s |

The same picture as a stack — small and fast at the top, huge and slow at the bottom:

```
                                   size           latency        bandwidth
                          ┌─────────────────┐
   per-lane registers     │  256 KB / SM    │   1 cycle     ░░░░░░░░░░░░░░░░░░  unbounded
                          ├─────────────────┤
   per-SM SRAM (smem)     │     228 KB      │   ~30 cycles  ██████████░░░░░░░░  ~10 TB/s
                          ├─────────────────┤
   chip-wide L2           │      50 MB      │   ~200 cycles █████░░░░░░░░░░░░░  ~5 TB/s
                          ├─────────────────┤
   HBM (global memory)    │      80 GB      │   ~500 cycles ███░░░░░░░░░░░░░░░  3.4 TB/s
                          └─────────────────┘
                                                ^^^ every load from here
                                                    is what kernels avoid
```
*The game of fast kernels: load each piece of HBM once, do as much work as possible upstairs, write back once.*

For a T4 (free Colab), the numbers are smaller but the *ratios* are similar: ~96 KB SRAM per SM, ~300 GB/s HBM bandwidth. The shape of the memory hierarchy is the same.

The entire game of writing a fast GPU kernel is: **load each piece of HBM data once, do as much work with it as possible while it lives in registers or SRAM, write back to HBM once.** When you hear "kernel fusion saves HBM round-trips" — this is what's meant. When you hear "memory-bound kernel" — it means the kernel reads/writes more HBM bytes per useful FLOP than the chip can sustain, so most cycles are spent waiting on memory.

There's a precise way to think about this. For any kernel, you can compute its **arithmetic intensity**: FLOPs done per byte loaded from HBM. The chip has a peak: H100 is ~290 FLOP/byte. If your kernel's arithmetic intensity is below 290, you are memory-bound — adding more compute won't make it faster. Above 290, you're compute-bound — adding memory bandwidth won't help. This is the roofline model, and you've already met it in `systems-for-ml`. The reason we obsess about it here: most LLM inference kernels (RMSNorm, RoPE, SiLU, residual adds, even attention in decode mode) are memory-bound, sometimes catastrophically so. That's why we spend a whole sub-module taking RMSNorm from 11% to 88% of peak HBM bandwidth — because that's exactly the kind of operation we care about.

## 4. Tensor cores are a separate matrix-multiply engine

Inside each SM there is a small specialized unit called a tensor core (NVIDIA) or matrix core (AMD). It is not a general ALU. It does one thing: take three small matrices (A, B, C), compute `C += A @ B`, write the result back. On H100 it does this for `4×8 × 8×8 → 4×8` FP16 inputs per warp per clock, and the throughput when fully fed is roughly two orders of magnitude higher than the regular FP32 ALUs.

Two things follow:

- **If your kernel is not using tensor cores, you are leaving most of the compute on the floor.** Matrix multiply uses them via `tl.dot` in Triton. Most other operations don't, and that's fine — they're usually memory-bound anyway. The mistake is writing a matmul-shaped operation by hand and not realizing tensor cores would do it for you.
- **Tensor cores want very specific tile shapes and dtypes.** Hopper added FP8 tensor cores; Blackwell added NVFP4 and MXFP8. The block sizes you autotune (BLOCK_M, BLOCK_N, BLOCK_K in a matmul) need to be multiples of the tensor-core fragment size or you waste cycles. The compiler handles most of this — your job is to pick block sizes that are compatible (powers of two, large enough that fragment overhead is small, not so large you can't fit in SRAM).

Blackwell added **TMEM** (Tensor Memory) — a 256 KB-per-SM cache sitting right next to the tensor cores, dedicated to accumulator state. The new tcgen05 MMA instructions write into TMEM. The point is to keep accumulators close to the unit that's using them, decoupled from SRAM pressure. This is one of the major new things about Blackwell, and it's why FA4 is written in CuTe-DSL (which exposes TMEM directly) rather than Triton (which is starting to but doesn't fully).

## 5. The hardware can move memory in the background

Older GPUs (pre-Hopper, so anything before 2023) had a real problem: every load from HBM to SRAM had to be staged by the warp itself. The warp issued a load instruction, the data trickled in, and during those ~500 cycles the warp couldn't compute. You could pipeline loads (issue the next load while crunching the current data), but the bookkeeping was painful and the latency was always staring at you.

Hopper added **TMA — the Tensor Memory Accelerator**. It is dedicated hardware on the chip that, given a descriptor (here's a 128×128 tile of a tensor in HBM, here's where to put it in SRAM), will perform the copy asynchronously. Your warp issues one instruction kicking off the TMA, and goes back to computing. The TMA notifies the warp when the data is in SRAM. Multiple TMA operations can be in flight at once.

This changed kernel writing fundamentally. With TMA you can write the pattern:

```
producer warps: issue TMA loads of tile n+1, n+2, n+3
consumer warps: compute on tile n that was loaded earlier
```

and the producers and consumers run *truly in parallel*. The compute never waits on memory if you've sized the pipeline right. This is **warp specialization** and it's the single most important pattern in modern fast kernels. FlashAttention-3 got its ~1.5× speedup over FA2 almost entirely from this. We do an entire sub-module on it in this level.

Blackwell extends TMA further (more concurrent operations, deeper pipelines, integration with the new tcgen05 family). AMD's equivalent is called **TDM (Tensor Data Movement)** and landed in Triton 3.7 alongside AMD's wave specialization. You don't need to know the PTX-level details — the compiler emits the right instructions. You need to know that this hardware exists and what kernel pattern it enables.

## 6. Putting it together: anatomy of a fast kernel

Here's a fast GEMM kernel in words, on Hopper:

1. Launch as many programs (Triton's word for thread blocks) as you have SMs. Each program runs persistently — it loops over multiple output tiles instead of dying after one.
2. Inside each program, split the warps into producer and consumer groups.
3. The producer warps loop: issue a TMA load of the next A-tile and B-tile into a circular SRAM buffer.
4. The consumer warps loop: take the next loaded (A-tile, B-tile) from the SRAM buffer, run `tl.dot` (which lowers to WGMMA on Hopper or tcgen05 on Blackwell) to accumulate into a register tile.
5. When done with all tiles for this output, the consumers run the epilogue (apply bias, activation, quantization) and store the output back to HBM.
6. The program picks the next output tile from a flat work queue and goes back to step 3.

That's it. Every fast modern GEMM is some variation of this. Same for FA3/FA4 attention, with the inner math swapped from straight matmul to the online-softmax-then-matmul tile pattern.

If you understand the six points in this document, you understand 80% of why fast kernels look the way they do. The remaining 20% is reading source code and writing your own — which is the rest of this level.

---

## Answer key for diagnostics

Don't read this until you've written your own answers in `notes.md`.

<details>
<summary>Click here when ready.</summary>

**Q1. What is the smallest unit of execution on an NVIDIA GPU, and why does it matter?**
A warp: 32 lanes that share an instruction pointer. Matters because divergent branches across lanes serialize, and uncoalesced memory access wastes bandwidth.

**Q2. An H100 has 132 SMs. If I launch a Triton kernel with `grid = (8,)`, what's the problem?**
You're using 8 SMs out of 132. ~94% of the chip is idle. Either increase the grid size or use a persistent kernel that has each program handle multiple output tiles.

**Q3. RMSNorm reads N floats, computes one reduction, writes N floats. What's its arithmetic intensity, roughly?**
About 2 FLOPs per byte (one square + one accumulate per element, plus a few epilogue ops). HBM bandwidth on H100 is 3.4 TB/s, tensor-core peak is ~990 TFLOPS, ridgeline is ~290 FLOP/byte. RMSNorm at ~2 FLOP/byte is deeply memory-bound. The kernel can never compute faster than HBM can feed it; the only thing that matters is reading and writing as little as possible at peak bandwidth.

**Q4. What does warp divergence cost in clock cycles, qualitatively?**
If a warp has K different branch paths and the warp executes all of them serially, total time is roughly K × time_for_one_path. Worst case (all 32 lanes go different ways), 32× slowdown for that section. Solutions: predicate the branch (run both sides, mask the writes), restructure data so similar work goes to the same warp, use `tl.where` instead of `if`.

**Q5. What is TMA and what kernel pattern does it enable?**
Tensor Memory Accelerator — dedicated hardware on Hopper+ that does async tile copies HBM→SRAM. Enables the warp-specialized producer/consumer pattern: some warps issue TMA loads of upcoming tiles while other warps compute on already-loaded tiles. The two run in parallel and HBM latency is hidden behind compute.

**Q6. Why is `tl.dot` important and what hardware does it use?**
It's the Triton primitive that lowers to the tensor-core (NVIDIA) or matrix-core (AMD) instruction. Tensor cores do FP16/BF16 matmul roughly two orders of magnitude faster than the regular ALUs. If a matmul-shaped operation doesn't use them, the kernel is leaving most of the compute on the floor.

**Q7. What is the most common reason a memory-bound kernel runs at 15% of peak HBM bandwidth instead of 85%?**
Usually one or more of: tile size too small (each launch underutilizes the memory bus), multiple passes over the data (the unfused version of an op makes 2–4 HBM round-trips where one would do), uncoalesced access pattern (strides not matching memory layout), no async pipelining (warp stalls fully on each load instead of overlapping with compute), or autotune not searching the right configs. The sub-module 03 RMSNorm journey will show you each of these in turn.

**Q8. Compare a CPU's L1 cache (~32 KB, ~5 cycles) to an H100's shared memory (~228 KB per SM, ~30 cycles). What's the analogy and where does it break?**
Both are small fast caches close to the compute. Analogy holds: both reward locality. Breaks because (a) CPU L1 is per-core and hardware-managed (transparent), GPU SRAM is per-SM and software-managed (you decide what goes there via tile sizes and `tl.load` patterns), (b) GPU SRAM is shared across hundreds of lanes that can communicate through it — there's no CPU equivalent of "all 128 threads in this block reading from the same SRAM tile simultaneously," and (c) GPU SRAM is sized so that the *whole tile* of a matmul lives in it; CPU L1 is sized for typical working sets, not for one specific computation.

</details>
