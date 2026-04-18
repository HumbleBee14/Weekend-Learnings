# Level 1 — Hardware Fundamentals

Understand the machine before trying to make it fast. Everything in performance engineering traces back to how CPUs, caches, and memory actually work.

## Topics

### 01 — CPU Pipeline and Branch Prediction
- How instruction pipelines work (fetch → decode → execute → memory → writeback)
- Why branch mispredictions flush the pipeline (~5ns, 15-20 wasted cycles)
- The classic demo: sorted vs unsorted array (3-6x difference from branch prediction alone)
- Branchless alternatives: `__builtin_expect`, conditional moves (`cmov`), bitwise tricks
- **Code**: Java (JMH) vs C++ (Google Benchmark + `__builtin_expect`) vs Rust (branchless select)
- **Measure**: `perf stat -e branches,branch-misses`

### 02 — CPU Caches and Memory Hierarchy
- L1 (0.5ns) → L2 (7ns) → L3 (30ns) → RAM (100ns) — the 200x gap between hit and miss
- Cache lines (64 bytes) — the fundamental data transfer unit
- Spatial locality (sequential access) vs temporal locality (reuse)
- Why Java's `ArrayList<Integer>` is 10-50x slower than C++ `vector<int>` (pointer chasing vs contiguous)
- **Code**: Java pointer-chasing demo vs C++/Rust contiguous access + stride pattern benchmarks
- **Measure**: `perf stat -e cache-references,cache-misses,L1-dcache-load-misses`

### 03 — False Sharing
- MESI cache coherence protocol (Modified, Exclusive, Shared, Invalid)
- Two threads writing different variables on the same cache line → 10-30x slowdown
- Detection: `perf c2c`
- Fixes: Java `@Contended`, C++ `alignas(64)`, Rust `CachePadded<T>`
- **Code**: Before/after demos in all 3 languages
- **Measure**: `perf c2c record && perf c2c report`

### 04 — NUMA and Memory Topology
- Non-Uniform Memory Access — local vs remote memory (1.5-3x penalty)
- Using `lstopo`, `numactl`, `numastat` to visualize and control placement
- Why NUMA-unaware code silently loses 30-50% performance on multi-socket servers
- **Code**: C++ demo allocating on local vs remote NUMA node

### 05 — TLB and Huge Pages
- Page tables, TLB (Translation Lookaside Buffer), TLB misses
- 4KB pages → TLB covers 256KB. 2MB pages → TLB covers 128MB. Massive difference.
- Transparent Huge Pages (THP) vs explicit (`mmap MAP_HUGETLB`)
- Why HFT always uses explicit huge pages (THP causes latency spikes from compaction)
- **Code**: C++ mmap with huge pages, Java `-XX:+UseLargePages`
- **Measure**: `perf stat -e dTLB-load-misses`

## Key Reading for This Level

- Ulrich Drepper — "What Every Programmer Should Know About Memory" (Sections 3-6)
- Algorithmica HPC — "CPU Cache" chapter (en.algorithmica.org/hpc/)
- MIT 6.172 — Lectures 1-4
- Scott Meyers — "CPU Caches and Why You Care" (talk)
- Mike Acton — "Data-Oriented Design and C++" (CppCon 2014 talk)

## Mini-Project: Matrix Multiply (`06-mini-project-matrix-multiply/`)

**Apply everything from this level in one focused build.**

Implement matrix multiplication in Java, C++, and Rust. Same algorithm (naive, then optimized), measure how hardware fundamentals dominate performance.

| Variant | What it demonstrates | Expected result |
|---------|---------------------|-----------------|
| **Naive (ijk order)** | Cache-hostile: inner loop strides across rows of B | Slow — high L1 cache miss rate |
| **Transposed (ikj order)** | Cache-friendly: inner loop is sequential on both A and B^T | 3-10x faster — sequential access, prefetcher works |
| **Blocked (tiled)** | Fits sub-matrices into L1 cache, reuses them before eviction | Fastest — minimal cache misses, exploits temporal locality |
| **Java baseline** | All 3 variants in Java — show that even with JIT, cache behavior matters | Java is comparable to C++ for compute, but object overhead hurts for complex types |

**Measure with**: `perf stat -e L1-dcache-load-misses,LLC-load-misses,instructions,cycles`
**Produce**: Flame graphs for each variant, cache miss rates, IPC comparison, written analysis

This is the "hello world" of performance engineering — the difference between ijk and ikj order is 100% explained by cache behavior.
