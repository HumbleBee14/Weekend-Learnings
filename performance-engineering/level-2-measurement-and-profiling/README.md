# Level 2 — Measurement and Profiling

You can't optimize what you can't measure. This level explains proper benchmarking methodology, the `perf` tool, flame graphs, and JVM-specific profiling — so every optimization in later levels is backed by real numbers.

## Topics

### 01 — Benchmarking Methodology
- Why naive benchmarks are wrong: dead code elimination, JIT warmup, coordinated omission
- Gil Tene's "How NOT to Measure Latency" — the talk every perf engineer must watch
- Proper tools: JMH (Java), Google Benchmark (C++), Criterion (Rust)
- HdrHistogram for latency recording without coordinated omission
- **Code**: Bad benchmark (all the mistakes) vs good benchmark (proper methodology) in all 3 languages

### 02 — perf and Hardware Counters
- `perf stat` for hardware counter reading (cache misses, branch misses, IPC)
- `perf record` + `perf report` for sampling-based profiling
- Key metrics: IPC (good: 2-4, bad: <1), cache miss rate, branch miss rate
- **Code**: Deliberately cache-unfriendly program to profile and fix
- **Output**: `perf stat` cheat sheet

### 03 — Flame Graphs
- What flame graphs show (x-axis = time proportion, y-axis = stack depth)
- Generating: `perf` → `flamegraph.pl` (C++), `async-profiler` (Java), `cargo flamegraph` (Rust)
- On-CPU vs off-CPU flame graphs
- **Code**: Program with a clear hotspot, generate flame graphs from all 3 languages

### 04 — Compiler Explorer and Compiler Optimizations
- **Reading x86 assembly** (just enough: `mov`, `cmp`, `jg`, `cmov`, `vaddps`, `lock cmpxchg`)
- **Using godbolt.org** to see what the compiler actually generates
- **Optimization levels**: -O0 vs -O1 vs -O2 vs -O3 — what changes at each step
- **Spotting auto-vectorization** (SIMD) in compiler output — look for `vaddps`, `vmovdqu` instructions
- **Compiler optimizations that matter for performance**:
  - **Inlining**: Compiler replaces function call with function body — eliminates call overhead, enables further optimization. `-finline-limit`, `__attribute__((always_inline))`, Rust `#[inline(always)]`. Check with `-Rpass=inline` (Clang) or godbolt.
  - **Loop unrolling**: Compiler replicates loop body N times — reduces loop overhead, enables SIMD. `-funroll-loops`. Sometimes hurts (code bloat → instruction cache pressure).
  - **Aliasing and `restrict`**: If two pointers might point to the same memory, the compiler can't reorder loads/stores freely. `restrict` (C) / `__restrict__` (C++) promises no aliasing → enables much better optimization. Rust's borrow checker gives this guarantee automatically (one `&mut` at a time).
  - **Auto-vectorization limits**: Compiler can vectorize simple loops but fails on: loop-carried dependencies, non-contiguous access, complex control flow, potential aliasing. Use `-fopt-info-vec-missed` (GCC) or `-Rpass-missed=loop-vectorize` (Clang) to see WHY it didn't vectorize.
  - **Undefined behavior (UB) impact on C++**: The compiler ASSUMES UB never happens and optimizes accordingly. Signed overflow UB lets it optimize `x + 1 > x` to `true`. This can cause unexpected behavior AND unexpected performance (the compiler removes "unnecessary" checks). Rust avoids most UB by default (checked arithmetic in debug, wrapping in release).
  - **Link-Time Optimization (LTO)**: `-flto` enables optimization across translation units — the compiler sees your entire program and can inline, devirtualize, and eliminate dead code across file boundaries. Significant wins for large C++ projects.
  - **Profile-Guided Optimization (PGO)**: Compile → run with profiling → recompile using profile data. The compiler knows which branches are hot, which functions to inline aggressively, which code paths to optimize. 10-20% improvement typical.
- **Code**: Same C++ function at different optimization levels — diff the assembly on godbolt. Auto-vectorization success vs failure examples. `restrict` impact demo.
- **Godbolt flags to try**: `-O2 -march=native`, `-O3 -ffast-math`, `-O2 -flto`, `-O2 -fopt-info-vec-missed`

### 05 — Java-Specific Profiling
- Why Java profiling is different: JIT, GC, safepoints
- async-profiler: CPU, allocation, and lock profiling with flame graphs
- JFR + JMC: production-safe profiling
- GC log analysis: proving GC is (or isn't) the bottleneck
- JIT compilation logging: `-XX:+PrintCompilation`, what gets compiled and what doesn't
- **Code**: GC pressure demo, JIT warmup demo, escape analysis demo — showing what C++/Rust don't deal with

## Key Reading for This Level

- Gil Tene — "How NOT to Measure Latency" (talk — MUST WATCH)
- Brendan Gregg — "perf Examples" (brendangregg.com)
- Brendan Gregg — "Flame Graphs" (brendangregg.com/flamegraphs.html)
- Denis Bakhvalov — "Performance Analysis and Tuning on Modern CPUs"
- Matt Godbolt — "What Has My Compiler Done for Me Lately?" (CppCon talk)
