# Level 4 — Data-Oriented Design

Stop fighting the hardware — make it work FOR you. This level is about laying out data for cache performance, using SIMD to process multiple elements per instruction, eliminating branches, and choosing the right allocator and container.

The core principle: **organize data by how it's accessed, not by how it's conceptually related.** OOP groups data by object (a `Particle` has x, y, z, mass, velocity). DOD groups data by access pattern (iterate over all x's, then all y's). The CPU cache doesn't care about your class hierarchy — it cares about contiguous memory access.

## Topics

### 01 — Struct of Arrays vs Array of Structs
- **AoS (Array of Structs)**: `vector<Particle>` where each Particle has x, y, z, mass, velocity, color...
  - Iteration loads entire struct into cache even if you only need x and y
  - For a 64-byte struct, each cache line holds 1 particle. For 16-byte useful data, you waste 75% of each cache line.
  - Java is AoS by default and ONLY — every object is a heap blob with a 12-16 byte header. No escape. (Project Valhalla may fix this someday with value types, but it's not shipping.)
- **SoA (Struct of Arrays)**: separate `vector<float> x, y, z, mass, velocity`
  - Iteration over x loads 16 floats per cache line (64 bytes / 4 bytes per float)
  - SIMD-friendly: `_mm256_load_ps(&x[i])` loads 8 consecutive x values — perfect alignment
  - 3-5x faster for iteration-heavy workloads (physics simulation, particle systems, columnar analytics)
- **Hybrid AoSoA** (Array of Structs of Arrays): group data in SIMD-width blocks
  - 8 x's, then 8 y's, then 8 z's, then next block — best of both worlds for some workloads
  - Used in game engines and high-performance physics
- **Java's problem**: No control over memory layout. `ArrayList<Particle>` stores pointers to heap-scattered objects. Even `Particle[]` has object headers per element. The only SoA option in Java is parallel primitive arrays (`float[] x, float[] y`) — awkward and error-prone.
- **Code**: AoS vs SoA in Java (show the limitation), C++ (full control), Rust (full control)
- **Measure**: `perf stat -e L1-dcache-load-misses,LLC-load-misses` — SoA should show dramatically fewer cache misses
- **Measure**: IPC (instructions per cycle) — SoA gets higher IPC because the CPU isn't stalling on cache misses

### 02 — SIMD Intrinsics
- **What SIMD is**: Single Instruction, Multiple Data — one instruction processes 4/8/16/32 elements simultaneously
  - SSE: 128-bit registers (4 floats or 2 doubles)
  - AVX2: 256-bit registers (8 floats or 4 doubles) — most widely available
  - AVX-512: 512-bit registers (16 floats) — available on server CPUs, sometimes throttles clock speed
- **Auto-vectorization**: When the compiler does it for you
  - Compile with `-O3 -march=native` and the compiler may vectorize your loops
  - Check on godbolt.org: look for `vaddps`, `vmovdqu`, `vmulps` instructions = vectorized
  - When it fails: loop-carried dependencies, non-contiguous memory, complex control flow, aliased pointers
  - Use `#pragma omp simd` (C++) or `-fopt-info-vec-missed` (GCC) to see why vectorization failed
- **Manual intrinsics** (C++):
  - `_mm256_load_ps(ptr)` — load 8 floats from aligned memory into YMM register
  - `_mm256_add_ps(a, b)` — add 8 float pairs in one instruction
  - `_mm256_store_ps(ptr, result)` — store 8 floats back
  - `_mm256_fmadd_ps(a, b, c)` — fused multiply-add: a*b+c in one instruction (huge for dot products)
  - Alignment matters: `_mm256_load_ps` requires 32-byte alignment, `_mm256_loadu_ps` doesn't but is slower
- **Rust SIMD**:
  - `std::simd` (nightly): `f32x8::from_slice(&data)`, `.reduce_sum()` — portable SIMD
  - `packed_simd2` crate: stable alternative
  - Intrinsics via `std::arch::x86_64` — same as C++ but wrapped in `unsafe`
- **Java Vector API** (incubating, JEP 338/417):
  - `FloatVector.fromArray(SPECIES_256, array, offset)` — loads 8 floats
  - `.add(other)`, `.mul(other)`, `.reduceLanes(ADD)` — vectorized operations
  - Limitations: incubating (API may change), limited operations, can't match manual intrinsics
  - Still a significant improvement over scalar Java — 4-8x speedup possible
- **Study simdjson**: How Daniel Lemire parses JSON at 2.5 GB/s
  - SIMD to find structural characters (`{`, `}`, `[`, `]`, `:`, `,`) in 64 bytes at once
  - Branchless classification of characters
  - This is what peak SIMD usage looks like in real-world code
- **Code**: Scalar sum vs SSE vs AVX2 sum in C++, Rust SIMD sum, Java Vector API sum — benchmark all
- **Measure**: `perf stat -e instructions,cycles` — SIMD version executes fewer instructions for same work (higher throughput per instruction)
- **Godbolt**: Compare scalar loop assembly vs auto-vectorized vs manual intrinsics — see the SIMD instructions

### 03 — Branchless Programming
- **Why branches are expensive**: On unpredictable data, branch predictor guesses wrong ~50% of the time → pipeline flush → ~5ns per mispredict → 15-20 wasted cycles
- **Conditional moves (`cmov`)**: The CPU's branchless instruction
  - `cmov` always executes both paths and selects the result — no pipeline flush, ever
  - The compiler sometimes emits `cmov` for you, but not always — check on godbolt.org
  - Force branchless: `result = condition ? a : b` may or may not compile to `cmov` — depends on compiler heuristics
- **Branchless patterns**:
  - Min/max: `int min = a < b ? a : b;` → compiler often uses `cmov`
  - Abs: `int abs = (x ^ (x >> 31)) - (x >> 31);` — no branch, pure arithmetic
  - Clamp: `int c = x < lo ? lo : (x > hi ? hi : x);` — can be branchless with two `cmov`
  - Predicate masking: `sum += (x > threshold) * x;` — multiply by 0 or 1 instead of branching
  - Bitwise select: `sum += x & -(x > threshold);` — mask with all-ones or all-zeros
  - Lookup tables: replace `switch` with array indexing — one memory load vs N branch checks
- **When NOT to go branchless**: If the branch is highly predictable (>99%), branching is faster than `cmov` because `cmov` always evaluates both sides. Branchless is for unpredictable data.
- **Code**: Branchy vs branchless implementations in C++ and Rust
- **Measure**: `perf stat -e branch-misses,branches` — branchless version should show near-zero branch misses
- **Godbolt**: Compare branchy vs branchless assembly — see `jg`/`jl` (branch) vs `cmov` (branchless)

### 04 — Memory Allocators
- **Why `malloc`/`new` is slow on the hot path**:
  - Thread synchronization: global heap requires a lock (or thread-local caches with occasional sync)
  - Fragmentation: after many alloc/free cycles, memory is scattered — cache-unfriendly
  - Metadata overhead: malloc stores size + alignment info per allocation (16-32 bytes overhead)
  - For Java: every `new` allocates on the GC-managed heap → potential GC trigger → latency spike
- **Arena allocator (bump allocator)**:
  - Pre-allocate a large block. Allocation = bump a pointer forward. O(1), branchless, cache-friendly.
  - Deallocation = reset the pointer to the beginning. Free everything at once.
  - Perfect for: per-request allocation (allocate during request, free all at end), temporary computations
  - C++: `std::pmr::monotonic_buffer_resource` (C++17), or custom implementation (10 lines of code)
  - Rust: `bumpalo` crate — `Bump::new()`, `bump.alloc(value)`
  - Java: no equivalent. `ByteBuffer.allocateDirect()` is the closest but much more limited.
- **Pool allocator (slab allocator)**:
  - Pre-allocate N objects of the same size. Hand them out, return them. Zero fragmentation.
  - O(1) alloc and free (linked list of free slots, or bitmap)
  - Perfect for: fixed-size objects that are frequently created/destroyed (orders, messages, connections)
  - C++: `boost::pool`, or custom (freelist-based)
  - Rust: `typed-arena` crate, or custom pool
  - Java: object pooling pattern — `ArrayDeque<Order>` as a pool. Ugly but necessary for zero-alloc Java.
- **`jemalloc` vs `tcmalloc` vs `mimalloc` vs system malloc**:
  - `jemalloc` (Facebook, Redis, Firefox): thread-local arenas, low fragmentation, well-suited for long-running servers
  - `tcmalloc` (Google): thread-caching malloc, fast for small allocations, used in Google services
  - `mimalloc` (Microsoft): compact design, excellent for mixed workloads, often fastest in benchmarks
  - System malloc (glibc): decent default, but all three above beat it under contention
  - How to switch: `LD_PRELOAD=libjemalloc.so ./program` — no recompilation needed
- **Code**: Java `new` on hot path (measure GC pauses) vs C++ arena allocator vs Rust `bumpalo` — allocation throughput and latency
- **Measure**: `perf stat -e cache-misses` — arena allocations are contiguous → cache-friendly
- **Measure**: async-profiler allocation profiling for Java — flame graph of where allocations happen

### 05 — Hash Maps and Cache-Friendly Containers
- **Java `HashMap` — why it's slow**:
  - Chaining: each bucket is a linked list (or tree for >8 entries) — pointer chasing on every lookup
  - Autoboxing: `HashMap<Integer, Long>` stores `Integer` objects (16 bytes each) not `int` (4 bytes)
  - Object headers: every key and value has 12-16 bytes of overhead
  - Result: a lookup walks: array → bucket → Entry node → key object → value object — 3-4 cache misses minimum
- **Open addressing** (all entries in a flat array):
  - On collision, probe to the next slot (linear probing, quadratic probing, or Robin Hood)
  - All entries are in one contiguous array — cache-friendly, no pointer chasing
  - Lookup: compute hash → go to slot → if occupied, probe forward — typically 1-2 cache misses
- **Swiss table** (Google's revolutionary design):
  - `absl::flat_hash_map` (C++), Rust's default `std::collections::HashMap` (since 1.36)
  - Uses SIMD to check 16 slots in parallel during probing — `_mm_cmpeq_epi8` checks 16 control bytes at once
  - Control byte per slot: 7 bits of hash + 1 bit empty/full — fits in one SIMD register
  - Result: most lookups are a single SIMD probe — incredibly fast
- **Robin Hood hashing**: When inserting, if the new element is "poorer" (further from its ideal slot) than the existing element, swap them. Keeps probe distances short and uniform.
  - Facebook's `folly::F14FastMap` uses a Robin Hood variant
- **Alternatives for Java**:
  - Eclipse Collections: `IntIntHashMap`, `LongObjectHashMap` — no boxing, primitive-native
  - Koloboke: open-addressing hash maps for Java — much faster than `HashMap`
  - Agrona: `Int2IntHashMap` — used in Aeron and Disruptor ecosystem
- **Code**: Java HashMap (chaining + boxing) vs Eclipse Collections vs C++ abseil flat_hash_map vs Rust HashMap — lookup latency
- **Measure**: `perf stat -e L1-dcache-load-misses` — chaining shows 3-5x more cache misses than open addressing
- **Measure**: Throughput (ops/sec) at various load factors — open addressing degrades gracefully, chaining degrades sharply

## Key Reading for This Level

- Mike Acton — "Data-Oriented Design and C++" (CppCon 2014 talk — paradigm shift)
- Chandler Carruth — "Efficiency with Algorithms, Performance with Data Structures" (CppCon talk)
- Algorithmica HPC — "SIMD" and "Branchless" chapters (en.algorithmica.org/hpc/)
- Daniel Lemire — "Parsing Gigabytes of JSON per Second" (simdjson paper)
- Agner Fog — "Optimizing software in C++" (instruction tables, data structures, SIMD sections)
- Google — "Swiss Tables Design Notes" (abseil.io blog) — how absl::flat_hash_map works
- Matt Kulukundis — "Designing a Fast, Efficient, Cache-friendly Hash Table" (CppCon 2017 talk — Swiss table deep dive)
- Andrew Hunter — "Basics of futexes" and "tcmalloc" (Google design docs)

## Mini-Project: LRU Cache in 3 Languages (`06-mini-project-lru-cache/`)

**Apply everything from this level in one focused build.**

Implement an LRU (Least Recently Used) cache in Java, C++, and Rust. Same API, same capacity, same workload — measure how data structure choices dominate performance.

| Variant | Data structure | What it demonstrates |
|---------|---------------|---------------------|
| **Java `LinkedHashMap`** | Hash map + doubly-linked list (all heap objects, pointer chasing) | Baseline — standard library, GC-managed, autoboxing for primitive keys |
| **Java manual** | Open-addressing hash map + intrusive linked list with primitive arrays | How far you can push Java when you fight the object model |
| **C++ `std::unordered_map` + `std::list`** | Standard library (heap-allocated nodes) | Better than Java (no boxing) but still pointer-chasing in the list |
| **C++ flat** | `absl::flat_hash_map` + intrusive list with pool-allocated nodes | Cache-friendly: flat hash map + contiguous node pool. Minimal cache misses. |
| **Rust `std::collections::HashMap` + `VecDeque`** | Swiss table + deque (contiguous backing) | Already cache-friendly by default (Swiss table). Compare with linked-list approach. |
| **Rust custom** | Custom open-addressing map + arena-allocated nodes | Maximum control — benchmark against standard library |

**Workload**: Zipfian distribution (hot keys accessed frequently, long tail of cold keys) — realistic cache access pattern. Vary cache capacity to show hit rate vs latency tradeoff.

**What to measure**:
- `get()` latency: p50, p99 (this is where cache misses show up)
- `put()` latency: p50, p99 (eviction cost)
- `perf stat -e L1-dcache-load-misses` — pointer-chasing variants will have 3-5x more misses
- Memory footprint: Java's object headers + boxing vs C++/Rust flat structures
- Throughput under concurrent access (reader-writer lock vs lock-free)

**Produce**: Latency comparison table, cache miss rates, memory footprint per entry, flame graphs
