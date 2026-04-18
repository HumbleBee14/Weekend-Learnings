# Performance Engineering — Learning Plan

## The fast track (weekend warrior path)

If you only have limited weekends, do these in order. Each one builds on the previous.

| # | File | Time | What you get |
|---|------|------|-------------|
| 1 | `level-1/02-cpu-caches/cache_cpp.cpp` | 45 min | The single biggest perf factor — see cache misses destroy throughput |
| 2 | `level-1/02-cpu-caches/cache_java.java` | 30 min | Same logic in Java — see why ArrayList<Integer> is 10-50x slower than int[] |
| 3 | `level-2/01-benchmarking-methodology/good_benchmark_cpp.cpp` | 45 min | Learn to measure properly — Google Benchmark + HdrHistogram |
| 4 | `level-2/02-perf-and-hardware-counters/target_program.cpp` | 30 min | Use `perf stat` to see cache misses, branch misses, IPC in real numbers |
| 5 | `level-3/02-lock-free/spsc_queue_cpp.cpp` | 1 hr | Build a lock-free SPSC queue from scratch in C++ |
| 6 | `level-3/02-lock-free/spsc_queue_java.java` | 30 min | Same in Java — compare throughput, see GC pauses in the tail |
| 7 | `level-4/02-simd/simd_sum_cpp.cpp` | 1 hr | Your first SIMD code — 4-8x speedup on a simple sum |
| 8 | `level-7/cpp-optimized/` (start here) | 2-3 hrs | Build the order book matching engine — the capstone |

**After this track**: you understand caches, can benchmark properly, write lock-free code, use SIMD, and have built a production-style matching engine.

---

## The full curriculum

### Phase 1: Understand the Machine (Level 1)
*Before you can make things fast, understand why they're slow.*

| # | Directory | Focus | What you'll see |
|---|-----------|-------|-----------------|
| 1.1 | `01-cpu-pipeline-and-branch-prediction/` | Pipelines, branch prediction, speculative execution | sorted vs unsorted array: 5x difference from branch prediction alone |
| 1.2 | `02-cpu-caches-and-memory-hierarchy/` | L1/L2/L3 caches, cache lines, memory access patterns | ArrayList<Integer> (pointer chasing) vs int[] (contiguous): 10-50x gap |
| 1.3 | `03-false-sharing/` | MESI protocol, cache line contention between threads | Two threads on adjacent fields: 10x slowdown. Add padding: problem gone. |
| 1.4 | `04-numa-and-memory-topology/` | NUMA nodes, local vs remote memory, lstopo | Cross-node memory access: 2-3x slower than local |
| 1.5 | `05-tlb-and-huge-pages/` | TLB misses, 4KB vs 2MB pages, huge page config | Huge pages reduce TLB misses by 90%+ for large allocations |

**Java counter-examples in this phase**: Object headers eating cache lines, ArrayList pointer chasing, `@Contended` as the only false sharing fix, `-XX:+UseLargePages` JVM flag.

---

### Phase 2: Learn to Measure (Level 2)
*"If you can't measure it, you can't improve it." — Lord Kelvin (and every perf engineer)*

| # | Directory | Focus | What you'll learn |
|---|-----------|-------|-------------------|
| 2.1 | `01-benchmarking-methodology/` | Coordinated omission, warmup, dead code elimination | Why your `System.nanoTime()` benchmarks lie, and how to fix them |
| 2.2 | `02-perf-and-hardware-counters/` | `perf stat`, `perf record`, `perf report` | See cache-misses, branch-misses, IPC — the CPU's view of your code |
| 2.3 | `03-flame-graphs/` | Flame graph generation, on-CPU vs off-CPU | Find the hotspot in seconds instead of hours of guessing |
| 2.4 | `04-compiler-explorer/` | Read assembly, godbolt.org, optimization levels | See what -O2 vs -O3 actually does, when auto-vectorization kicks in |
| 2.5 | `05-java-specific-profiling/` | JFR, async-profiler, GC log analysis | Prove that GC is the bottleneck — GC pressure demo with allocation-heavy code |

**Tools mastered**: `perf`, flame graphs, Google Benchmark, Criterion, JMH, HdrHistogram, async-profiler, JFR, Compiler Explorer.

---

### Phase 3: Master Concurrency (Level 3)
*The hardest part of performance engineering. Where most people get it wrong.*

| # | Directory | Focus | What you'll build |
|---|-----------|-------|-------------------|
| 3.1 | `01-memory-ordering/` | acquire/release, seq_cst, x86-TSO vs ARM, JMM | See store reordering on ARM that x86 hides from you |
| 3.2 | `02-lock-free-data-structures/` | CAS, ABA problem, SPSC/MPSC queues | Lock-free SPSC queue in all 3 languages — benchmark vs mutex |
| 3.3 | `03-disruptor-pattern/` | LMAX ring buffer, sequence barriers, wait strategies | Disruptor vs ArrayBlockingQueue: 10-100x throughput difference |
| 3.4 | `04-atomics-deep-dive/` | fetch_add, compare_exchange, spinlocks | AtomicLong vs LongAdder vs synchronized — when each wins |
| 3.5 | `05-thread-pools-and-work-stealing/` | ForkJoinPool, work-stealing, thread-per-core | Rayon's work-stealing vs Java's ForkJoinPool vs C++ custom pool |

**Java counter-examples**: `synchronized` keyword overhead, `AtomicLong` contention, `ForkJoinPool` vs raw threads, `ArrayBlockingQueue` vs Disruptor (10-100x gap).

---

### Phase 4: Data-Oriented Design (Level 4)
*Stop fighting the hardware. Make it work for you.*

| # | Directory | Focus | What you'll see |
|---|-----------|-------|-----------------|
| 4.1 | `01-struct-of-arrays-vs-array-of-structs/` | SoA vs AoS, cache utilization, DOD | Same algorithm, different layout: 3-5x difference |
| 4.2 | `02-simd-intrinsics/` | SSE, AVX2, manual intrinsics, auto-vectorization | Scalar vs SIMD sum: 4-8x. Study simdjson: how to parse JSON at GB/s |
| 4.3 | `03-branchless-programming/` | cmov, select patterns, lookup tables | Remove branches from hot path: branch-misses drop to near zero |
| 4.4 | `04-memory-allocators/` | Arena, bump, pool, slab allocators, jemalloc vs tcmalloc | Java `new` (potential GC) vs arena allocator: 10-100x allocation speed |
| 4.5 | `05-hash-maps-and-cache-friendly-containers/` | Open addressing, Swiss table, Robin Hood hashing | Java HashMap (chaining + boxing) vs absl::flat_hash_map: 3-5x lookup |

**Java counter-examples**: `ArrayList<Object>` AoS layout, no SIMD control, `new` on hot path, `HashMap` with chaining and autoboxing.

---

### Phase 5: OS-Level Optimization (Level 5)
*When userspace optimization isn't enough, squeeze the kernel.*

| # | Directory | Focus | What you'll learn |
|---|-----------|-------|-------------------|
| 5.1 | `01-syscall-overhead/` | Syscall cost, vDSO, batching | `clock_gettime()` via vDSO: 50ns. Via syscall: 500ns. 10x from one flag. |
| 5.2 | `02-io-uring/` | SQ/CQ rings, submission batching, vs epoll | io_uring echo server vs epoll: 2-3x throughput at lower latency |
| 5.3 | `03-cpu-isolation-and-pinning/` | isolcpus, taskset, thread affinity, nohz_full | Scheduling jitter: 50μs without pinning → <1μs with pinning |
| 5.4 | `04-huge-pages-in-practice/` | THP vs explicit, hugetlbfs, mmap flags | TLB miss reduction with huge pages — measured with perf |
| 5.5 | `05-realtime-scheduling/` | SCHED_FIFO, mlockall(), RT kernel | CFS jitter histogram vs SCHED_FIFO: night and day |

**Java counter-examples**: JVM syscall overhead (JNI bridge), can't do io_uring from Java efficiently, Java-Thread-Affinity library as the only pinning option.

---

### Phase 6: Network-Level Optimization (Level 6)
*When the kernel's network stack is too slow, bypass it entirely.*

| # | Directory | Focus | What you'll build |
|---|-----------|-------|-------------------|
| 6.1 | `01-tcp-tuning/` | TCP_NODELAY, SO_BUSY_POLL, buffer sizes | Tuned TCP server — measure before/after Nagle's algorithm |
| 6.2 | `02-udp-multicast/` | Multicast groups, market data distribution | Multicast sender + receiver — how HFT distributes market data |
| 6.3 | `03-dpdk-kernel-bypass/` | Poll-mode drivers, hugepages for DMA | DPDK echo server — bypass the kernel entirely |
| 6.4 | `04-aeron-messaging/` | Aeron IPC/UDP, publications, subscriptions | Aeron vs TCP: sub-microsecond IPC messaging |
| 6.5 | `05-rdma-and-infiniband/` | RDMA concepts, verbs API, RoCE | Architecture overview — when to use RDMA |

---

### Phase 7: JVM Deep Dive (Level 7)
*Know your enemy. Understand exactly why Java is slow before building the fast alternatives.*

| # | Directory | Focus | What you'll see |
|---|-----------|-------|-----------------|
| 7.1 | `01-gc-internals-and-tuning/` | G1, ZGC, Shenandoah, Epsilon — how GC actually works | Same workload: measure pause times per GC. Even ZGC's sub-ms pauses kill HFT p999. |
| 7.2 | `02-jit-compilation-deep-dive/` | Tiered compilation, OSR, inlining, escape analysis | Cold vs warm: 10-100x difference. PrintCompilation + JITWatch visualization. |
| 7.3 | `03-off-heap-and-unsafe/` | ByteBuffer, Unsafe, MemorySegment (Panama) | On-heap vs off-heap benchmark. How Chronicle/Agrona dodge GC entirely. |
| 7.4 | `04-allocation-free-java/` | Object pooling, flyweight, SBE, Agrona | Allocation flame graph: before (millions of allocs) vs after (zero allocs on hot path). |
| 7.5 | `05-vector-api-and-panama-ffi/` | Java Vector API (SIMD), Panama FFI (native calls) | Java SIMD vs C++ intrinsics vs Rust std::simd — the gap in numbers. |

**After this level**: You can explain exactly what the JVM does under the hood and why each mechanism costs performance. This sets up levels 8-9 where you build the same thing in C++/Rust and show the difference.

---

### Phase 8: Build an HFT Order Book (Level 8) — CAPSTONE PROJECT 1
*Everything from levels 1-7, applied to a real matching engine.*

Build the same order book in three languages. Compare performance head-to-head.

| Component | Java (baseline) | C++ (optimized) | Rust (optimized) |
|-----------|----------------|-----------------|------------------|
| Order book | TreeMap (GC, boxing) | Flat arrays, intrusive lists | Custom allocator, no Box |
| Orders | Object per order (heap) | Packed struct (stack/pool) | repr(C), Copy trait |
| Parsing | String-based FIX | SBE zero-copy decode | SBE zero-copy decode |
| Market data | TCP, serialized objects | UDP multicast, kernel bypass | UDP multicast |
| Matching | Object-oriented, GC | Single-threaded, lock-free I/O | Single-threaded, lock-free I/O |

**Target**: Java ~50μs per order. C++/Rust <5μs per order. 10x gap, fully explained.

**What you produce**:
- Working matching engine in all 3 languages
- JMH / Google Benchmark / Criterion benchmarks
- Flame graphs for each implementation
- HdrHistogram latency percentile comparison
- Written analysis: where exactly Java loses and why

---

### Phase 9: Build an RTB Bidder (Level 9) — CAPSTONE PROJECT 2
*High-throughput, low-latency ad serving under production constraints.*

| Component | Java (baseline) | C++ (optimized) | Rust (optimized) |
|-----------|----------------|-----------------|------------------|
| HTTP handling | Spring MVC (object per request) | Custom HTTP parser, connection pool | Tokio + hyper async |
| User segments | HashMap<Long, Set<Integer>> (boxing) | Roaring Bitmaps, flat memory | Roaring Bitmaps via roaring crate |
| ML inference | — | ONNX Runtime, batched | ONNX Runtime via ort crate |
| Budget tracking | synchronized counter | Atomic, relaxed ordering | AtomicU64, Relaxed |

**Target**: Java ~30ms p99. C++/Rust <10ms p99 at 100K+ QPS.

---

### Phase 10: Production Performance Systems (Level 10)
*The textbooks teach you to make code fast. This level teaches you to make SYSTEMS fast.*

This is the biggest level — 11 topics covering everything that causes latency in production that isn't covered by hardware/OS optimization. Database internals, network failures, caching, serialization, distributed systems, observability, and real-world case studies.

| # | Directory | Focus |
|---|-----------|-------|
| 10.1 | `01-tail-latency-and-percentiles/` | p99/p999, fan-out amplification, hedged requests, HdrHistogram |
| 10.2 | `02-database-and-storage-performance/` | WAL, LSM vs B-Tree, SSD internals, compaction storms, connection pooling |
| 10.3 | `03-network-resilience-and-cascading-failures/` | Retry storms, circuit breakers, backpressure, load shedding, timeouts |
| 10.4 | `04-caching-strategies/` | Hit rate math, cache stampede, eviction policies, working set sizing |
| 10.5 | `05-serialization-and-data-transfer/` | JSON vs Protobuf vs SBE, zero-copy, payload optimization |
| 10.6 | `06-distributed-systems-latency/` | Fan-out amplification, cross-region, service mesh overhead, noisy neighbor |
| 10.7 | `07-high-throughput-transactions/` | Sharding, load balancing (P2C), queue theory, backpressure |
| 10.8 | `08-capacity-planning/` | Little's Law, USL, throughput vs latency tradeoff |
| 10.9 | `09-performance-observability/` | RED/USE/Golden Signals, distributed tracing, percentile alerting |
| 10.10 | `10-performance-regression-testing/` | CI/CD perf gates, benchmark baselines, statistical comparison |
| 10.11 | `11-case-studies/` | LMAX, HFT kernel bypass, ad exchange p99, retry storm post-mortem, compaction storm |

---

## Suggested weekend schedule

| Weekend | What to do | Hours |
|---------|-----------|-------|
| 1 | Level 1 (1.1-1.3) — caches, pipelines, false sharing | 3 hrs |
| 2 | Level 1 (1.4-1.5) + Level 2 (2.1-2.2) — NUMA, huge pages, benchmarking, perf | 4 hrs |
| 3 | Level 2 (2.3-2.5) — flame graphs, compiler explorer, Java profiling | 3 hrs |
| 4 | Level 3 (3.1-3.2) — memory ordering, build SPSC queue | 4 hrs |
| 5 | Level 3 (3.3-3.5) — Disruptor, atomics, thread pools | 4 hrs |
| 6 | Level 4 (4.1-4.3) — SoA/AoS, SIMD, branchless | 4 hrs |
| 7 | Level 4 (4.4-4.5) — allocators, hash maps | 3 hrs |
| 8 | Level 5 (5.1-5.3) — syscalls, io_uring, CPU pinning | 4 hrs |
| 9 | Level 5 (5.4-5.5) + Level 6 (6.1-6.2) — huge pages, RT, TCP, multicast | 4 hrs |
| 10 | Level 6 (6.3-6.5) — DPDK, Aeron, RDMA | 4 hrs |
| 11 | Level 7 (7.1-7.3) — GC internals, JIT deep dive, off-heap | 4 hrs |
| 12 | Level 7 (7.4-7.5) — allocation-free Java, Vector API, Panama | 3 hrs |
| 13 | Level 8 (Java baseline + C++ order book start) | 5 hrs |
| 14 | Level 8 (C++ finish + Rust order book) | 5 hrs |
| 15 | Level 8 (benchmarks, flame graphs, comparison writeup) | 4 hrs |
| 16 | Level 9 (RTB bidder: C++ + Rust + Java baseline) | 5 hrs |
| 17 | Level 9 (benchmarks + comparison) | 4 hrs |
| 18 | Level 10 (10.1-10.3) — tail latency, database/storage, network resilience | 4 hrs |
| 19 | Level 10 (10.4-10.6) — caching, serialization, distributed systems | 4 hrs |
| 20 | Level 10 (10.7-10.9) — high-throughput, capacity planning, observability | 4 hrs |
| 21 | Level 10 (10.10-10.11) — CI/CD perf gates, case studies | 3 hrs |

**21 weekends, ~84 hours total.** After this you can:
- Reason about performance from first principles (caches, pipelines, memory models)
- Measure properly with `perf`, flame graphs, JMH, Google Benchmark, Criterion
- Write lock-free concurrent code in C++, Rust, and Java
- Use SIMD, branchless techniques, custom allocators
- Tune the OS: io_uring, CPU isolation, huge pages, RT scheduling
- Do kernel bypass networking with DPDK and Aeron
- Explain exactly what the JVM does under the hood and why each mechanism costs performance
- Build a complete HFT matching engine and RTB bidder
- Explain exactly why C++/Rust outperforms Java in latency-critical paths with numbers
- Understand production latency: database I/O, cascading failures, caching, serialization overhead
- Design resilient systems: circuit breakers, retries with backoff, load shedding, backpressure
- Debug distributed latency: distributed tracing, percentile alerting, fan-out analysis
- Plan capacity: Little's Law, USL, queue theory

## Prerequisite knowledge

| You need | Why | Where to get it |
|----------|-----|-----------------|
| C++ basics (pointers, structs, templates) | Primary language for perf-critical code | Your `rusty-rust/` module covers Rust; pick up C++ from learncpp.com |
| Rust basics (ownership, borrowing, traits) | Second primary language | Your existing `rusty-rust/` module |
| Java intermediate (collections, threading) | Counter-example language or to learn how to optimize it | - |
| Linux command line | All profiling tools are Linux-native | WSL2 is sufficient for most levels |
| Basic computer architecture | Helps with level 1, but we learn it | CS:APP chapters 5-6 or MIT 6.172 first lectures |

## What to do next

1. **Read the Disruptor source code**: The single most educational codebase for perf engineering
2. **Profile your own code**: Take any project you've written and run `perf stat` on it — the numbers will surprise you
3. **Contribute**: Submit patches to JCTools, Crossbeam, simdjson, or any perf-critical library
5. **Go deeper**: Pick a specialization — FPGA acceleration, GPU compute (CUDA), or database internals (B-trees, LSM trees, buffer pool management)
