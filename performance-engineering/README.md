# Performance Engineering

Learning performance engineering from hardware fundamentals to production-scale systems — with C++ and Rust as the primary languages, and Java as the counter-example to show exactly where and why managed runtimes bleed performance.

## Why This Matters

In latency-critical systems, the difference between winning and losing is measured in microseconds:

- **HFT**: Tick-to-trade latency under 10 microseconds separates profitable firms from dead ones
- **Ad Serving**: RTB bidders must respond in <10ms p99 — miss the window, lose the auction
- **Gaming**: 16ms frame budget at 60fps — one GC pause and the frame drops
- **Databases**: A single cache miss costs ~100ns. A disk seek costs ~10ms. That's a 100,000x difference.

We rarely learn *why* th code is slow. We profile, find a hotspot, and cargo-cult a fix. Here I'll try to learn the physics of performance — from how CPUs actually execute instructions to how the kernel handles your packets — so you can reason about performance from first principles.

## The Approach: Java as Counter-Example

Every section follows the same pattern:

1. **Explain the concept** — what the hardware/OS actually does
2. **Show the Slower/Normal code** — that has the performance problem (GC pauses, allocation overhead, memory indirection, JIT warmup, boxing, false sharing through object headers, etc.)
3. **Show the C++/Rust code** — that solves it (deterministic memory, zero-cost abstractions, cache-friendly layouts, SIMD, kernel bypass)
4. **Benchmark both** — with proper methodology (JMH for Java, Google Benchmark for C++, Criterion for Rust) and show the numbers

Java trades **determinism for productivity**. The GC, JIT, and managed heap make developers enormously productive — but in latency-critical systems, unpredictability becomes the problem. A GC pause that's invisible in a web app is catastrophic in an HFT system. The goal isn't to bash Java — it's to understand exactly what the runtime does, why it costs performance, and what C++/Rust do differently.

### Where Java's Abstractions Cost Performance

| Problem | What happens | C++/Rust solution |
|---------|-------------|-------------------|
| **GC pauses** | Stop-the-world pauses (even ZGC has sub-ms pauses that kill p99) | No GC — deterministic deallocation (RAII, ownership) |
| **Object headers** | Every Java object has 12-16 bytes of header overhead | Plain structs, no headers, no vtable unless you ask |
| **Heap indirection** | `ArrayList<Order>` stores pointers to scattered heap objects | `std::vector<Order>` / `Vec<Order>` — contiguous memory |
| **Boxing** | `HashMap<Integer, Long>` boxes every key and value | `std::unordered_map<int, long>` — no boxing |
| **False sharing** | Object headers on same cache line, `@Contended` is the only fix | Control struct layout with padding, `alignas()` |
| **JIT warmup** | First 10K invocations run in interpreter, C2 compilation takes time | AOT compiled — fast from the first instruction |
| **No SIMD control** | Vector API is incubating, limited control | Full SIMD intrinsics (`_mm256_*`, `std::simd`) |
| **No kernel bypass** | Can't do DPDK/io_uring efficiently from JVM | Direct syscalls, mmap, DPDK native integration |
| **Allocation pressure** | Every `new` is a potential GC trigger on the hot path | Stack allocation, arena allocators, placement new |

## Languages Used

| Language | Role | Tools |
|----------|------|-------|
| **C++** (C++20/23) | Primary — where most HFT/ad-tech/game engine code lives | GCC/Clang, Google Benchmark, perf, Valgrind |
| **Rust** | Primary — modern alternative with safety guarantees + zero-cost abstractions | Cargo, Criterion, flamegraph-rs |
| **Java** (21+) | Counter-example — showing performance pitfalls of managed runtimes | JMH, async-profiler, JFR, VisualVM |
| **C** | Kernel/OS level — DPDK, io_uring, syscalls | GCC, perf, strace, bpftrace |

## Tools Covered

### Profiling & Measurement (9 tools)

| Tool | What it does | Level |
|------|-------------|-------|
| **`perf`** | Linux hardware performance counters — the primary profiling tool | `level-2` |
| **Flame Graphs** | Visualization of profiler output — see where time goes | `level-2` |
| **Intel VTune** | Best-in-class x86 microarchitecture analysis | `level-2` |
| **`perf c2c`** | Detect false sharing (cache-to-cache transfers) | `level-2` |
| **Google Benchmark** | C++ microbenchmark framework | `level-2` |
| **Criterion** | Rust statistical benchmarking | `level-2` |
| **JMH** | Java microbenchmark harness (handles JIT warmup, dead code) | `level-2` |
| **async-profiler** | Low-overhead JVM sampling profiler + flame graphs | `level-2` |
| **HdrHistogram** | Latency recording without coordinated omission | `level-2` |

### System Observability (7 tools)

| Tool | What it does | Level |
|------|-------------|-------|
| **`bpftrace`** | eBPF-based dynamic tracing | `level-5` |
| **`strace`** | Syscall tracing | `level-5` |
| **`numactl` / `numastat`** | NUMA topology and statistics | `level-5` |
| **`lstopo` (hwloc)** | Visualize hardware topology | `level-1` |
| **`rdtsc`** | CPU timestamp counter for nanosecond timing | `level-2` |
| **JFR + JMC** | Java Flight Recorder + Mission Control | `level-2` |
| **Compiler Explorer (godbolt.org)** | See what assembly your code compiles to | `level-1` |

### Networking & I/O (4 tools/libraries)

| Tool | What it does | Level |
|------|-------------|-------|
| **DPDK** | Kernel bypass networking — poll-mode drivers | `level-6` |
| **`liburing` / io_uring** | Linux async I/O — batched syscalls, zero-copy | `level-5` |
| **Aeron** | Reliable UDP messaging + shared memory IPC | `level-6` |
| **`tcpdump` / Wireshark** | Packet capture and analysis | `level-6` |

### Libraries Studied (12 libraries)

| Library | Language | What it does | Level |
|---------|----------|-------------|-------|
| **LMAX Disruptor** | Java | Lock-free ring buffer — the gold standard | `level-3` |
| **JCTools** | Java | Lock-free queues (SPSC/MPSC) — used inside Netty | `level-3` |
| **Agrona** | Java | Off-heap buffers, lock-free collections | `level-3` |
| **folly** | C++ | Facebook's concurrent data structures, allocators | `level-4` |
| **abseil** | C++ | Google's Swiss table hash map, synchronization | `level-4` |
| **crossbeam** | Rust | Lock-free structures, epoch-based reclamation | `level-3` |
| **SBE** | Multi | Zero-copy serialization — used by CME, LSE | `level-8` |
| **Chronicle Queue** | Java | Microsecond-level persisted queue | `level-8` |
| **simdjson** | C++ | SIMD JSON parsing at GB/s speeds | `level-4` |
| **Roaring Bitmaps** | Multi | Compressed bitmaps — used across ad tech | `level-9` |
| **jemalloc** | C | High-performance memory allocator (Facebook, Redis) | `level-4` |
| **mimalloc** | C | Microsoft's compact allocator — faster than jemalloc in some workloads | `level-4` |

## Setup

### C++ (Linux / WSL2)

```bash
# Compiler + build tools
sudo apt install build-essential cmake ninja-build

# Profiling tools
sudo apt install linux-tools-common linux-tools-generic
sudo apt install valgrind

# Google Benchmark
git clone https://github.com/google/benchmark.git
cd benchmark && cmake -E make_directory build
cd build && cmake .. -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
sudo cmake --install .

# hwloc (hardware topology)
sudo apt install hwloc

# io_uring
sudo apt install liburing-dev

# DPDK (level 6+)
sudo apt install dpdk dpdk-dev
```

### Rust

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Benchmarking
cargo install cargo-criterion

# Flamegraph
cargo install flamegraph
```

### Java (Counter-Example)

```bash
# Java 21+ (for Vector API, Panama FFI)
sudo apt install openjdk-21-jdk

# JMH (via Maven/Gradle in each project)
# async-profiler
wget https://github.com/async-profiler/async-profiler/releases/latest/download/async-profiler-linux-x64.tar.gz
tar xzf async-profiler-linux-x64.tar.gz
```

### All Platforms

```bash
# Compiler Explorer — use in browser
# https://godbolt.org

# HdrHistogram (available in all 3 languages)
# C++: vcpkg install hdrhistogram
# Rust: cargo add hdrhistogram
# Java: Maven dependency org.hdrhistogram:HdrHistogram
```

> **Note**: Most performance work requires **Linux** (perf, io_uring, DPDK, CPU isolation). Use WSL2 on Windows, or a dedicated Linux VM/machine. macOS works for C++/Rust development but lacks `perf` and kernel tuning capabilities.

### Reproducible Benchmarks (Docker, optional)

For consistent results across machines, use Docker with pinned compiler versions:

```bash
# Build a benchmark container
docker build -t perf-bench .

# Run with privileged (needed for perf, huge pages) and pinned CPUs
docker run --privileged --cpuset-cpus=4-7 perf-bench ./run_benchmarks.sh
```

Every experiment in this repo follows the methodology defined in [`experiment-template.md`](experiment-template.md) — hypothesis, setup, metrics, graphs, conclusions.

## Structure

Levels 1-6 are **concept-driven** — each lesson has a Java counter-example + C++/Rust implementation + benchmarks.
Level 7 is the **JVM deep dive** — understand exactly why Java is slow, the internals, and the workarounds.
Levels 8-9 are **project-driven** — build complete systems from scratch.
Level 10 is **production wisdom** — case studies and real-world patterns.

```
performance-engineering/
│
├── level-1-hardware-fundamentals/                    # Understand the machine
│   ├── 01-cpu-pipeline-and-branch-prediction/
│   │   ├── README.md                                ← what pipelines are, why branches kill perf
│   │   ├── branch_java.java                         ← sorted vs unsorted array — classic demo
│   │   ├── branch_cpp.cpp                           ← same demo + __builtin_expect, branchless
│   │   ├── branch_rust.rs                           ← likely/unlikely, branchless with select
│   │   └── benchmark_results.md                     ← perf stat output, branch miss rates
│   │
│   ├── 02-cpu-caches-and-memory-hierarchy/
│   │   ├── README.md                                ← L1/L2/L3, cache lines (64B), associativity
│   │   ├── cache_java.java                          ← ArrayList<Integer> vs int[] traversal
│   │   ├── cache_cpp.cpp                            ← array vs linked list, stride patterns
│   │   ├── cache_rust.rs                            ← Vec<T> vs LinkedList<T>, cache effects
│   │   └── benchmark_results.md                     ← perf stat: cache-misses, cache-references
│   │
│   ├── 03-false-sharing/
│   │   ├── README.md                                ← MESI protocol, cache line contention
│   │   ├── false_sharing_java.java                  ← two threads, adjacent fields, @Contended fix
│   │   ├── false_sharing_cpp.cpp                    ← alignas(64) padding, perf c2c demo
│   │   ├── false_sharing_rust.rs                    ← CachePadded<T> from crossbeam
│   │   └── benchmark_results.md                     ← perf c2c output, before/after
│   │
│   ├── 04-numa-and-memory-topology/
│   │   ├── README.md                                ← NUMA nodes, local vs remote memory, lstopo
│   │   ├── numa_demo_cpp.cpp                        ← numactl allocation, cross-node penalty
│   │   └── topology_output.md                       ← lstopo output, hwloc visualization
│   │
│   └── 05-tlb-and-huge-pages/
│       ├── README.md                                ← TLB misses, 4KB vs 2MB vs 1GB pages
│       ├── hugepages_cpp.cpp                        ← mmap MAP_HUGETLB, TLB miss reduction
│       ├── hugepages_java.java                      ← -XX:+UseLargePages, JVM transparent huge pages
│       └── benchmark_results.md                     ← perf stat: dTLB-load-misses before/after
│
├── level-2-measurement-and-profiling/                # You can't optimize what you can't measure
│   ├── 01-benchmarking-methodology/
│   │   ├── README.md                                ← Gil Tene's "How NOT to Measure Latency", coordinated omission
│   │   ├── bad_benchmark_java.java                  ← all the mistakes: no warmup, dead code, System.nanoTime()
│   │   ├── good_benchmark_java.java                 ← JMH: @Benchmark, @Warmup, @State, Blackhole
│   │   ├── good_benchmark_cpp.cpp                   ← Google Benchmark: DoNotOptimize, proper setup
│   │   ├── good_benchmark_rust.rs                   ← Criterion: black_box, statistical analysis
│   │   └── hdr_histogram_demo.cpp                   ← HdrHistogram: percentiles without coordinated omission
│   │
│   ├── 02-perf-and-hardware-counters/
│   │   ├── README.md                                ← perf stat, perf record, perf report
│   │   ├── target_program.cpp                       ← deliberately cache-unfriendly code to profile
│   │   └── perf_commands.md                         ← cheat sheet: perf stat -e cache-misses,branches,...
│   │
│   ├── 03-flame-graphs/
│   │   ├── README.md                                ← reading flame graphs, on-CPU vs off-CPU
│   │   ├── hot_path_cpp.cpp                         ← program with clear hotspot to visualize
│   │   ├── hot_path_java.java                       ← same program, async-profiler → flame graph
│   │   └── generate_flamegraph.sh                   ← perf record → perf script → flamegraph.pl
│   │
│   ├── 04-compiler-explorer/
│   │   ├── README.md                                ← reading assembly, godbolt.org workflow
│   │   ├── optimization_levels.cpp                  ← same code at -O0, -O1, -O2, -O3 — diff the assembly
│   │   └── auto_vectorization.cpp                   ← when the compiler SIMD-ifies your loop (and when it doesn't)
│   │
│   └── 05-java-specific-profiling/
│       ├── README.md                                ← JFR, JMC, async-profiler, GC log analysis
│       ├── gc_pressure_demo.java                    ← allocation-heavy code → GC pauses
│       ├── jit_warmup_demo.java                     ← cold vs warm performance, -XX:+PrintCompilation
│       └── escape_analysis_demo.java                ← when JVM avoids heap allocation (and when it fails)
│
├── level-3-memory-models-and-concurrency/            # The hardest part of performance engineering
│   ├── 01-memory-ordering/
│   │   ├── README.md                                ← happens-before, acquire/release, seq_cst, x86-TSO vs ARM
│   │   ├── ordering_java.java                       ← volatile, synchronized, VarHandle — JMM guarantees
│   │   ├── ordering_cpp.cpp                         ← std::atomic, memory_order_*, compiler/CPU reordering demo
│   │   └── ordering_rust.rs                         ← std::sync::atomic, Ordering enum, unsafe for raw control
│   │
│   ├── 02-lock-free-data-structures/
│   │   ├── README.md                                ← CAS, ABA problem, hazard pointers, epoch reclamation
│   │   ├── spsc_queue_java.java                     ← lock-free SPSC ring buffer from scratch
│   │   ├── spsc_queue_cpp.cpp                       ← same in C++ with std::atomic
│   │   ├── spsc_queue_rust.rs                       ← same in Rust with crossbeam
│   │   └── benchmark_results.md                     ← throughput: lock-free vs mutex-based vs channel
│   │
│   ├── 03-disruptor-pattern/
│   │   ├── README.md                                ← LMAX Disruptor: ring buffer, sequence barriers, wait strategies
│   │   ├── disruptor_study.java                     ← use the real Disruptor library, benchmark vs ArrayBlockingQueue
│   │   ├── ring_buffer_cpp.cpp                      ← implement the core pattern in C++
│   │   └── benchmark_results.md                     ← Disruptor vs queue: latency percentiles
│   │
│   ├── 04-atomics-deep-dive/
│   │   ├── README.md                                ← fetch_add, compare_exchange, atomic flags, spinlocks
│   │   ├── counter_java.java                        ← AtomicLong vs LongAdder vs synchronized — which wins when
│   │   ├── counter_cpp.cpp                          ← std::atomic<int64_t>, relaxed vs seq_cst, XADD on x86
│   │   └── counter_rust.rs                          ← AtomicU64, fetch_add, Ordering choices
│   │
│   └── 05-thread-pools-and-work-stealing/
│       ├── README.md                                ← work-stealing, ForkJoinPool internals, thread-per-core
│       ├── threadpool_java.java                     ← ForkJoinPool vs FixedThreadPool vs Virtual Threads
│       ├── threadpool_cpp.cpp                       ← custom work-stealing pool, thread pinning
│       └── threadpool_rust.rs                       ← Rayon's work-stealing, tokio runtime internals
│
├── level-4-data-oriented-design/                     # Make the hardware work FOR you
│   ├── 01-struct-of-arrays-vs-array-of-structs/
│   │   ├── README.md                                ← SoA vs AoS, cache utilization, data-oriented design
│   │   ├── aos_vs_soa_java.java                     ← Object[] (AoS by default) vs parallel primitive arrays
│   │   ├── aos_vs_soa_cpp.cpp                       ← struct of vectors vs vector of structs
│   │   ├── aos_vs_soa_rust.rs                       ← same pattern in Rust
│   │   └── benchmark_results.md                     ← cache miss rates: AoS vs SoA
│   │
│   ├── 02-simd-intrinsics/
│   │   ├── README.md                                ← SSE, AVX2, AVX-512, auto-vectorization, manual intrinsics
│   │   ├── simd_sum_cpp.cpp                         ← scalar vs SSE vs AVX2 sum — _mm256_add_ps
│   │   ├── simd_sum_rust.rs                         ← std::simd (nightly) or packed_simd
│   │   ├── simd_java.java                           ← Vector API (incubating) — Java's attempt at SIMD
│   │   └── simdjson_study.cpp                       ← study simdjson: how to parse JSON at GB/s
│   │
│   ├── 03-branchless-programming/
│   │   ├── README.md                                ← conditional moves (cmov), select patterns, lookup tables
│   │   ├── branchless_cpp.cpp                       ← min/max/abs/clamp without branches
│   │   ├── branchless_rust.rs                       ← same patterns in Rust
│   │   └── benchmark_results.md                     ← branch-misses before/after
│   │
│   ├── 04-memory-allocators/
│   │   ├── README.md                                ← malloc internals, arena/bump/pool/slab allocators
│   │   ├── allocator_java.java                      ← object pooling, off-heap ByteBuffer, Unsafe — the workarounds
│   │   ├── arena_allocator_cpp.cpp                  ← custom arena allocator, placement new
│   │   ├── bump_allocator_rust.rs                   ← bumpalo crate, typed-arena
│   │   ├── jemalloc_vs_tcmalloc.cpp                 ← benchmark different allocators under contention
│   │   └── benchmark_results.md                     ← allocation throughput: new/delete vs arena vs pool
│   │
│   └── 05-hash-maps-and-cache-friendly-containers/
│       ├── README.md                                ← open addressing, Robin Hood, Swiss table, flat maps
│       ├── hashmap_java.java                        ← HashMap (chaining + boxing) vs Koloboke/Eclipse Collections
│       ├── hashmap_cpp.cpp                          ← std::unordered_map vs absl::flat_hash_map vs folly::F14
│       ├── hashmap_rust.rs                          ← std HashMap (SwissTable-based) vs custom
│       └── benchmark_results.md                     ← lookup latency: chaining vs open addressing
│
├── level-5-os-and-kernel-tuning/                     # Squeeze every drop from the OS
│   ├── 01-syscall-overhead/
│   │   ├── README.md                                ← syscall cost, vDSO, batching, strace
│   │   ├── syscall_cost.cpp                         ← measure getpid(), clock_gettime() via vDSO vs syscall
│   │   └── strace_analysis.md                       ← strace a program, count syscalls, find the bottleneck
│   │
│   ├── 02-io-uring/
│   │   ├── README.md                                ← SQ/CQ rings, submission batching, zero-copy, vs epoll
│   │   ├── echo_server_epoll.cpp                    ← traditional epoll echo server
│   │   ├── echo_server_iouring.cpp                  ← same server with io_uring — compare
│   │   ├── echo_server_rust.rs                      ← tokio-uring or io-uring crate
│   │   └── benchmark_results.md                     ← throughput + latency: epoll vs io_uring
│   │
│   ├── 03-cpu-isolation-and-pinning/
│   │   ├── README.md                                ← isolcpus, nohz_full, rcu_nocbs, taskset, thread affinity
│   │   ├── pinning_cpp.cpp                          ← pthread_setaffinity_np, measure jitter before/after
│   │   ├── pinning_java.java                        ← OpenHFT Java-Thread-Affinity library
│   │   └── kernel_params.md                         ← grub config: isolcpus=4-7 nohz_full=4-7 rcu_nocbs=4-7
│   │
│   ├── 04-huge-pages-in-practice/
│   │   ├── README.md                                ← THP vs explicit huge pages, when each helps/hurts
│   │   ├── hugepage_mmap.cpp                        ← MAP_HUGETLB, measure TLB miss reduction
│   │   └── hugepage_config.md                       ← /proc/sys/vm/nr_hugepages, hugetlbfs mount
│   │
│   └── 05-realtime-scheduling/
│       ├── README.md                                ← SCHED_FIFO, SCHED_RR, mlockall(), RT kernel patches
│       ├── realtime_thread.cpp                       ← SCHED_FIFO thread with mlockall, measure scheduling jitter
│       └── jitter_comparison.md                     ← CFS vs FIFO jitter histograms
│
├── level-6-networking-and-kernel-bypass/             # When the kernel is too slow
│   ├── 01-tcp-tuning/
│   │   ├── README.md                                ← TCP_NODELAY, SO_BUSY_POLL, buffer sizes, congestion control
│   │   ├── tcp_server_cpp.cpp                       ← echo server with tuned sockets
│   │   ├── tcp_server_rust.rs                       ← same in Rust with socket2 crate
│   │   └── tuning_checklist.md                      ← sysctl settings for low-latency networking
│   │
│   ├── 02-udp-multicast/
│   │   ├── README.md                                ← why HFT uses multicast for market data, join/leave groups
│   │   ├── multicast_sender.cpp                     ← send market data updates
│   │   ├── multicast_receiver.cpp                   ← receive and process with minimal latency
│   │   └── multicast_receiver_rust.rs               ← same in Rust
│   │
│   ├── 03-dpdk-kernel-bypass/
│   │   ├── README.md                                ← poll-mode drivers, hugepages for DMA, EAL init
│   │   ├── dpdk_echo.c                              ← simple DPDK packet echo — bypass the kernel entirely
│   │   └── dpdk_setup.md                            ← binding NICs, hugepage config, EAL parameters
│   │
│   ├── 04-aeron-messaging/
│   │   ├── README.md                                ← Aeron architecture: media driver, publications, subscriptions
│   │   ├── aeron_publisher.java                     ← send messages via Aeron IPC/UDP
│   │   ├── aeron_subscriber.java                    ← receive with busy-spin, measure latency
│   │   └── benchmark_results.md                     ← Aeron vs TCP vs raw UDP latency
│   │
│   └── 05-rdma-and-infiniband/
│       ├── README.md                                ← RDMA concepts, verbs API, RoCE vs InfiniBand
│       └── rdma_overview.md                         ← architecture diagram, when to use, latency numbers
│
├── level-7-jvm-performance/                          # Know your enemy — why Java is slow, and the workarounds
│   ├── 01-gc-internals-and-tuning/
│   │   ├── gc_comparison.java                       ← same workload under G1, ZGC, Shenandoah, Epsilon
│   │   └── gc_analysis.md                           ← pause times, throughput, tail latency per GC
│   │
│   ├── 02-jit-compilation-deep-dive/
│   │   ├── jit_warmup.java                          ← cold vs warm, -XX:+PrintCompilation, JITWatch
│   │   └── jit_analysis.md                          ← C1 vs C2, OSR, inlining decisions
│   │
│   ├── 03-off-heap-and-unsafe/
│   │   ├── off_heap_demo.java                       ← ByteBuffer.allocateDirect, Unsafe, MemorySegment
│   │   └── off_heap_analysis.md                     ← on-heap vs off-heap allocation benchmark
│   │
│   ├── 04-allocation-free-java/
│   │   ├── allocation_free.java                     ← object pooling, flyweight, SBE, Agrona collections
│   │   └── allocation_analysis.md                   ← async-profiler alloc flame graph: before/after
│   │
│   └── 05-vector-api-and-panama-ffi/
│       ├── vector_api_demo.java                     ← SIMD from Java — FloatVector, benchmark vs scalar
│       ├── panama_ffi_demo.java                     ← call native C from Java without JNI
│       └── simd_comparison.md                       ← Java Vector API vs C++ intrinsics vs Rust std::simd
│
├── level-8-project-order-book/                       # BUILD: HFT matching engine
│   ├── README.md                                    ← project spec: price-time priority, FIX/SBE, sub-μs target
│   ├── docs/
│   │   ├── architecture.md                          ← component diagram, data flow, hot path identification
│   │   ├── protocol_spec.md                         ← SBE message schemas, FIX-inspired order types
│   │   └── latency_budget.md                        ← nanosecond budget per component
│   │
│   ├── java-baseline/                               ← The "before" — idiomatic Java implementation
│   │   ├── pom.xml
│   │   ├── src/main/java/orderbook/
│   │   │   ├── OrderBook.java                       ← TreeMap-based, object-per-order, GC-heavy
│   │   │   ├── Order.java
│   │   │   ├── MatchingEngine.java
│   │   │   ├── FIXParser.java                       ← string-based FIX parsing
│   │   │   └── MarketDataPublisher.java
│   │   └── src/test/java/orderbook/
│   │       └── OrderBookBenchmark.java              ← JMH benchmarks
│   │
│   ├── cpp-optimized/                               ← The "after" — every technique from levels 1-7
│   │   ├── CMakeLists.txt
│   │   ├── include/
│   │   │   ├── order_book.hpp                       ← flat array price levels, intrusive lists
│   │   │   ├── order.hpp                            ← packed struct, no heap allocation
│   │   │   ├── matching_engine.hpp                  ← lock-free, single-threaded hot path
│   │   │   ├── sbe_codec.hpp                        ← SBE encode/decode — zero-copy
│   │   │   └── market_data_publisher.hpp            ← UDP multicast, kernel bypass ready
│   │   ├── src/
│   │   │   ├── main.cpp
│   │   │   ├── order_book.cpp
│   │   │   └── matching_engine.cpp
│   │   ├── bench/
│   │   │   └── order_book_bench.cpp                 ← Google Benchmark + HdrHistogram
│   │   └── test/
│   │       └── order_book_test.cpp
│   │
│   ├── rust-optimized/                              ← Rust alternative — safety + performance
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── main.rs
│   │   │   ├── order_book.rs                        ← custom allocator, no Box in hot path
│   │   │   ├── order.rs                             ← repr(C), packed, Copy
│   │   │   ├── matching_engine.rs
│   │   │   ├── sbe_codec.rs
│   │   │   └── market_data.rs
│   │   └── benches/
│   │       └── order_book_bench.rs                  ← Criterion benchmarks
│   │
│   └── results/
│       ├── java_vs_cpp_vs_rust.md                   ← head-to-head comparison with numbers
│       ├── flamegraphs/                             ← flame graph SVGs for each implementation
│       └── hdr_histograms/                          ← latency percentile data
│
├── level-9-project-rtb-bidder/                       # BUILD: Real-Time Bidding ad serving engine
│   ├── README.md                                    ← project spec: <10ms p99, millions QPS target
│   ├── docs/
│   │   ├── architecture.md                          ← bid request flow, feature lookup, ML inference
│   │   └── latency_budget.md                        ← per-component time budget
│   │
│   ├── cpp-bidder/                                  ← Primary implementation
│   │   ├── CMakeLists.txt
│   │   ├── include/
│   │   │   ├── bid_handler.hpp                      ← HTTP handler, parse bid request, respond
│   │   │   ├── user_store.hpp                       ← Roaring Bitmap audience segments
│   │   │   ├── feature_store.hpp                    ← in-memory feature lookup
│   │   │   ├── ml_scorer.hpp                        ← ONNX Runtime inference for bid price
│   │   │   └── budget_pacer.hpp                     ← atomic budget tracking
│   │   ├── src/
│   │   │   ├── main.cpp
│   │   │   ├── bid_handler.cpp
│   │   │   └── user_store.cpp
│   │   └── bench/
│   │       └── bidder_bench.cpp
│   │
│   ├── rust-bidder/                                 ← Rust alternative — tokio async + zero-copy
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── main.rs                              ← Tokio async HTTP server
│   │   │   ├── bid_handler.rs                       ← Request parsing, bid logic
│   │   │   ├── user_store.rs                        ← Roaring Bitmaps via roaring crate
│   │   │   ├── ml_scorer.rs                         ← ONNX Runtime via ort crate
│   │   │   └── budget_pacer.rs                      ← AtomicU64 budget tracking
│   │   └── benches/
│   │       └── bidder_bench.rs                      ← Criterion benchmarks
│   │
│   ├── java-baseline/                               ← Counter-example: same logic, Spring-style
│   │   ├── pom.xml
│   │   └── src/main/java/bidder/
│   │       ├── BidController.java                   ← Spring MVC handler — object allocation per request
│   │       ├── UserStore.java                       ← HashMap<Long, Set<Integer>> — boxing, GC pressure
│   │       └── BidderBenchmark.java                 ← JMH
│   │
│   └── results/
│       ├── throughput_comparison.md                  ← QPS at various latency percentiles
│       └── flamegraphs/
│
└── level-10-production-systems/                      # What the books don't teach you
    ├── 01-tail-latency/
    │   ├── tail_latency_demo.cpp                    ← simulate: one slow component poisons everything
    │   ├── hedged_requests.cpp                      ← hedging, backup requests, tail-at-scale
    │   └── tail_latency_analysis.md                 ← p99 vs p999 vs max, why averages lie
    │
    ├── 02-capacity-planning/
    │   ├── capacity_model.py                        ← simple capacity model with USL (Python for plotting)
    │   └── capacity_planning.md                     ← Little's Law, USL, throughput vs latency tradeoff
    │
    ├── 03-performance-regression-testing/
    │   ├── ci_benchmark.yml                         ← GitHub Action: run benchmarks, fail on regression
    │   ├── baseline_comparison.py                   ← compare against stored baselines
    │   └── perf_ci.md                               ← CI/CD perf gates, benchmark baselines, alerting
    │
    └── 04-case-studies/
        ├── lmax_disruptor_case_study.md             ← how LMAX got 6M TPS on a single thread
        ├── hft_kernel_bypass_case_study.md          ← from 50μs to 5μs with OpenOnload
        └── ad_tech_p99_case_study.md                ← how an ad exchange cut p99 from 40ms to 8ms
```

## Hardware / GPU Requirements

| Level | Hardware | Notes |
|---|---|---|
| 1-2 | Any Linux (WSL2 ok) | `perf` needs Linux, godbolt works everywhere |
| 3 | Multi-core CPU | Concurrency demos need 4+ cores |
| 4 | Modern x86 CPU | SIMD needs SSE4/AVX2 support |
| 5 | Linux (bare metal preferred) | Kernel tuning, io_uring, huge pages |
| 6 | Linux + 2nd NIC (DPDK) | DPDK requires dedicated NIC; Aeron works anywhere |
| 7 | Any Linux (WSL2 ok) | JVM tuning, profiling — Java 21+ required |
| 8-9 | Linux, 8+ cores, 16GB+ RAM | Project builds need resources for benchmarking |
| 10 | Any | Conceptual + CI/CD config |

## Key Resources

### The One Course

- [MIT 6.172: Performance Engineering of Software Systems](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/) — Charles Leiserson. Full lectures free. **THE** structured course on this topic.

### Books (Priority Order)

| # | Book | Author | Why |
|---|------|--------|-----|
| 1 | **Systems Performance** (2nd ed) | Brendan Gregg | The bible. CPU, memory, I/O, networking, observability methodology. |
| 2 | **Algorithms for Modern Hardware** | Sergey Slotin | Free at [algorithmica.org/hpc](https://en.algorithmica.org/hpc/). SIMD, caches, branchless. Hands-on gold. |
| 3 | **What Every Programmer Should Know About Memory** | Ulrich Drepper | Free PDF. The memory hierarchy deep dive every systems programmer needs. |
| 4 | **The Art of Multiprocessor Programming** | Herlihy & Shavit | Lock-free everything. The concurrency bible. |
| 5 | **Computer Systems: A Programmer's Perspective** (CS:APP) | Bryant & O'Hallaron | How programs actually execute on hardware. Chapter 5 (Optimizing Performance) and 6 (Memory Hierarchy) are essential. |
| 6 | **Performance Analysis and Tuning on Modern CPUs** | Denis Bakhvalov | Free at [easyperf.net/perf_book](https://book.easyperf.net/perf_book). Modern CPU microarchitecture analysis. |
| 7 | **C++ Concurrency in Action** | Anthony Williams | C++ memory model, atomics, lock-free. |
| 8 | **Is Parallel Programming Hard?** | Paul McKenney | Free [perfbook](https://mirrors.edge.kernel.org/pub/linux/kernel/people/paulmck/perfbook/perfbook.html). Linux kernel concurrency, RCU. |
| 9 | **Building Low Latency Applications with C++** | Sourav Ghosh | HFT-focused C++ book. |
| 10 | **BPF Performance Tools** | Brendan Gregg | Advanced Linux observability with eBPF. |
| 11 | **Computer Architecture: A Quantitative Approach** | Hennessy & Patterson | The hardware reference — pipelines, caches, branch prediction at the architecture level. |
| 12 | **Release It!** (2nd ed) | Michael Nygard | Circuit breakers, timeouts, cascading failures, bulkheads — production resilience patterns. |
| 13 | **Designing Data-Intensive Applications** (DDIA) | Martin Kleppmann | Storage engines, replication, partitioning, consistency — how databases actually work. |
| 14 | **Database Internals** | Alex Petrov | B-Trees, LSM trees, WAL, compaction, distributed storage — deep database mechanics. |

### Must-Watch Talks

| Talk | Speaker | Why |
|------|---------|-----|
| **How NOT to Measure Latency** | Gil Tene | Coordinated omission, HdrHistogram. **Every perf engineer must watch this.** |
| **Mechanical Sympathy** (multiple) | Martin Thompson | Low-latency Java, Disruptor design, writing code that works with hardware. |
| **Linux Performance Tools** | Brendan Gregg | perf, BPF, flame graphs — the observability toolkit. |
| **Atomic Weapons: The C++ Memory Model** | Herb Sutter | C++ atomics explained properly. Two-part talk. |
| **CPU Caches and Why You Care** | Scott Meyers | Cache effects on performance — accessible and visual. |
| **Efficiency with Algorithms, Performance with Data Structures** | Chandler Carruth | Data-oriented design at CppCon. |
| **Data-Oriented Design and C++** | Mike Acton | CppCon 2014. The paradigm shift talk. Changes how you think about code. |

### Blogs & Online References

| Resource | Author | Focus |
|----------|--------|-------|
| [brendangregg.com](https://www.brendangregg.com/) | Brendan Gregg | Performance methodology, flame graphs, observability |
| [mechanical-sympathy.blogspot.com](https://mechanical-sympathy.blogspot.com/) | Martin Thompson | Mechanical sympathy, Disruptor, low-latency Java |
| [agner.org/optimize](https://agner.org/optimize/) | Agner Fog | CPU microarchitecture tables, instruction latencies, optimization manuals |
| [lemire.me/blog](https://lemire.me/blog/) | Daniel Lemire | SIMD, branchless programming, data processing |
| [travisdowns.github.io](https://travisdowns.github.io/) | Travis Downs | Deep CPU microarchitecture analysis |
| [johnnysswlab.com](https://johnnysswlab.com/) | Johnny's Software Lab | Practical performance optimization tutorials |
| [godbolt.org](https://godbolt.org/) | Matt Godbolt | Compiler Explorer — see your code's assembly |
| [algorithmica.org/hpc](https://en.algorithmica.org/hpc/) | Sergey Slotin | Free book on algorithms for modern hardware |
| [aerospike.com/blog](https://aerospike.com/blog/what-is-p99-latency/) | Aerospike | Excellent P99 latency overview — causes, measurement, reduction strategies |
| [aws.amazon.com/builders-library](https://aws.amazon.com/builders-library/) | Amazon | Production patterns: timeouts, retries, circuit breakers, load shedding |
| [sre.google/sre-book](https://sre.google/sre-book/table-of-contents/) | Google SRE | Monitoring, alerting, capacity planning, cascading failures |

### GitHub Repositories (Study These)

| Repository | What to study |
|------------|--------------|
| [LMAX-Exchange/disruptor](https://github.com/LMAX-Exchange/disruptor) | Ring buffer, memory barriers, cache line padding |
| [real-logic/aeron](https://github.com/real-logic/aeron) | Reliable UDP, shared memory IPC, busy-spin |
| [real-logic/simple-binary-encoding](https://github.com/real-logic/simple-binary-encoding) | Zero-copy serialization used by CME, LSE |
| [real-logic/agrona](https://github.com/real-logic/agrona) | Off-heap buffers, lock-free collections |
| [JCTools/JCTools](https://github.com/JCTools/JCTools) | Lock-free SPSC/MPSC queues — used inside Netty |
| [OpenHFT/Chronicle-Queue](https://github.com/OpenHFT/Chronicle-Queue) | Microsecond-level persisted queue |
| [OpenHFT/Chronicle-Map](https://github.com/OpenHFT/Chronicle-Map) | Off-heap concurrent map |
| [OpenHFT/Java-Thread-Affinity](https://github.com/OpenHFT/Java-Thread-Affinity) | Thread pinning for Java |
| [lemire/simdjson](https://github.com/simdjson/simdjson) | Parsing JSON at GB/s with SIMD |
| [RoaringBitmap/RoaringBitmap](https://github.com/RoaringBitmap/RoaringBitmap) | Compressed bitmaps for ad tech |
| [axboe/liburing](https://github.com/axboe/liburing) | io_uring library |
| [DPDK/dpdk](https://github.com/DPDK/dpdk) | Kernel bypass networking |
| [google/highway](https://github.com/google/highway) | Portable SIMD library (C++) |
| [crossbeam-rs/crossbeam](https://github.com/crossbeam-rs/crossbeam) | Rust lock-free structures, epoch reclamation |
| [facebook/folly](https://github.com/facebook/folly) | Facebook's concurrent data structures |
| [abseil/abseil-cpp](https://github.com/abseil/abseil-cpp) | Google's Swiss table hash map |
| [jemalloc/jemalloc](https://github.com/jemalloc/jemalloc) | Memory allocator (Facebook, Redis) |
| [microsoft/mimalloc](https://github.com/microsoft/mimalloc) | Microsoft's compact allocator |
| [brendangregg/FlameGraph](https://github.com/brendangregg/FlameGraph) | Flame graph generation scripts |
| [async-profiler/async-profiler](https://github.com/async-profiler/async-profiler) | Low-overhead JVM profiler |
| [tokio-rs/tokio](https://github.com/tokio-rs/tokio) | Rust async runtime — study the work-stealing scheduler |
| [openjdk/jmh](https://github.com/openjdk/jmh) | JMH source — how proper benchmarking works |
| [djiangtw/tech-column-public](https://github.com/djiangtw/tech-column-public/) | 100 articles on computer architecture, cache design, storage (NVMe/CXL), embedded RTOS — hardware-level systems knowledge |
| [djiangtw/data-structures-in-practice-public](https://github.com/djiangtw/data-structures-in-practice-public) | Data structures from a hardware-aware perspective — cache behavior, memory hierarchy, SIMD, lock-free structures, allocators. 20 chapters with benchmarks. |

### Research Papers & Specs

| Paper / Spec | Why |
|-------------|-----|
| LMAX Disruptor Technical Paper | The design document behind the Disruptor |
| Nasdaq ITCH 5.0 Protocol Spec | How real exchange market data works |
| CME MDP 3.0 (SBE) | How real exchange messages are encoded |
| "What Every Programmer Should Know About Memory" — Drepper | The foundational memory hierarchy paper |
| Intel 64 and IA-32 Optimization Reference Manual | Official CPU optimization guidance from Intel |

## Latency Numbers Every Programmer Should Know

```
L1 cache reference ......................... 0.5 ns
Branch mispredict .......................... 5   ns
L2 cache reference ......................... 7   ns
Mutex lock/unlock .......................... 25  ns
L3 cache reference ......................... 30  ns      (shared across cores)
Main memory reference ...................... 100 ns      (20x L2, 200x L1)
SIMD instruction ........................... 1-5 ns      (processes 4-16 elements)
System call (via vDSO) ..................... 50  ns
System call (real) ......................... 500 ns
Context switch ............................. 5   μs
io_uring submission ........................ 1   μs
epoll_wait ................................. 5   μs
SSD random read ............................ 16  μs
SSD sequential read (1MB) .................. 50  μs
HFT tick-to-trade (competitive) ........... 5   μs
RTB bid response deadline .................. 10  ms      (= 10,000 μs)
HDD seek .................................. 10  ms      (= 10,000,000 ns)
TCP packet roundtrip (same datacenter) ..... 500 μs
TCP packet roundtrip (cross-continent) ..... 150 ms
```
