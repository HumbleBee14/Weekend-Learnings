# Level 3 — Memory Models and Concurrency

The hardest part of performance engineering. Most "fast" code is actually broken under concurrency. And most concurrent code that "works" is actually slower than single-threaded because of lock contention, false sharing, and memory ordering overhead.

The Jane Street insight applies here: **concurrency we think is helping can actually cause latency spikes if not implemented properly.** Adding threads to a CPU-bound workload just adds context switch overhead. Putting a lock on a hot path serializes your "parallel" code. Using `synchronized` in Java where `AtomicLong` would suffice costs 10-50x in contention.

This level teaches memory ordering, lock-free data structures, and the Disruptor pattern — with all the traps that Java's managed concurrency hides from you.

## Topics

### 01 — Memory Ordering
- **Hardware memory models**: x86-TSO (relatively strict — stores are ordered, only store-load reordering) vs ARM/RISC-V (relaxed — reorders everything, stores, loads, all of it)
  - x86 hides most ordering bugs from you — your code "works" on x86 but breaks on ARM
  - This is why concurrency bugs are so insidious: they're architecture-dependent
- **Language memory models**: Java Memory Model (JMM) vs C++ `std::memory_order` vs Rust `Ordering`
  - JMM: `volatile` gives acquire/release. `synchronized` gives full sequential consistency. No way to relax further.
  - C++: 6 memory orderings from `relaxed` (fastest, fewest guarantees) to `seq_cst` (slowest, full ordering)
  - Rust: Same orderings as C++ via `std::sync::atomic::Ordering`, plus the borrow checker prevents data races at compile time
- **`acquire`/`release` semantics** — the most important concept in lock-free programming
  - Release store: "everything I wrote before this store is visible to anyone who acquires this value"
  - Acquire load: "I can see everything the releaser wrote before their release store"
  - This is how you communicate between threads without locks
- **`seq_cst` cost**: Sequential consistency is the easiest to reason about but the most expensive
  - On x86: `seq_cst` store emits `MFENCE` or `XCHG` (full memory barrier) — 20-50ns overhead
  - On ARM: `seq_cst` emits `DMB ISH` barriers — even more expensive
  - `acquire`/`release` on x86 is FREE (x86-TSO already provides it) — use it when you can
- **Code**: Demo showing store reordering on ARM that x86 hides from you
- **Measure**: `perf stat -e cache-misses,bus-cycles` to see coherence traffic from atomic operations
- **Godbolt**: Compare assembly output of `relaxed` vs `acquire/release` vs `seq_cst` on x86 and ARM

### 02 — Lock-Free Data Structures
- **CAS (Compare-And-Swap)** — the building block of all lock-free code
  - `if (*ptr == expected) { *ptr = desired; return true; } else { return false; }`
  - Hardware instruction: `LOCK CMPXCHG` on x86, `LDXR`/`STXR` on ARM
  - CAS can fail spuriously — always use in a retry loop
- **ABA problem**: Value changes A→B→A, CAS thinks nothing changed
  - Solutions: tagged pointers (pack a counter with the pointer), hazard pointers, epoch-based reclamation
  - Java: `AtomicStampedReference` solves ABA with a version stamp
  - Rust: crossbeam's epoch-based reclamation — the standard approach
- **Build a lock-free SPSC (Single-Producer Single-Consumer) ring buffer from scratch**
  - Fixed-size array, head and tail indices, no locks
  - Producer only writes tail, consumer only reads head — no contention if properly padded
  - Cache line padding between head and tail to prevent false sharing (Level 1 lesson 03)
- **MPSC and MPMC queues**: Multiple producers or consumers — significantly harder
  - JCTools: `MpscArrayQueue`, `MpmcArrayQueue` — production-grade Java implementations used by Netty
  - crossbeam: `SegQueue`, `ArrayQueue` — Rust equivalents
- **Lock-free vs lock-based vs channel throughput comparison**:
  - Lock-free SPSC: highest throughput, lowest latency, but limited to single producer/consumer
  - Mutex-based queue: simple, correct, but 10-100x slower under contention
  - Go channels / Rust mpsc: convenient but have allocation and synchronization overhead
- **Code**: SPSC queue in Java, C++, and Rust. Benchmark vs mutex-based queue, Java ArrayBlockingQueue, and Rust channels.
- **Measure**: `perf stat -e context-switches,cache-misses` — lock-based shows high context switches, lock-free shows near zero

### 03 — The Disruptor Pattern
- **LMAX Disruptor**: The gold standard for inter-thread messaging in latency-critical Java
  - Pre-allocated ring buffer: events are reused, never GC'd — zero allocation on hot path
  - Sequence barriers: producers and consumers coordinate via atomic sequence numbers, not locks
  - Wait strategies: `BusySpinWaitStrategy` (lowest latency, burns CPU), `YieldingWaitStrategy`, `BlockingWaitStrategy`
  - Single-writer principle: each slot in the ring buffer is written by exactly one thread — eliminates contention
- **Why it's 10-100x faster than `ArrayBlockingQueue`**:
  - ABQ uses `ReentrantLock` → lock contention → context switches → latency spikes
  - Disruptor uses CAS on sequences → no locks → no context switches → predictable latency
  - ABQ allocates per enqueue (boxed entries). Disruptor pre-allocates everything.
  - Disruptor pads sequences to prevent false sharing. ABQ doesn't.
- **Study the actual source code**:
  - `Sequence.java`: the padded atomic long (128 bytes of padding around an 8-byte value)
  - `RingBuffer.java`: the pre-allocated event array, sequence management
  - `ProcessingSequenceBarrier.java`: how consumers wait for available events
- **C++ implementation of the core pattern**: Translate the ring buffer + sequence barrier to C++
  - Show that without GC, without object headers, the same pattern is even faster
- **Code**: Java Disruptor library benchmark vs ArrayBlockingQueue. C++ ring buffer implementation.
- **Measure**: Latency percentiles (p50, p99, p999) for both — Disruptor p999 should be dramatically better because no lock contention spikes

### 04 — Atomics Deep Dive
- **Atomic operations on x86**:
  - `fetch_add`: emits `LOCK XADD` — atomic increment, returns previous value
  - `compare_exchange`: emits `LOCK CMPXCHG` — the CAS instruction
  - `store` with `seq_cst`: emits `XCHG` or `MOV` + `MFENCE` — full barrier
  - `load` with `acquire`: just `MOV` on x86 (free! x86-TSO gives you acquire for free)
- **Java specifics**:
  - `AtomicLong`: single CAS-based counter — contends under high parallelism
  - `LongAdder`: striped counter (per-CPU cells, merge on read) — scales linearly but reads are slower
  - `synchronized`: OS-level lock → context switch potential → worst latency under contention
  - `VarHandle` (Java 9+): low-level access with explicit memory ordering control
  - When to use which: `AtomicLong` for low contention, `LongAdder` for high contention counters, `VarHandle` for lock-free algorithms
- **C++ specifics**:
  - `std::atomic<int64_t>` with `memory_order_relaxed`: no ordering guarantees, fastest
  - `memory_order_acquire`/`release`: sufficient for most lock-free algorithms
  - `memory_order_seq_cst`: full ordering, default, safest but slowest
  - **See the assembly on godbolt.org**: `relaxed` load = `MOV`, `seq_cst` store = `XCHG` — visible difference
- **Rust specifics**:
  - `AtomicU64` with `Ordering::Relaxed` vs `Ordering::SeqCst`
  - Rust's type system prevents accidental non-atomic access — compiler error, not runtime bug
  - `crossbeam::utils::CachePadded<AtomicU64>` for padding
- **Spinlocks**: Build one from `compare_exchange`. Show why it's terrible under contention (busy-waiting wastes CPU, priority inversion). Show when it IS appropriate (very short critical sections, real-time, kernel).
- **Code**: Counter benchmark in all 3 languages under 1, 2, 4, 8, 16 threads — show scaling curve
- **Measure**: `perf stat -e cache-misses,bus-cycles,context-switches` — `synchronized` shows context switches, atomics don't

### 05 — Thread Pools and Work Stealing
- **The concurrency pitfall**: Adding more threads ≠ more performance
  - **Amdahl's Law**: If 5% of code is serial, max speedup with ∞ cores is 20x. Period.
  - **CPU-bound**: optimal thread count = number of cores. More threads = context switch overhead.
  - **I/O-bound**: optimal thread count = cores × (1 + wait_time/compute_time). More threads ok because they're waiting.
  - At 80% CPU utilization, adding threads increases queue time faster than it adds throughput (Little's Law)
- **Work-stealing schedulers**:
  - Each thread has its own local deque (double-ended queue)
  - When a thread's deque is empty, it steals from another thread's deque
  - Stealing happens from the opposite end (LIFO for local, FIFO for stealing) — cache-friendly
  - Rayon (Rust): excellent work-stealing, automatic parallelism with `par_iter()`
  - ForkJoinPool (Java): work-stealing but GC pressure from task objects
  - Custom C++ pool: full control, can pin threads to cores
- **Thread-per-core architecture** (used in ScyllaDB, Seastar, Glommio):
  - One thread per core, no sharing, no locks, no context switches
  - Each thread owns its data — communication via message passing only
  - Eliminates ALL lock contention but requires careful data partitioning
  - Why it's popular in databases: predictable latency, no lock storms under load
- **Virtual Threads (Java 21)**:
  - Lightweight (few KB stack), millions possible — like goroutines
  - But still GC-managed, still have safepoint overhead, still can't do kernel bypass
  - Good for I/O-bound workloads (HTTP servers, DB connections) — NOT for CPU-bound or latency-critical
  - Don't use for HFT hot paths — use for the surrounding infrastructure (logging, admin endpoints)
- **Code**: Java ForkJoinPool vs Virtual Threads vs Rust Rayon vs C++ custom work-stealing pool
- **Measure**: Throughput and latency at 1, 2, 4, 8, 16, 32, 64 threads — find the inflection point where more threads hurts
- **Measure**: `perf stat -e context-switches,cpu-migrations` — thread-per-core should show near-zero

## Concurrency Anti-Patterns to Demonstrate

| Anti-Pattern | What happens | Fix |
|-------------|-------------|-----|
| Lock on hot path | Serializes parallel code, p99 spikes from lock contention | Lock-free (CAS), or restructure to single-writer |
| Too many threads (CPU-bound) | Context switch storm, cache thrashing, worse than single-threaded | Thread count = core count |
| `synchronized` everywhere (Java) | Every method call acquires a lock → contention → context switches | `AtomicLong`, `LongAdder`, or lock-free design |
| Shared mutable state | Cache coherence traffic, false sharing, MESI ping-pong | Thread-local state, message passing, or thread-per-core |
| Unbounded thread pool | Under load, creates thousands of threads → OOM or context switch death | Fixed pool, backpressure, load shedding |
| Blocking in async runtime | One blocking call stalls the entire Tokio/ForkJoin worker thread | Use `spawn_blocking` or dedicated thread for blocking ops |

## Key Reading for This Level

- "The Art of Multiprocessor Programming" — Herlihy & Shavit (Chapters 1-7, 10)
- "C++ Concurrency in Action" — Anthony Williams (Chapters 5-7: memory model, lock-free)
- Herb Sutter — "Atomic Weapons: The C++ Memory Model and Modern Hardware" (two-part talk)
- Martin Thompson — "LMAX Disruptor" technical paper + source code + "Mechanical Sympathy" blog
- "Is Parallel Programming Hard, And, If So, What Can You Do About It?" — Paul McKenney (perfbook, Chapters 1-5)
- Mara Bos — "Rust Atomics and Locks" — Rust-specific concurrency and atomics book
- Jane Street Tech Blog — concurrency pitfalls and latency analysis
- JCTools source code (GitHub) — production lock-free queues for Java
- crossbeam source code (GitHub) — production lock-free structures for Rust

## Mini-Project: Lock-Free Message Broker (`06-mini-project-message-broker/`)

**Apply everything from this level in one focused build.**

Build a simple in-process pub/sub message broker in Java, C++, and Rust. Publishers push messages to topics, subscribers consume them. The hot path must be lock-free.

| Component | What it exercises | Technique from this level |
|-----------|------------------|--------------------------|
| **Topic ring buffer** | Pre-allocated, fixed-size, lock-free | SPSC/MPSC queue (topic 02), Disruptor pattern (topic 03) |
| **Publish operation** | Atomic sequence increment, write to slot | CAS / fetch_add (topic 04), acquire/release ordering (topic 01) |
| **Subscribe operation** | Read from ring buffer, track per-subscriber position | Atomic load with acquire, busy-spin or yield wait strategy (topic 03) |
| **Multi-subscriber fan-out** | One publisher, N subscribers, each at their own position | Sequence barriers from Disruptor pattern (topic 03) |
| **Throughput scaling** | Benchmark with 1, 2, 4, 8 subscribers | Thread pool, work-stealing considerations (topic 05) |

**Java counter-example**: Use `ArrayBlockingQueue` with `synchronized` — show lock contention destroying throughput under multiple subscribers. Then show the lock-free version with 10-100x better p99.

**Measure with**: `perf stat -e context-switches,cache-misses`, HdrHistogram latency percentiles
**Produce**: Throughput vs subscriber count graph, p99 latency comparison (locked vs lock-free), flame graphs
