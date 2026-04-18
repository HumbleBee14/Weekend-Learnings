# Level 7 — JVM Performance

The "know your enemy" level. Before we build the projects (levels 8-9), understand exactly what's happening inside the JVM — how GC works, how JIT compiles your code, what escape analysis does, and the workarounds that LMAX, Aeron, and Chronicle use to make Java competitive with C++ in latency-critical systems.

This level is NOT about making Java fast enough. It's about understanding WHY Java is slow, seeing the internals, and appreciating the engineering that goes into fighting the runtime. Then in levels 8-9, we build in C++/Rust where these problems simply don't exist.

## Topics

### 01 — GC Internals and Tuning
- How garbage collection actually works: mark-sweep, generational hypothesis, card tables
- GC algorithms compared: G1 (default), ZGC (sub-ms pauses), Shenandoah (concurrent), Epsilon (no-op GC)
- Azul Zing / C4 — the "pauseless" collector used by HFT Java shops
- GC tuning parameters: heap sizing, generation ratios, GC logging (`-Xlog:gc*`)
- Why even ZGC's sub-millisecond pauses kill p999 in HFT (microsecond world vs millisecond GC)
- **Code**: Same workload under G1, ZGC, Shenandoah, Epsilon — measure pause times, throughput, tail latency
- **Contrast**: C++ — no GC. Rust — no GC. Deterministic deallocation via RAII/ownership. Zero pauses. Always.

### 02 — JIT Compilation Deep Dive
- Tiered compilation: interpreter → C1 (fast compile, simple opts) → C2 (slow compile, aggressive opts)
- OSR (On-Stack Replacement): compiling a method while it's running
- What C2 optimizes: inlining, loop unrolling, escape analysis, intrinsics, dead code elimination
- `-XX:+PrintCompilation`, `-XX:+PrintInlining`, JITWatch visualization
- Warmup problem: first 5-10 seconds of a Java app run 10-100x slower than steady state
- GraalVM Native Image: AOT compilation — eliminates warmup, but loses adaptive optimization
- **Code**: Benchmark cold vs warm performance, `-XX:CompileThreshold` tuning, JITWatch analysis
- **Contrast**: C++/Rust — AOT compiled, performance is consistent from first instruction. No warmup.

### 03 — Off-Heap and sun.misc.Unsafe
- The problem: every Java object lives on the heap → subject to GC scanning
- `ByteBuffer.allocateDirect()`: allocate off-heap, avoid GC, but clunky API
- `sun.misc.Unsafe`: direct memory access, allocation, CAS — the backdoor that powers low-latency Java
- `MemorySegment` (Panama/FFM API, Java 21+): the official replacement for Unsafe
- How Chronicle Queue and Chronicle Map use off-heap memory for microsecond-level persistence
- How Agrona's `MutableDirectBuffer` wraps off-heap memory for zero-copy
- **Code**: On-heap vs off-heap allocation benchmark, Unsafe vs MemorySegment comparison
- **Contrast**: C++ — all memory is "off-heap" by default. Rust — stack by default, heap explicit. No GC to dodge.

### 04 — Allocation-Free Java
- The goal: zero `new` on the hot path — if you don't allocate, GC can't pause you
- Object pooling: pre-allocate, reuse, return — `ArrayDeque` as a pool
- Flyweight pattern: one object, swap the underlying buffer — used by SBE
- `@Contended` annotation: prevent false sharing (adds 128 bytes of padding)
- SBE (Simple Binary Encoding): zero-allocation serialization — encode/decode without creating objects
- Agrona collections: `Int2IntHashMap`, `Long2ObjectHashMap` — no boxing, no autoboxing
- The Disruptor's pre-allocated ring buffer: events are reused, never GC'd
- **Code**: Allocation-heavy vs allocation-free version of the same logic — async-profiler allocation flame graph
- **Contrast**: C++ — stack allocation is the default. Rust — move semantics + no GC. The workarounds Java needs are just... how C++/Rust work normally.

### 05 — Vector API and Panama FFI
- Vector API (JEP 338/417, incubating): SIMD from Java — `FloatVector.fromArray()`, `add()`, `intoArray()`
- Limitations: incubating (not stable), limited to simple operations, can't match manual intrinsics
- Panama FFI (JEP 424, Java 21+): call native C/C++ code from Java without JNI overhead
- When to use Panama FFI: when you need C++ performance for one function but Java for everything else
- How this relates to Python FFI (Maturin/PyO3, Cython, pybind11) — same idea, different runtime
- **Code**: Vector API SIMD sum vs scalar Java sum vs C++ intrinsics — benchmark all three
- **Contrast**: C++ — full `_mm256_*` intrinsics. Rust — `std::simd` or `packed_simd`. No incubating, no limitations.

## The Bigger Picture

After this level, you'll understand exactly what the JVM does under the hood and why each of these mechanisms exists:

| JVM Mechanism | What it does | Performance cost | C++/Rust equivalent |
|--------------|-------------|-----------------|-------------------|
| GC | Automatic memory management | Pauses (ms to sub-ms) | RAII / ownership (zero cost) |
| JIT | Runtime compilation + optimization | Warmup (5-30 seconds) | AOT compilation (instant) |
| Object headers | Type info, GC metadata, lock state | 12-16 bytes per object | No headers (0 bytes) |
| Heap allocation | All objects on managed heap | GC pressure, cache pollution | Stack/arena/pool (you choose) |
| Autoboxing | `int` → `Integer` for collections | Extra allocation + indirection | No boxing (templates/generics) |
| Safepoints | JVM thread suspension points | Latency jitter | No safepoints |

**This is the level where the "Java as counter-example" philosophy comes full circle.** You understand the machine (levels 1-6), you understand the JVM (level 7), and now you can explain precisely why C++/Rust wins in the projects (levels 8-9).

## Key Reading for This Level

- "Java Performance" — Scott Oaks (GC, JIT, threading chapters)
- "Optimizing Java" — Evans, Gough, Newland (JIT analysis, JMH, GC tuning)
- Shipilev — "JVM Anatomy Quarks" series (30+ deep-dive posts on JVM internals)
- Nitsan Wakart — psy-lob-saw.blogspot.com (JCTools author, JVM perf blog)
- Martin Thompson — mechanical-sympathy.blogspot.com (Disruptor, allocation-free Java)
- OpenJDK source code: `hotspot/src/share/gc/` (GC implementations)
- SBE, Agrona, Disruptor source code — the allocation-free Java playbook

## Mini-Project: Allocation-Free Queue (`06-mini-project-allocation-free-queue/`)

**Apply everything from this level in one focused build.**

Build the same FIFO queue twice in Java: once "normally" (allocating), once allocation-free using every trick from this level. Then build the C++/Rust equivalents to show the baseline these tricks are trying to reach.

| Variant | Techniques used | Allocations per enqueue |
|---------|----------------|------------------------|
| **Java naive** | `new Node<>()` per enqueue, GC manages everything | 1 object (Node) + potential GC trigger |
| **Java allocation-free** | Pre-allocated node pool, flyweight pattern, off-heap with `Unsafe`/`MemorySegment`, `@Contended` padding | **Zero** — reuse pooled nodes, no GC pressure |
| **C++ baseline** | `new` per enqueue (heap allocation) | 1 malloc (but no GC) |
| **C++ pool-allocated** | Arena/pool allocator, placement new | Zero malloc — bump pointer |
| **Rust baseline** | `Box::new()` per enqueue | 1 alloc (but no GC) |
| **Rust arena** | `bumpalo` arena allocation | Zero alloc — bump pointer |

**What this demonstrates**:
- The Java allocation-free version should approach C++/Rust throughput for the queue operations themselves
- BUT: the Java version requires 10x more code, `Unsafe` usage, and careful object lifecycle management
- Meanwhile C++/Rust get this performance naturally — no GC to dodge, no object headers, no boxing

**Measure with**:
- async-profiler allocation flame graph: Java naive shows millions of allocations, allocation-free shows zero
- GC log: Java naive triggers GC pauses under sustained load, allocation-free doesn't
- JMH / Google Benchmark / Criterion: throughput and latency percentiles
- `perf stat -e cache-misses`: pool-allocated versions have better cache behavior

**Produce**: Allocation flame graph (before/after), GC pause correlation, throughput comparison, written analysis of effort vs performance for each approach
