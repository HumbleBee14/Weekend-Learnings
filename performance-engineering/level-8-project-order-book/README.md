# Level 8 — Project: HFT Order Book Matching Engine

## The Capstone Project

Build the same price-time-priority order book matching engine in three languages: Java (the slow baseline), C++ (optimized), and Rust (optimized). Apply every technique from levels 1-7. Measure everything. Explain exactly where and why Java loses.

## Target Latency

| Implementation | Target per-order latency | Why |
|---------------|-------------------------|-----|
| Java (baseline) | ~50-100μs | GC pauses, object overhead, pointer chasing, JIT warmup |
| C++ (optimized) | <5μs | Flat arrays, zero-copy SBE, no GC, cache-friendly layout |
| Rust (optimized) | <5μs | Same as C++ with safety guarantees, custom allocator |

**Goal: demonstrate 10-20x gap with full explanation of every contributing factor.**

## Techniques Applied from Each Level

This project is the integration point — every level's lessons converge here.

| Level | Technique | How it's applied in the order book |
|-------|-----------|-----------------------------------|
| **L1** Cache | Contiguous price level array, no pointer chasing | Java TreeMap = pointer chasing per lookup. C++/Rust = flat array indexed by price → 1 cache miss vs 5+ |
| **L1** False sharing | Pad hot fields across cache lines | Sequence counters, head/tail indices on separate cache lines |
| **L1** Huge pages | Pre-allocate order pool on huge pages | `mmap MAP_HUGETLB` for the order pool — eliminate TLB misses on hot data |
| **L2** Benchmarking | JMH / Google Benchmark / Criterion with HdrHistogram | Proper warmup, no dead code elimination, latency percentiles (p50/p99/p999) |
| **L2** Flame graphs | Profile each implementation | Identify remaining hotspots after optimization — flame graph SVGs in results/ |
| **L2** perf counters | `perf stat` comparison | cache-misses, branch-misses, IPC side-by-side for all 3 implementations |
| **L3** Lock-free | Single-writer matching engine, lock-free I/O boundary | Matching engine runs on one dedicated thread (no locks). I/O threads communicate via lock-free ring buffer. |
| **L3** Disruptor pattern | Ring buffer for order input/output | Orders flow through a pre-allocated ring buffer — zero allocation, cache-friendly |
| **L4** SoA layout | Order book price levels as flat array | Price levels = contiguous array indexed by price tick. Not a tree. Not a linked list. |
| **L4** Branchless | Branchless order matching | Side selection (buy/sell), price comparison — eliminate branches in the hot match loop |
| **L4** Pool allocator | Pre-allocated order pool | Fixed pool of Order structs. Allocate = pop from freelist. Free = push to freelist. Zero malloc on hot path. |
| **L5** CPU pinning | Matching engine thread pinned to isolated core | `isolcpus` + `taskset` — no scheduling jitter, no context switches |
| **L5** Huge pages | Order pool and ring buffer on huge pages | Eliminate TLB misses for the most frequently accessed memory |
| **L6** UDP multicast | Market data distribution | Price updates sent via multicast — one send, all subscribers receive |
| **L6** TCP tuning | Order entry connection | `TCP_NODELAY`, tuned buffer sizes for order entry TCP connection |
| **L7** GC analysis | Java baseline GC impact | Measure GC pause frequency and duration under load — show p999 spikes from GC |
| **L7** Allocation profiling | Java allocation flame graph | async-profiler allocation mode — show exactly where Java allocates on the hot path |

## Components — Detailed

| Component | Java approach (slow) | C++/Rust approach (fast) | Perf impact |
|-----------|---------------------|-------------------------|-------------|
| **Order Book** | `TreeMap<Price, Queue<Order>>` — boxing, pointer chasing, O(log N) lookup | Flat array indexed by price tick — O(1) lookup, contiguous memory | 10-50x: cache misses dominate Java's tree traversal |
| **Order struct** | Object with 16-byte header, heap-allocated, GC-tracked | Packed struct (`__attribute__((packed))` / `repr(C, packed)`), pool-allocated, no header | 3-5x: Java's 16-byte header per order wastes cache space |
| **Order pool** | `new Order()` per incoming order — GC pressure | Pre-allocated pool on huge pages, freelist-based alloc/free | 10-100x: Java `new` = potential GC trigger. Pool = pointer bump. |
| **Message parsing** | `String.split()` FIX parsing — allocates strings per field | SBE (Simple Binary Encoding) — flyweight decoder, zero-copy, zero allocation | 5-20x: Java creates 20+ String objects per message. SBE creates zero. |
| **Market data out** | TCP, `ObjectOutputStream` or JSON serialization | UDP multicast, SBE-encoded messages, kernel bypass ready | 3-10x: TCP overhead + serialization vs raw multicast + SBE |
| **Matching engine** | `synchronized` methods, object-oriented dispatch | Single-threaded, branchless matching, no locks on hot path | 5-10x: lock acquisition + method dispatch overhead eliminated |
| **Ring buffer I/O** | `ArrayBlockingQueue` (lock-based) | Disruptor-style ring buffer (lock-free, padded sequences) | 10-100x: lock contention under load destroys ABQ's p99 |

## Project Structure

```
docs/
  architecture.md           ← Component diagram, data flow, hot path identification
  protocol_spec.md          ← SBE message schemas, order types (limit, market, cancel)
  latency_budget.md         ← Nanosecond budget: parse (X ns) + match (X ns) + publish (X ns)

java-baseline/              ← Idiomatic Java — the "before"
  pom.xml                   ← Maven with JMH dependency
  src/main/java/orderbook/
    OrderBook.java          ← TreeMap-based, GC-heavy
    Order.java              ← POJO with boxing
    MatchingEngine.java     ← Synchronized, allocating
    FIXParser.java          ← String.split() parsing
    MarketDataPublisher.java
  src/test/java/orderbook/
    OrderBookBenchmark.java ← JMH benchmarks with HdrHistogram latency recording

cpp-optimized/              ← Every technique from levels 1-7 applied — the "after"
  CMakeLists.txt
  include/
    order_book.hpp          ← Flat array price levels, intrusive linked lists
    order.hpp               ← Packed struct, cache-line aligned, pool-allocated
    matching_engine.hpp     ← Single-threaded hot path, branchless matching
    sbe_codec.hpp           ← SBE zero-copy encode/decode
    market_data_publisher.hpp ← UDP multicast, SBE-encoded
    ring_buffer.hpp         ← Disruptor-style lock-free ring buffer
    order_pool.hpp          ← Pre-allocated pool on huge pages
  src/
    main.cpp
    order_book.cpp
    matching_engine.cpp
  bench/
    order_book_bench.cpp    ← Google Benchmark + HdrHistogram
  test/
    order_book_test.cpp

rust-optimized/             ← Rust alternative — safety + zero-cost abstractions
  Cargo.toml
  src/
    main.rs
    order_book.rs           ← Custom allocator, no Box on hot path
    order.rs                ← repr(C), packed, Copy trait
    matching_engine.rs      ← Single-threaded, branchless
    sbe_codec.rs            ← Zero-copy encode/decode
    market_data.rs          ← UDP multicast
    ring_buffer.rs          ← Lock-free ring buffer
    order_pool.rs           ← Arena/pool allocator
  benches/
    order_book_bench.rs     ← Criterion benchmarks

results/
  java_vs_cpp_vs_rust.md    ← Head-to-head: latency percentiles, throughput, analysis
  flamegraphs/              ← Flame graph SVGs for each implementation
  hdr_histograms/           ← Raw HdrHistogram data for percentile comparison
  perf_stat_comparison.md   ← perf stat output side-by-side (cache misses, IPC, branches)
```

## What We'll Produce

1. Working matching engine in all 3 languages (same logic, different performance)
2. JMH / Google Benchmark / Criterion benchmarks with proper methodology (Level 2)
3. Flame graphs for each implementation — where does time actually go? (Level 2)
4. HdrHistogram latency comparison: p50, p99, p999, max — the full percentile picture (Level 2)
5. `perf stat` hardware counter comparison: cache misses, branch misses, IPC (Level 2)
6. GC analysis for Java: pause frequency, duration, correlation with p999 spikes (Level 7)
7. Allocation flame graph for Java: exactly which `new` calls create GC pressure (Level 7)
8. Written analysis: where exactly Java loses, which technique fixes it, and how much each technique contributes to the speedup

## How to Verify the 10x Gap

For each contributing factor, measure before and after:

| Factor | How to measure | Expected contribution |
|--------|---------------|----------------------|
| Cache misses (TreeMap vs flat array) | `perf stat -e L1-dcache-load-misses` | 3-5x |
| GC pauses | GC log + HdrHistogram p999 | 2-5x on tail |
| Allocation overhead | async-profiler alloc mode | 2-3x |
| Lock contention (ABQ vs ring buffer) | `perf stat -e context-switches` | 5-10x on tail |
| Serialization (FIX strings vs SBE) | JMH per-message benchmark | 5-20x |
| JIT warmup | Cold vs warm benchmark comparison | 10-100x (first N seconds) |
| Object headers | Memory footprint comparison | 2-4x memory waste |

**The total 10-20x gap is the product of all these factors compounding.**

## References

- Nasdaq ITCH 5.0 Protocol Specification — how real exchange market data is structured
- CME MDP 3.0 (SBE) — how real exchange messages are encoded
- SBE (Simple Binary Encoding) — real-logic/simple-binary-encoding GitHub
- LMAX Disruptor technical paper — for the single-writer principle and ring buffer design
- "Building Low Latency Applications with C++" — Sourav Ghosh
- "Trading and Exchanges" — Larry Harris (market microstructure, order book mechanics)
