# Level 9 — Project: Real-Time Bidding (RTB) Ad Serving Engine

## The Project

Build a real-time bidding engine that responds to bid requests in <10ms p99 at 100K+ QPS. Three implementations: Java (baseline, Spring-style), C++ (optimized), and Rust (optimized). Different domain than the order book — this is about **throughput at scale** with bounded latency, not single-event microsecond latency.

## Target Performance

| Implementation | Target p99 latency | Target throughput | Why |
|---------------|-------------------|-------------------|-----|
| Java (baseline) | ~30-50ms | ~10K QPS | Spring MVC overhead, GC pressure, boxing in collections, Jackson JSON allocation |
| C++ (optimized) | <10ms | 100K+ QPS | Custom HTTP handling, Roaring Bitmaps, ONNX inference, zero-copy parsing |
| Rust (optimized) | <10ms | 100K+ QPS | Tokio async, zero-copy parsing, Roaring Bitmaps, ort for ML inference |

## How RTB Works

```
1. User visits a webpage
2. Ad exchange sends bid request to all bidders (JSON/Protobuf, 100ms deadline)
3. Each bidder:
   a. Parse bid request (user info, page context, ad slots)
   b. Look up user segments (is this user in "sports fans" audience?)
   c. Score with ML model (what's this impression worth?)
   d. Apply budget pacing (don't overspend)
   e. Return bid response with price
4. Exchange picks highest bidder, serves the ad
5. Total time budget: <100ms. Competitive bidders target <10ms p99.
```

## Techniques Applied from Each Level

| Level | Technique | How it's applied in the RTB bidder |
|-------|-----------|-----------------------------------|
| **L1** Cache | Roaring Bitmaps = contiguous compressed data, not scattered HashSet objects | Java `HashSet<Integer>` per user = heap-scattered objects. Roaring Bitmap = cache-friendly compressed bitset. |
| **L1** False sharing | Pad atomic budget counter | Budget pacer's atomic counter on its own cache line — prevents contention from concurrent bid threads |
| **L2** Benchmarking | wrk2/vegeta load testing with HdrHistogram | wrk2 corrects for coordinated omission — proper latency percentile measurement under load |
| **L2** Flame graphs | Profile each implementation under load | Identify: is the bottleneck in parsing? ML inference? User lookup? Serialization? |
| **L3** Atomics | Budget pacing with relaxed atomics | `AtomicU64` with `Ordering::Relaxed` for budget tracking — no lock, eventual consistency is fine for pacing |
| **L3** Concurrency | Async/concurrent request handling | Rust: Tokio async with work-stealing. C++: thread pool with connection pinning. Java: Spring MVC thread-per-request (the bottleneck). |
| **L4** SIMD | Roaring Bitmap intersection uses SIMD | Checking if user is in audience segment = bitmap AND operation → SIMD-accelerated on both C++ and Rust |
| **L4** Hash maps | Feature store with cache-friendly maps | Java HashMap (chaining, boxing) vs C++ abseil flat_hash_map vs Rust HashMap (Swiss table) |
| **L4** Allocators | Arena allocation per request | C++: arena allocator for per-request temp data, freed when request completes. Rust: similar. Java: allocates on every request → GC. |
| **L5** CPU pinning | Pin hot threads to cores | HTTP accept loop and worker threads pinned to specific cores — reduce scheduling jitter |
| **L6** TCP tuning | `TCP_NODELAY`, `SO_REUSEPORT` | Every ms of TCP buffering matters in a 10ms budget. `SO_REUSEPORT` for multi-threaded accept. |
| **L7** GC analysis | Java GC under sustained load | 100K QPS × objects per request = massive allocation rate. Show GC pause frequency destroying p99. |
| **L7** Allocation profiling | Jackson JSON allocation measurement | async-profiler allocation flame graph: show that Jackson creates 50+ objects per JSON parse |

## Components — Detailed

| Component | Java approach (slow) | C++ approach (fast) | Rust approach (fast) | Perf impact |
|-----------|---------------------|--------------------|--------------------|-------------|
| **HTTP server** | Spring MVC — new thread per request, reflection-based routing, creates objects per request | Custom HTTP parser on thread pool, reuse buffers, minimal allocation | Tokio + hyper — async I/O, zero-copy, work-stealing runtime | 2-3x: Spring's per-request object creation + reflection overhead |
| **JSON parsing** | Jackson — creates `JsonNode` tree, allocates strings, boxed numbers | simdjson (SIMD-accelerated, 2.5 GB/s) or manual field extraction from buffer | simd-json crate or serde_json with zero-copy (`&str` borrowing) | 3-10x: Jackson creates 50+ objects per parse. simdjson creates ~0. |
| **User segments** | `HashMap<Long, Set<Integer>>` — boxed Long keys, HashSet per user with boxed Integers | `roaring::Roaring64Map` — compressed bitmaps, SIMD intersection | `roaring` crate — same compressed bitmap, SIMD-accelerated | 5-20x: HashSet lookup = pointer chasing. Bitmap = single AND + popcount. |
| **Feature store** | `HashMap<String, Object>` — string keys, boxed values, chaining | `absl::flat_hash_map<uint64_t, Feature>` — flat, no boxing, open addressing | `std::collections::HashMap<u64, Feature>` — Swiss table | 2-5x: cache-friendly open addressing vs pointer-chasing chains |
| **ML inference** | — (not practical in Java without native bridge) | ONNX Runtime C++ API — batched inference, optimized graph execution | `ort` crate (Rust ONNX Runtime bindings) — same engine | N/A for Java. ONNX Runtime adds ~1-5ms per inference depending on model. |
| **Budget pacing** | `synchronized` block around budget counter | `std::atomic<uint64_t>` with `memory_order_relaxed` | `AtomicU64` with `Ordering::Relaxed` | 5-10x under contention: `synchronized` = lock. Atomic = lock-free. |
| **Response serialization** | Jackson JSON serialization — allocates strings, boxed numbers | Manual buffer write or simdjson serialization | serde_json with `Vec<u8>` buffer reuse | 2-5x: Jackson allocates. Manual write = pre-allocated buffer. |
| **Connection pool** | Spring's default pool — per-request allocation overhead | Pre-allocated connection pool, buffer reuse across requests | hyper connection pooling with keep-alive | 1-2x: connection setup cost amortized |

## Latency Budget (10ms target)

| Phase | Budget | Java actual | C++/Rust actual |
|-------|--------|-------------|-----------------|
| HTTP parsing | 1ms | ~2-3ms (Spring + Jackson ObjectMapper) | ~0.5ms (Vert.x + Jackson Streaming) | ~0.1-0.5ms (simdjson/hyper) |
| User segment lookup | 1ms | ~3-5ms (HashMap boxing) | ~1ms (Redis SISMEMBER via Lettuce) | ~0.1-0.3ms (Roaring Bitmap) |
| Feature lookup | 1ms | ~1-2ms (HashMap chaining) | ~0.5ms (Redis or in-process cache) | ~0.1-0.3ms (flat_hash_map) |
| ML inference | 3ms | N/A | ~3-5ms (ONNX via Panama FFI) | ~1-5ms (ONNX Runtime native) |
| Budget check | 0.5ms | ~1-2ms (synchronized) | ~0.01ms (AtomicLong) | ~0.01ms (atomic) |
| Response serialization | 1ms | ~2-3ms (Jackson ObjectMapper) | ~0.3ms (Jackson Streaming / JsonGenerator) | ~0.1-0.5ms (manual) |
| **Total** | **<10ms** | **~10-15ms (Spring), 30-50ms p99 with G1 GC** | **~5-10ms, <15ms p99 with ZGC** | **~2-7ms** |

> Note: "Java optimized" column = Java 21+ with Vert.x, Virtual Threads, ZGC, Jackson Streaming, Lettuce async Redis — the version we build first. "Java Spring baseline" = typical Spring Boot for comparison later.

## Project Structure

```
docs/
  architecture.md           ← Bid request flow diagram, component responsibilities
  latency_budget.md         ← Per-component time budget with measurements

cpp-bidder/
  CMakeLists.txt
  include/
    bid_handler.hpp         ← HTTP handler: parse request, lookup, score, respond
    user_store.hpp          ← Roaring Bitmap audience segments, SIMD intersection
    feature_store.hpp       ← absl::flat_hash_map feature lookup
    ml_scorer.hpp           ← ONNX Runtime inference, batched scoring
    budget_pacer.hpp        ← Atomic budget tracking with relaxed ordering
  src/
    main.cpp
    bid_handler.cpp
    user_store.cpp
  bench/
    bidder_bench.cpp        ← Google Benchmark per-component + end-to-end

rust-bidder/
  Cargo.toml
  src/
    main.rs                 ← Tokio async HTTP server (hyper-based)
    bid_handler.rs          ← Request parsing, bid logic, response
    user_store.rs           ← Roaring Bitmaps via roaring crate
    ml_scorer.rs            ← ONNX Runtime via ort crate
    budget_pacer.rs         ← AtomicU64 budget tracking
  benches/
    bidder_bench.rs         ← Criterion per-component + end-to-end

java-baseline/
  pom.xml                   ← Maven with Spring Boot + JMH
  src/main/java/bidder/
    BidController.java      ← Spring MVC — object allocation per request, Jackson parsing
    UserStore.java          ← HashMap<Long, Set<Integer>> — boxing, GC pressure
    FeatureStore.java       ← HashMap<String, Object> — string keys, boxing
    BidderBenchmark.java    ← JMH per-component + end-to-end

results/
  throughput_comparison.md  ← QPS at various latency percentiles (p50, p99, p999)
  latency_curves.md         ← QPS vs latency graph data — find the knee point
  flamegraphs/              ← Flame graph SVGs under load for each implementation
  gc_analysis.md            ← Java GC behavior under sustained 10K+ QPS
```

## What We'll Produce

1. Working bidder in all 3 languages that can handle simulated bid requests (OpenRTB-like JSON)
2. Load test with wrk2 (coordinated-omission-corrected) — QPS vs latency curves
3. Flame graphs under load showing where time goes in each implementation
4. Per-component latency breakdown: parsing, lookup, scoring, pacing, serialization
5. Throughput comparison at p50, p99, p999 latency targets — find the saturation point
6. Java GC analysis: allocation rate, pause frequency, p999 correlation
7. Java allocation flame graph: which components create the most garbage
8. Written analysis: Java allocation pressure vs C++/Rust zero-allocation hot paths, per-component

## Key Differences from Level 8 (Order Book)

| Dimension | Order Book (Level 8) | RTB Bidder (Level 9) |
|-----------|---------------------|---------------------|
| Optimization target | Per-event latency (microseconds) | Throughput at bounded latency (QPS at <10ms) |
| Concurrency model | Single-threaded hot path, lock-free I/O | Highly concurrent: thousands of simultaneous requests |
| I/O pattern | UDP multicast (fire and forget) | HTTP request/response (TCP, connection pooling) |
| Data structures | Order book (flat array by price) | Bitmaps (audience), hash maps (features), ML model |
| Key bottleneck | Cache misses, allocation per order | Serialization overhead, GC pressure, contention at scale |
| Async patterns | Not applicable (synchronous hot path) | Critical — async I/O (Tokio/epoll) handles thousands of concurrent connections |
| ML component | None | ONNX Runtime inference on every request |

## References

- OpenRTB 2.6 Specification (IAB) — the protocol ad exchanges use
- Roaring Bitmaps paper — "Better bitmap performance with Roaring bitmaps" (Chambi et al.)
- simdjson — "Parsing Gigabytes of JSON per Second" (Lemire et al.)
- ONNX Runtime documentation — model loading, session options, inference API
- wrk2 — coordinated-omission-corrected load testing (Gil Tene's fork)
- vegeta — HTTP load testing tool (Go, generates HdrHistogram-compatible output)
- "Building Low Latency Applications with C++" — Sourav Ghosh (ad tech chapter)
