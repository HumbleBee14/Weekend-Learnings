# Level 10 — Production Performance Systems

Levels 1-7 taught the hardware/OS/runtime fundamentals. Levels 8-9 built projects. This level is about **why production systems are slow** — the application-level, architecture-level, and distributed-systems-level causes of latency that no amount of SIMD or kernel bypass will fix.

A single dropped packet can cascade into thousands of users seeing high latency. A cache stampede can bring down a database. A retry storm can amplify a minor hiccup into a full outage. Understanding these patterns is what separates "wrote fast code" from "built a fast system."

## Topics

### 01 — Tail Latency and Percentiles
- p50, p90, p95, p99, p999, max — what each tells you and when to use which
- Why averages are useless: a system with 1ms p50 and 500ms p99 feels broken
- "Tail at scale" (Google): if your request fans out to 100 services, the probability of hitting at least one p99 outlier is 63%
- Long-tail distributions: most requests are fast, but the slow ones are VERY slow
- Sources of tail latency: GC pauses, context switches, page faults, THP compaction, NIC interrupts, SSD wear-leveling, noisy neighbors
- Hedged requests: send the same request to 2 servers, take the first response
- HdrHistogram for correct percentile measurement (no coordinated omission)
- SLA/SLO definition around percentiles (e.g., "p99 < 50ms" as an SLO)
- **Queueing collapse**: At >80% utilization, queue time grows exponentially (M/M/1 model). At 90%, average wait time is 9x processing time. At 95%, it's 19x. This is why you can't run servers at 95% CPU and expect good p99.
- **Admission control**: When overloaded, reject requests at the door rather than accepting them and timing out halfway through. Better to return 503 for 10% of requests than serve 100% of requests slowly.
  - Token-based admission: acquire a token before processing, fixed number of tokens = bounded concurrency
  - Queue-depth admission: reject if queue length > threshold
  - Latency-based admission: reject if recent p99 exceeds target — adaptive, self-correcting
- **Load shedding vs graceful degradation**:
  - Load shedding: reject entire request (503)
  - Graceful degradation: serve partial result (show cached data, skip recommendation engine, return default)
  - Choice depends on the domain — financial systems shed (wrong answer is worse than no answer), content systems degrade (stale data is better than no data)
- **Code**: C++ simulation of fan-out amplification + hedged request strategy + queueing collapse demo
- **Analysis**: Calculate P(at least one p99 hit) for fan-out of N services. Plot queue time vs utilization curve (M/M/1).
- **Reference**: [Aerospike — What is P99 Latency?](https://aerospike.com/blog/what-is-p99-latency/)

### 02 — Database and Storage Performance
- **Disk and I/O outliers — the fundamental problem**:
  - Most reads are fast (served from page cache or SSD). But when a request triggers a cache miss or hits cold storage, it's 100-1000x slower.
  - HDD: most seeks ~10ms, but random seeks can hit 100ms. High variance.
  - SSD: mostly consistent ~16μs, but internal operations (GC, wear-leveling) create unpredictable stalls
  - Any non-uniform workload on disk or network produces outliers — this is the #1 source of p99 spikes in database-backed systems
  - The kernel page cache helps but is unpredictable — eviction decisions can cause sudden cache misses under memory pressure
- **Write-Ahead Log (WAL)**: Why databases write to a log before updating data — durability vs latency tradeoff
  - fsync frequency: every commit (safe, slow) vs batched (fast, risk of data loss)
  - WAL in PostgreSQL, MySQL InnoDB, RocksDB, etcd
  - The fsync dilemma: `fsync` on every write guarantees durability but costs ~1-10ms per call. Group commit batches multiple transactions into one fsync — trades latency for throughput
- **LSM Trees vs B-Trees**: Two fundamental storage engine architectures
  - B-Trees (PostgreSQL, MySQL): fast reads, slower writes, in-place updates
  - LSM Trees (RocksDB, Cassandra, LevelDB): fast writes, read amplification, compaction storms
  - Compaction: background merging of SSTables — can spike latency unpredictably
  - Compaction throttling: rate-limiting compaction I/O so it doesn't starve foreground queries
  - Leveled vs tiered compaction: leveled has more predictable read latency, tiered has better write throughput
- **SSD internals that affect latency**:
  - SSD garbage collection (reclaiming dead blocks — can pause I/O for milliseconds)
  - Wear-leveling (moving data between blocks — background I/O that steals bandwidth)
  - Write amplification (writing more data than requested — wears SSD faster AND slows writes)
  - Over-provisioning: leaving 10-20% of SSD unused reduces GC pressure, smooths out latency
  - TRIM support: tells SSD which blocks are free — reduces future GC pauses
- **mmap vs read/write vs O_DIRECT — three I/O strategies**:
  - `read()`/`write()`: Standard syscalls, data copied through kernel page cache → userspace buffer. Safe, simple, double-copy.
  - `mmap()`: Map file directly into virtual address space. Access file like memory. No explicit syscalls after mapping.
    - Advantage: no copy (page cache IS your buffer). Good for random-access read-heavy workloads.
    - Trap: page faults are unpredictable — first access to a page triggers a fault → disk read → latency spike
    - Trap: TLB pressure from large mappings. `munmap` is expensive. Hard to control memory usage.
    - Used by: MongoDB (WiredTiger), LMDB, some log readers
  - `O_DIRECT`: Bypass page cache entirely. Your buffer, your responsibility.
    - Advantage: no surprise evictions, deterministic latency, no double-buffering memory waste
    - Disadvantage: must handle alignment (512-byte or 4KB), must implement your own caching
    - Used by: Aerospike, ScyllaDB, MySQL InnoDB (optional), databases with custom buffer pools
- **Page cache behavior**:
  - Linux kernel caches recently read disk pages in free RAM — transparent to the application
  - Works great for most workloads. But the eviction policy (LRU-based) is unpredictable:
    - A sequential scan of a large file can evict your entire hot working set from cache
    - `posix_fadvise(FADV_DONTNEED)`: tell the kernel to evict pages after reading (prevent cache pollution)
    - `posix_fadvise(FADV_WILLNEED)`: tell the kernel to prefetch pages you'll need soon
  - Monitoring: `/proc/meminfo` (Cached, Dirty), `vmtouch` to see which files are in page cache
  - For latency-critical systems: either control the page cache carefully (fadvise) or bypass it entirely (O_DIRECT)
- **Direct I/O and raw disk access (`O_DIRECT`)**:
  - Bypass the kernel page cache entirely — your application manages its own caching
  - Why: kernel page cache eviction is unpredictable, causes sudden latency spikes
  - Who uses it: Aerospike, ScyllaDB, some HFT systems, databases with custom buffer pools
  - Same philosophy as kernel bypass networking (DPDK) but for storage
  - Tradeoff: you lose the kernel's caching but gain predictable, deterministic I/O latency
- **Index design for latency**:
  - Missing indexes → full table scans → p99 disaster on large tables
  - Over-indexing → write amplification → slower inserts/updates
  - Covering indexes: include all needed columns to avoid table lookups (avoid "index then heap" double I/O)
  - Bloom filters: probabilistic "is this key here?" — avoids unnecessary disk reads (used in LSM trees)
  - Adaptive indexing: hot data indexed in memory, cold data on disk
- **Connection pooling**:
  - TCP connection setup costs ~1ms (3-way handshake + TLS)
  - Pool sizing: too small → queue buildup; too large → memory waste + connection overhead
  - Connection pool exhaustion: all connections in use → new requests queue → p99 spikes
  - HikariCP (Java), pgbouncer (PostgreSQL), connection pooling in Rust (deadpool, bb8)
  - Idle connection timeout: connections sitting idle waste server resources — reclaim them
- **Notes**: Deep dive into WAL mechanics, LSM vs B-Tree tradeoffs, O_DIRECT vs page cache, connection pool math, SSD internal architecture

### 03 — Network Resilience and Cascading Failures
- **The cascade**: packet drop → TCP retransmission (200ms+) → application retry → hits different server (cold cache) → DB query → queue buildup → ALL requests slow
- **Retry storms**: If every client retries 3x on failure, a 10% error rate becomes 30% extra load → overload → more errors → more retries → system collapse
- **Exponential backoff with jitter**: The fix for retry storms
  - Base delay × 2^attempt + random jitter
  - Without jitter: all retries happen simultaneously ("thundering herd")
  - With jitter: retries spread out over time
- **Circuit breakers**: Stop calling a failing service entirely for a cooldown period
  - States: Closed (normal) → Open (failures detected, stop calling) → Half-Open (test if recovered)
  - Prevents cascading failures from propagating across service boundaries
- **Timeouts**:
  - No timeout = infinite wait = thread pool exhaustion = cascading failure
  - Connect timeout vs read timeout vs write timeout
  - Adaptive timeouts: adjust based on observed p99 of the downstream service
- **Backpressure and load shedding**:
  - Backpressure: slow down the sender when the receiver is overwhelmed (TCP window, reactive streams)
  - Load shedding: reject requests early when overloaded (503 Service Unavailable)
  - Better to serve 90% of requests fast than 100% of requests slow
- **Graceful degradation**: Return partial results instead of timing out completely
  - Example: Show cached product data even if the recommendation service is slow
- **TCP failure modes that cause latency**:
  - Packet loss → retransmission (200ms+ RTO)
  - Connection reset → full reconnection overhead
  - Head-of-line blocking (HTTP/1.1) → one slow response blocks the connection
  - Solution: HTTP/2 multiplexing, gRPC, or raw TCP with application framing
- **Notes**: Build a cascading failure simulator. Show how retries without backoff amplify failures.

### 04 — Caching Strategies
- **Why caching dominates p99**: A cache hit = 1ms. A cache miss → database = 10-50ms. At p99, you're measuring the misses.
- **Cache hit rate math**:
  - 99% hit rate: 1% of requests hit the DB → manageable
  - 95% hit rate: 5% of requests hit the DB → 5x more DB load
  - 90% hit rate: 10% → database starts struggling under load
  - Every 1% drop in hit rate can double your perceived tail latency
- **Cache stampede / thundering herd**:
  - Popular key expires → 1000 concurrent requests all miss → 1000 simultaneous DB queries → DB overload
  - Fix: lock-based recomputation (only one request recomputes, others wait)
  - Fix: probabilistic early recomputation (refresh before expiry)
  - Fix: stale-while-revalidate (serve stale, refresh in background)
- **Cache eviction policies**: LRU, LFU, ARC, W-TinyLFU (Caffeine)
  - LRU: simple but scan-resistant (one full table scan evicts your hot data)
  - W-TinyLFU (Caffeine/Java, Moka/Rust): admission + eviction, near-optimal hit rates
- **Working set sizing**: Your cache must hold the "hot" data. If working set > cache size, hit rate collapses.
- **Cache warming**: Pre-populate cache on startup to avoid cold-start latency storm
- **Multi-tier caching**: L1 (in-process, microseconds) → L2 (Redis/Memcached, sub-ms) → DB (10ms+)
- **Notes**: Cache hit rate impact modeling, stampede simulation, eviction policy comparison

### 05 — Serialization and Data Transfer
- **Serialization cost per request**: Every request pays serialize + deserialize on both ends
- **Format comparison**:
  | Format | Allocations | Speed | Size | Schema | Use when |
  |--------|-------------|-------|------|--------|----------|
  | JSON (Jackson) | Many (strings, objects) | Slow | Large | No | External APIs, debug |
  | Protobuf | Few | Fast | Small | Yes | Service-to-service |
  | FlatBuffers | Zero (zero-copy) | Fastest read | Small | Yes | Gaming, hot paths |
  | SBE | Zero (flyweight) | Fastest | Smallest | Yes | HFT, exchange protocols |
  | MessagePack | Few | Medium | Medium | No | Redis, compact JSON |
- **Zero-copy serialization**: SBE and FlatBuffers don't create objects — they read directly from the buffer
  - Java: Jackson JSON creates 50+ objects per request. SBE creates zero.
  - This directly affects GC pressure → p99 latency
- **Payload size matters**: Larger payloads = more network time, more serialization time, more memory
  - Compression: gzip/zstd reduce size but add CPU cost — tradeoff
  - Pagination: don't send 10,000 results when the user sees 20
  - Binary vs text: Protobuf is 3-10x smaller than JSON for the same data
- **Notes**: Benchmark JSON vs Protobuf vs SBE in Java and C++ — measure allocations, throughput, latency

### 06 — Distributed Systems Latency
- **Fan-out amplification**:
  - Request hits API gateway → fans out to 5 microservices → each hits a database
  - P(all 5 are fast) = (0.99)^5 = 0.95 → 5% of requests hit at least one slow service
  - With 100 services: P(at least one slow) = 1 - (0.99)^100 = 63%
  - **This is why microservices have worse tail latency than monoliths**
- **Cross-region latency**:
  - Same datacenter: ~0.5ms RTT
  - Cross-region (US East → US West): ~60ms RTT
  - Cross-continent (US → Europe): ~100-150ms RTT
  - Every cross-region call adds 100ms+ to your request latency
- **Service mesh overhead**: Envoy/Linkerd sidecar proxies add 1-3ms per hop
  - 5 microservice hops × 2ms sidecar overhead = 10ms added from proxies alone
- **Noisy neighbor / multi-tenancy**:
  - Cloud VMs share physical hardware — neighbor's batch job can steal your CPU cycles
  - cgroups / CPU limits: isolate your workload from neighbors
  - Dedicated instances vs shared: cost vs latency consistency tradeoff
- **Distributed tracing**: Following a single request across 10+ services to find the slow one
  - Jaeger, Zipkin, OpenTelemetry — instrument every service
  - Trace waterfall: see exactly which service added latency and where
  - Span-level percentile analysis: break down p99 by service
- **Notes**: Fan-out probability calculator, cross-region latency map, tracing setup guide

### 07 — High-Throughput Transactions at Scale
- **Horizontal scaling patterns**:
  - Sharding: split data across N databases by key (user_id % N)
  - Replication: read replicas for read-heavy workloads
  - Partitioning strategies: range, hash, consistent hashing
- **Load balancing for latency**:
  - Round-robin: simple but ignores server health
  - Least-connections: better but doesn't account for request complexity
  - P2C (Power of Two Choices): pick 2 random servers, choose the one with lower load — near-optimal
  - Latency-aware: route to the server with the lowest observed p99 — best for latency
- **Queue theory basics**:
  - At 80% utilization, average queue length starts growing fast
  - At 90% utilization, tail latency explodes (queue time >> processing time)
  - This is why you can't run servers at 95% CPU and expect good p99
- **Batch processing vs real-time**:
  - Background jobs (compaction, backups, analytics) interfere with real-time requests
  - Schedule batch jobs during off-peak or on dedicated hardware
  - Use priority queues: latency-sensitive requests always go first
- **Backpressure in message queues**: Kafka, Aeron, LMAX Disruptor
  - What happens when producer is faster than consumer
  - Ring buffer overflow strategies: block, drop, oldest-first eviction
- **Notes**: Queue theory simulation (M/M/1 model), load balancing comparison, sharding strategy analysis

### 08 — Capacity Planning
- **Little's Law**: `L = λW` (concurrent requests = arrival rate × average latency)
  - If arrival rate = 1000 req/s and average latency = 50ms → need 50 concurrent connections
  - If p99 spikes to 500ms → need 500 concurrent connections for 1% of traffic
- **Universal Scalability Law (USL)**: Why adding more servers doesn't scale linearly
  - Contention: serialized work (locks, shared state) limits parallelism
  - Coherence: coordination overhead (cache invalidation, consensus) grows with scale
  - At some point, adding more servers makes things SLOWER (retrograde scaling)
- **Throughput vs latency tradeoff**: You can have high throughput OR low latency, rarely both at once
  - Batching increases throughput but adds latency (wait for batch to fill)
  - Pipelining increases throughput without adding latency (but adds complexity)
- **Code**: Python capacity model using USL — predict when scaling breaks down

### 09 — Performance Observability
- **The three pillars**: Metrics, logs, traces — and how they complement each other for perf analysis
- **Metrics for performance**:
  - RED method: Rate, Errors, Duration (for request-based services)
  - USE method (Brendan Gregg): Utilization, Saturation, Errors (for resources)
  - The four golden signals (Google SRE): Latency, Traffic, Errors, Saturation
- **Percentile-based alerting**:
  - Alert on p99 > threshold, not average > threshold
  - Sliding window percentiles: how to compute p99 over last 5 minutes
  - Histogram aggregation challenges: you can't average percentiles across servers
- **Distributed tracing for latency debugging**:
  - OpenTelemetry: vendor-neutral instrumentation
  - Jaeger: open-source trace storage and visualization
  - Critical path analysis: which span in the trace is the bottleneck?
  - Trace sampling: you can't trace every request in production — sample strategies
- **Profiling in production**: Continuous profiling (Pyroscope, Parca, async-profiler in production)
- **Tools**: Prometheus + Grafana, Datadog, Jaeger, OpenTelemetry, Pyroscope
- **Notes**: RED/USE/Golden Signals cheat sheet, alerting setup guide, tracing instrumentation example

### 10 — Performance Regression Testing
- Running benchmarks in CI/CD — detect perf regressions before they ship
- Establishing baselines, statistical comparison (is this noise or a real regression?)
- Benchmark stability: coefficient of variation, minimum sample size, outlier handling
- GitHub Actions integration for automated benchmark gates
- **Code**: GitHub Action workflow that runs Google Benchmark, compares to baseline, fails on regression

### 10 — Performance Failure Modes
A catalog of the ways production systems die performance-wise. Understanding these patterns is what separates "wrote fast code" from "built a reliable fast system."

- **Lock convoying**: Multiple threads acquire the same lock in sequence. When one thread holds the lock too long, all others queue up. Even after the slow holder releases, the convoy persists — threads wake up one at a time, each acquiring and releasing the lock in sequence. Throughput drops to near-single-threaded.
  - Fix: reduce critical section size, use lock-free structures, or shard the lock
- **Priority inversion**: Low-priority thread holds a lock that high-priority thread needs. High-priority thread blocks. Medium-priority thread preempts the low-priority thread — so the lock is never released. The high-priority thread starves.
  - Fix: priority inheritance (OS automatically boosts the lock holder's priority)
  - Real-world: Mars Pathfinder rover hit this exact bug in 1997 — solved with priority inheritance
- **GC death spiral**: Under sustained load, allocation rate exceeds GC throughput → GC runs more frequently → GC pauses cause request queueing → queued requests allocate more when they resume → even more GC pressure → repeat until OOM or timeout cascade
  - This is specifically why allocation-free Java (Level 7) matters — if you don't allocate on the hot path, GC can't spiral
  - Fix: reduce allocation rate, increase heap, switch to ZGC/Shenandoah, or eliminate GC entirely (C++/Rust)
- **Cache stampede** (covered in topic 04): Popular key expires → 1000 simultaneous cache misses → 1000 database queries → database overload
- **Thundering herd** (covered in topic 03): Event wakes all waiting threads but only one can proceed → N-1 threads wake up and immediately go back to sleep. Wastes CPU, causes latency spike.
  - `accept()` thundering herd: fixed by `SO_REUSEPORT` or `EPOLLEXCLUSIVE`
  - Cache thundering herd: fixed by single-flight / lock-based recomputation
- **Noisy neighbor (cloud)**: Your VM shares physical hardware with others. Neighbor's batch job steals CPU cycles, memory bandwidth, or I/O bandwidth → your latency spikes unpredictably.
  - Fix: dedicated instances, CPU pinning via cgroups, or over-provisioning
- **THP compaction stalls**: Transparent Huge Pages daemon runs compaction in the background to create contiguous 2MB pages → stalls your application for milliseconds
  - Fix: disable THP, use explicit huge pages (Level 5 topic 04)

### 12 — Case Studies
Real-world performance war stories analyzed in detail:

- **LMAX Disruptor**: How LMAX achieved 6 million transactions per second on a single thread
  - Single-writer principle, mechanical sympathy, pre-allocated ring buffer, busy-spin wait

- **HFT Kernel Bypass**: How a trading firm went from 50μs to 5μs with Solarflare OpenOnload
  - Kernel bypass, poll-mode NIC, CPU isolation, huge pages — the full stack

- **Ad Exchange P99**: How an ad exchange cut p99 latency from 40ms to 8ms
  - GC elimination, object pooling, Roaring Bitmaps, connection pooling, off-heap allocation

- **Cascading Failure Post-Mortem**: How a retry storm took down a microservices platform
  - Missing circuit breakers, exponential backoff without jitter, queue buildup

- **Database Compaction Storm**: How LSM tree compaction caused 10x p99 spikes
  - RocksDB tuning, leveled vs tiered compaction, rate limiting compaction I/O

## Key Reading for This Level

- Jeff Dean — "Achieving Rapid Response Times in Large Online Services" (Google, tail at scale paper)
- Gil Tene — "How NOT to Measure Latency" (coordinated omission and percentiles)
- Neil Gunther — "Universal Scalability Law" (USL)
- Martin Thompson — LMAX Disruptor technical paper + talks
- Brendan Gregg — "Systems Performance" (capacity planning + USE method chapters)
- [Aerospike — What is P99 Latency?](https://aerospike.com/blog/what-is-p99-latency/) — Comprehensive overview of latency causes and solutions
- Google SRE Book — Chapter 6 (Monitoring Distributed Systems) — RED/USE/Golden Signals
- Michael Nygard — "Release It!" — Circuit breakers, bulkheads, timeouts, cascading failures
- Martin Kleppmann — "Designing Data-Intensive Applications" (DDIA) — Storage engines, replication, partitioning
- Alex Petrov — "Database Internals" — B-Trees, LSM trees, WAL, compaction

## Mini-Project: Rate Limiter (`12-mini-project-rate-limiter/`)

**Apply topics 03 (network resilience), 07 (high-throughput), and 08 (capacity planning) in one focused build.**

Build a high-performance rate limiter in C++ and Rust that can handle 1M+ checks/sec. This is a core component in every ad server, API gateway, and HFT risk system.

| Algorithm | How it works | Tradeoffs |
|-----------|-------------|-----------|
| **Token bucket** | Tokens added at fixed rate, consumed per request. Bucket has max capacity. | Allows bursts up to bucket size. Simple. Most common. |
| **Sliding window log** | Store timestamp of each request, count requests in last N seconds | Accurate but memory-heavy (stores every timestamp) |
| **Sliding window counter** | Combine fixed window counts with weighted overlap | Memory efficient, slightly less accurate, good enough for production |
| **Leaky bucket** | Requests enter a queue, processed at fixed rate | Smooths bursts but adds latency (queuing) |

**What to build**:
- Token bucket in C++ (atomic counter, minimal overhead) and Rust (AtomicU64)
- Java counter-example: `synchronized` token bucket vs `AtomicLong` version
- Distributed rate limiter: approximate global rate limiting across N instances using local counters + periodic sync
- Benchmark: rate limit check must be <100ns to not dominate request latency

**Measure with**: Google Benchmark / Criterion, `perf stat`, contention under multiple threads
**Produce**: Check latency (must be sub-microsecond), accuracy under burst, throughput comparison

## Mini-Project: Distributed Cache Simulator (`13-mini-project-distributed-cache/`)

**Apply topics 04 (caching), 06 (distributed latency), and 07 (high-throughput) in one focused build.**

Build an in-process distributed cache simulator that demonstrates sharding, replication, and consistency tradeoffs. Not a full Redis clone — a focused experiment on how cache architecture affects latency.

| Component | What it demonstrates |
|-----------|---------------------|
| **Consistent hashing** | How keys are distributed across N cache nodes, what happens when a node joins/leaves |
| **Sharding** | Partition data across nodes — each node holds 1/N of the keyspace |
| **Replication** | Write to primary + replicas — tradeoff: write latency vs read availability |
| **Cache stampede simulation** | Popular key expires → all nodes get hit simultaneously → show mitigation strategies |
| **Hot key detection** | Track access frequency, identify keys that should be replicated more aggressively |

**What to build**:
- Single-node LRU cache (from Level 4 mini-project) as the base
- Consistent hash ring for key distribution
- Simulate network latency between "nodes" (add configurable delay)
- Measure: cache hit latency, miss latency, rebalancing cost when nodes change
- Show: how cross-node latency dominates when cache miss requires remote lookup

**This is a simulation, not a real distributed system** — the focus is on understanding the latency implications of different cache architectures, not building production infrastructure.

**Produce**: Hit rate vs node count, latency breakdown (local hit vs remote hit vs miss), rebalancing cost analysis
