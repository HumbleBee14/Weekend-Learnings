# Level 5 — OS and Kernel Tuning

When userspace optimization isn't enough, squeeze the operating system. This level covers syscall overhead, io_uring, CPU isolation, huge pages in practice, and real-time scheduling — the Linux kernel knobs that HFT firms and database vendors tune obsessively.

## Topics

### 01 — Syscall Overhead
- What a syscall costs: user→kernel transition, register save/restore (~500ns)
- vDSO: `clock_gettime()` via vDSO is ~50ns (no kernel entry). Same call via real syscall is ~500ns.
- Batching: why io_uring and DPDK exist — amortize or eliminate syscall overhead
- `strace`: trace syscalls of any program, find unexpected syscall storms
- **Code**: C++ measuring vDSO vs real syscall cost. `strace` analysis of a program.

### 02 — io_uring
- Linux's modern async I/O: Submission Queue (SQ) + Completion Queue (CQ) in shared memory
- Batched submissions: submit 32 operations with one syscall (or zero with SQPOLL)
- io_uring vs epoll: 2-3x throughput improvement at lower latency
- **Code**: epoll echo server (C++) vs io_uring echo server (C++) vs Rust io_uring — benchmark both

### 03 — CPU Isolation and Pinning
- `isolcpus`: remove CPUs from the scheduler — dedicate them to your hot path
- `taskset` / `pthread_setaffinity_np`: pin a thread to a specific core
- `nohz_full`: disable timer interrupts on isolated cores (no scheduling jitter)
- `rcu_nocbs`: move RCU callbacks off your isolated cores
- Java: OpenHFT Java-Thread-Affinity library (the only way to pin from JVM)
- **Code**: C++ thread pinning demo — measure scheduling jitter before/after isolation

### 04 — Huge Pages in Practice
- Transparent Huge Pages (THP): automatic but causes compaction latency spikes
- Explicit huge pages: `mmap MAP_HUGETLB`, hugetlbfs mount — deterministic, no spikes
- HFT rule: always explicit huge pages, never THP
- **Code**: C++ mmap with MAP_HUGETLB, Java `-XX:+UseLargePages` comparison

### 05 — Real-Time Scheduling
- `SCHED_FIFO`: fixed-priority scheduling — your thread never gets preempted by lower-priority
- `SCHED_RR`: round-robin among same-priority RT threads
- `mlockall()`: prevent page faults by locking all memory into RAM
- RT kernel patches (`PREEMPT_RT`): make the entire kernel preemptible
- **Code**: SCHED_FIFO thread with mlockall — measure scheduling jitter vs default CFS

## Key Reading for This Level

- Brendan Gregg — "Systems Performance" (CPU, Memory, I/O chapters)
- Brendan Gregg — "BPF Performance Tools" (eBPF/bpftrace chapters)
- Jens Axboe — io_uring documentation and liburing examples
- Linux kernel documentation: `Documentation/admin-guide/kernel-parameters.txt` (isolcpus, nohz_full)
- "The Linux Programming Interface" — Michael Kerrisk (syscall chapters)

## Mini-Project: High-Throughput Logger (`06-mini-project-high-throughput-logger/`)

**Apply everything from this level in one focused build.**

Build a structured logger that can handle 1M+ log entries/sec without blocking the application thread. The hot path (logging call) must be sub-microsecond. Compare Java, C++, and Rust implementations.

| Component | What it exercises | Technique from this level |
|-----------|------------------|--------------------------|
| **Hot path (log call)** | Write to ring buffer, return immediately — never block the caller | Lock-free ring buffer (Level 3), zero allocation |
| **Background writer thread** | Drain ring buffer, batch-write to disk | Dedicated thread on isolated core (`isolcpus`, topic 03) |
| **Disk I/O** | Buffered writes, `O_DIRECT` option, fsync strategy | Syscall overhead (topic 01), io_uring for async writes (topic 02) |
| **Memory** | Log entries on huge pages, pre-allocated buffers | Huge pages (topic 04), arena allocation (Level 4) |
| **Formatting** | Deferred formatting — don't format the string on the hot path | Format on the writer thread, not the caller thread |

**Java counter-example**: `java.util.logging` or Log4j2 — show that even Log4j2 AsyncAppender has measurable overhead from GC pressure and thread coordination. The C++/Rust version with io_uring and huge pages will have significantly lower tail latency.

**Variants to benchmark**:
- Synchronous write (baseline — terrible, blocks caller)
- Async with `write()` syscalls (good, but syscall overhead)
- Async with `io_uring` (best — batched, minimal syscalls)
- Async with `O_DIRECT` + `io_uring` (skip page cache — most predictable)

**Measure with**: Hot path latency (must be <1μs p99), throughput (entries/sec), disk I/O pattern (syscall count via strace)
**Produce**: Hot path latency histogram, throughput comparison, strace syscall count comparison, flame graphs
