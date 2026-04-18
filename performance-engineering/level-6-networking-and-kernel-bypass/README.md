# Level 6 — Networking and Kernel Bypass

When the kernel's network stack is too slow, bypass it entirely. The Linux kernel network stack adds 5-15μs of latency per packet (context switches, buffer copies, interrupt handling). For HFT, that's the entire latency budget. For ad tech at scale, it's throughput left on the table.

This level covers TCP tuning (getting the most out of the kernel stack), UDP multicast (how HFT distributes market data), DPDK (bypassing the kernel for raw packet I/O), Aeron (reliable low-latency messaging), and RDMA (bypassing everything including the remote CPU).

## Latency Comparison: Where Networking Time Goes

```
Application → send() syscall ............... 0.5 μs  (user→kernel transition)
Kernel TCP stack processing ................ 2-5 μs  (buffering, checksums, segmentation)
NIC driver + DMA ........................... 1-2 μs  (copy to NIC ring buffer)
Wire time (same datacenter) ................ 5-50 μs (speed of light + switch hops)
NIC driver + DMA (receive) ................. 1-2 μs
Kernel TCP stack processing (receive) ...... 2-5 μs
recv() syscall ............................. 0.5 μs
─────────────────────────────────────────────────────
Total kernel path (one way): ............... ~10-15 μs
Total with DPDK kernel bypass: ............. ~2-5 μs   (skip the kernel entirely)
Total with RDMA: ........................... ~1-2 μs   (skip the remote CPU too)
```

## Topics

### 01 — TCP Tuning
- **`TCP_NODELAY`** (Nagle's algorithm): By default, TCP buffers small packets to combine them into larger ones (reduces header overhead). For latency-sensitive apps, this adds up to 40ms delay. **Always set `TCP_NODELAY` for latency-critical connections.**
  - `setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));`
- **`SO_BUSY_POLL`**: Instead of sleeping until a packet arrives (interrupt-driven), busy-poll the NIC
  - Trades CPU for latency — the thread spins checking for data instead of sleeping
  - Reduces latency by 5-10μs (eliminates interrupt + context switch latency)
  - `setsockopt(fd, SOL_SOCKET, SO_BUSY_POLL, &timeout_us, sizeof(timeout_us));`
  - Kernel-side: `echo 50 > /proc/sys/net/core/busy_read` (enable busy-poll for 50μs)
- **Socket buffer sizes**: `SO_RCVBUF`, `SO_SNDBUF`
  - Default buffers are often too small (128KB) for high-throughput workloads
  - For bulk transfer: set to bandwidth-delay product (e.g., 1Gbps × 1ms RTT = 125KB)
  - For latency: smaller buffers = less queuing delay
  - `sysctl net.core.rmem_max=16777216` — raise the maximum allowed buffer size
- **Congestion control algorithms**:
  - `cubic` (Linux default): loss-based, good for general use
  - `bbr` (Google): model-based, better for high-bandwidth high-latency links (WAN)
  - `reno`: simple, old, rarely optimal
  - Set per-socket: `setsockopt(fd, IPPROTO_TCP, TCP_CONGESTION, "bbr", 3);`
- **Critical sysctl settings for low-latency networking**:
  ```
  net.core.somaxconn = 65535              # max listen backlog
  net.core.netdev_max_backlog = 65536     # max packets queued at NIC before kernel processes
  net.ipv4.tcp_tw_reuse = 1              # reuse TIME_WAIT sockets (essential for high connection rate)
  net.ipv4.tcp_fin_timeout = 10          # faster cleanup of closed connections
  net.core.rmem_max = 16777216           # max receive buffer
  net.core.wmem_max = 16777216           # max send buffer
  net.ipv4.tcp_rmem = 4096 87380 16777216  # min default max receive buffer
  net.ipv4.tcp_wmem = 4096 87380 16777216  # min default max send buffer
  net.ipv4.tcp_slow_start_after_idle = 0  # don't reset congestion window after idle
  ```
- **`SO_REUSEPORT`**: Multiple threads can bind to the same port — kernel load-balances across them. Used for high-throughput TCP servers.
- **Code**: C++ and Rust tuned TCP echo server with all socket options set, `SO_BUSY_POLL` comparison
- **Measure**: `ss -tnpi` to inspect socket state, buffer usage, congestion window
- **Measure**: `tcpdump` / Wireshark to see Nagle buffering vs `TCP_NODELAY` on the wire
- **Measure**: RTT comparison with and without busy-poll enabled

### 02 — UDP Multicast
- **Why HFT uses UDP multicast for market data**:
  - One sender, many receivers — exchange sends once, all subscribers receive simultaneously
  - No TCP overhead: no handshake, no congestion control, no retransmission (at transport layer)
  - Minimal latency: send and forget. Receivers get data in ~1-2μs from the wire.
  - TCP would require the exchange to maintain one connection per subscriber — doesn't scale
- **How multicast works**:
  - Sender sends to a multicast group address (224.0.0.0 – 239.255.255.255)
  - Receivers join the group via IGMP (Internet Group Management Protocol)
  - Network switches/routers replicate packets only to ports with interested receivers
  - No sender overhead — send once, hardware replicates
- **Market data distribution pattern**:
  ```
  Exchange ──multicast──→ [Switch] ──→ Firm A (market data handler)
                                   ──→ Firm B (market data handler)
                                   ──→ Firm C (market data handler)
  ```
- **Reliability**: UDP multicast is unreliable (packets can be lost). Solutions:
  - Sequence numbers in each packet — detect gaps
  - Retransmission request channel (separate TCP connection to request missed packets)
  - Redundant feeds: subscribe to both Feed A and Feed B, take the first arrival (hedged receive)
- **Code**: C++ multicast sender + receiver, Rust multicast receiver via `socket2` crate
- **Measure**: One-way latency with hardware timestamps (PTP) or `SO_TIMESTAMPNS`
- **Measure**: Packet loss rate with sequence number gap detection

### 03 — DPDK Kernel Bypass
- **The ultimate optimization**: Bypass the kernel's entire network stack
  - NIC is unbound from the kernel driver and given to a userspace poll-mode driver (PMD)
  - Your application polls the NIC directly — no interrupts, no syscalls, no context switches, no copies
  - Reduces per-packet latency from ~10-15μs (kernel) to ~1-2μs (DPDK)
- **How it works**:
  1. Bind NIC to DPDK-compatible driver (`vfio-pci` or `igb_uio`)
  2. Initialize EAL (Environment Abstraction Layer) — configures hugepages, cores, memory pools
  3. Set up `rte_mbuf` memory pool — pre-allocated packet buffers in hugepage memory
  4. Poll loop: `rte_eth_rx_burst()` → process packet → `rte_eth_tx_burst()` — no syscalls
  5. Your thread runs in a tight poll loop on a dedicated core — 100% CPU, zero latency
- **Hugepages for DMA**: NIC writes packets directly to hugepage memory via DMA. Hugepages ensure the physical address doesn't change (no page faults, no TLB misses during DMA).
- **Tradeoffs**:
  - Dedicated NIC: DPDK takes over the NIC — it's no longer available to the kernel (no SSH over that NIC)
  - Dedicated core: the poll loop burns 100% CPU. Must use `isolcpus` from Level 5.
  - Complexity: you handle everything yourself — ARP, IP, routing, fragmentation, etc. Or use a DPDK-based TCP stack (like `f-stack` or `mtcp`).
- **Alternatives to DPDK**:
  - **Solarflare OpenOnload**: Kernel bypass without modifying the application — intercepts socket API calls. Easier than DPDK but requires Solarflare/Xilinx NICs.
  - **Solarflare ef_vi**: Even lower-level than OpenOnload — direct NIC access, lowest latency
  - **XDP (eXpress Data Path)**: Run eBPF programs at the NIC driver level — process packets before the kernel stack. Less extreme than DPDK but still significant speedup.
- **Code**: Simple DPDK packet echo in C — receive and retransmit without touching the kernel
- **Setup notes**: NIC binding commands, hugepage allocation, EAL parameters

### 04 — Aeron Messaging
- **Martin Thompson's (creator of Disruptor) messaging library**:
  - Reliable UDP transport with congestion control — reliable like TCP, fast like UDP
  - Shared memory IPC: sub-microsecond latency between processes on the same machine
  - Designed for the financial industry — used in production by trading firms
- **Architecture**:
  - **Media Driver**: Manages transport (UDP, IPC). Runs as a separate process or embedded.
  - **Publications**: Send side. You write to a publication, media driver sends it.
  - **Subscriptions**: Receive side. Media driver delivers, you read from a subscription.
  - **Streams**: Named channels within a publication. Multiplexing.
- **IPC mode vs UDP mode**:
  - IPC: shared memory between processes on same machine — sub-microsecond (~200-500ns)
  - UDP: network transport with reliability — low-microsecond (~5-20μs depending on network)
  - Both: zero-copy on the receive path, pre-allocated buffers, no GC pressure
- **Why Aeron is fast**:
  - Lock-free ring buffers (same design as Disruptor)
  - Pre-allocated memory, no allocation on the hot path
  - Busy-spin option for lowest latency (trades CPU for latency)
  - Zero-copy: receiver reads directly from the media driver's buffer
- **Aeron vs raw TCP vs raw UDP**:
  | Transport | Reliability | Latency (IPC) | Latency (network) | Backpressure |
  |-----------|-------------|---------------|-------------------|-------------|
  | Raw TCP | Yes | N/A | ~15-50μs | Kernel TCP window |
  | Raw UDP | No | N/A | ~5-10μs | None |
  | Aeron IPC | Yes | ~200-500ns | N/A | Built-in |
  | Aeron UDP | Yes | N/A | ~5-20μs | Built-in |
- **Code**: Java Aeron publisher + subscriber (Java is Aeron's primary language — Martin Thompson writes Java)
  - IPC mode: measure latency histogram with HdrHistogram
  - UDP mode: measure throughput and latency
- **Also available**: Aeron has C++ and Rust clients via `aeron-rs` — for integrating with C++/Rust systems
- **Measure**: HdrHistogram latency percentiles (p50, p99, p999) — compare IPC vs UDP vs raw TCP

### 05 — RDMA and InfiniBand
- **Remote Direct Memory Access**: Read or write a remote machine's memory without involving its CPU at all
  - The NIC on machine A directly reads/writes memory on machine B — zero CPU involvement on B
  - Latency: ~1-2μs (compare: TCP over Ethernet = ~15-50μs)
  - Throughput: up to 200 Gbps (InfiniBand HDR)
- **How it works**:
  - Register memory regions with the NIC (pin them in physical memory)
  - Exchange connection info (QP numbers, memory keys)
  - One-sided operations: `RDMA_WRITE` and `RDMA_READ` — remote CPU doesn't even know it happened
  - Two-sided operations: `SEND`/`RECEIVE` — like normal messaging but via RDMA transport
- **Verbs API**: The low-level programming interface for RDMA
  - `ibv_reg_mr()`: register a memory region
  - `ibv_post_send()`: post a send/write/read work request
  - `ibv_poll_cq()`: poll for completion
  - Complex API — much harder to program than sockets
- **InfiniBand vs RoCE (RDMA over Converged Ethernet)**:
  | Feature | InfiniBand | RoCE v2 |
  |---------|-----------|---------|
  | Network | Dedicated InfiniBand fabric | Standard Ethernet |
  | Latency | ~1μs | ~2-3μs |
  | Cost | Expensive (dedicated switches) | Cheaper (Ethernet switches) |
  | Used by | HFT, HPC, supercomputers | Cloud providers, databases |
- **Use cases**:
  - HFT: cross-server communication between matching engine and risk system
  - Distributed databases: ScyllaDB, Oracle RAC use RDMA for inter-node communication
  - HPC: MPI over InfiniBand for scientific computing
  - AI training: GPU-to-GPU communication across machines (NCCL over RDMA)
- **Notes**: Architecture overview, when RDMA makes sense, cost/benefit analysis
  - Hardware required: RDMA-capable NICs (Mellanox/NVIDIA ConnectX series), compatible switches
  - Not practical for hobby projects (requires specialized hardware) — but understanding the architecture deepens your mental model of how low-latency systems work

## The Networking Latency Hierarchy

```
Application code optimization .............. saves nanoseconds
TCP socket tuning (NODELAY, buffers) ....... saves ~5-40μs
Busy-poll (SO_BUSY_POLL) ................... saves ~5-10μs
Kernel bypass (DPDK) ....................... saves ~5-15μs (eliminates kernel stack)
Kernel bypass (OpenOnload) ................. saves ~5-15μs (easier than DPDK, requires Solarflare NIC)
Aeron IPC (same machine) ................... ~200-500ns total
RDMA (cross-machine) ....................... ~1-2μs total
FPGA on NIC ................................ ~100ns total (not covered — hardware engineering territory)
```

## Key Reading for This Level

- DPDK documentation (dpdk.org) — programmer's guide and examples
- Martin Thompson — Aeron design docs and wiki (real-logic/aeron on GitHub)
- Solarflare/Xilinx OpenOnload documentation and performance guides
- "High Performance Browser Networking" — Ilya Grigorik (TCP tuning chapters)
- Linux networking sysctl documentation: `Documentation/networking/ip-sysctl.rst`
- Mellanox/NVIDIA RDMA programming guide (RDMA Aware Networks Programming User Manual)
- "Understanding Linux Network Internals" — Christian Benvenuti (how the kernel stack works — to understand what you're bypassing)

## Mini-Project: Echo Server Shootout (`06-mini-project-echo-server/`)

**Apply everything from this level in one focused build.**

Build the same TCP echo server 5 different ways, benchmark them head-to-head. Same workload (100 concurrent clients, 64-byte messages, measure round-trip latency and throughput).

| Variant | How it works | Expected result |
|---------|-------------|-----------------|
| **Blocking (thread-per-connection)** | `accept()` → spawn thread → `read()` → `write()` in loop | Worst — context switches per connection, doesn't scale past ~1000 connections |
| **epoll (C++)** | Event-driven, single-threaded, non-blocking sockets | Good — handles 10K+ connections, ~10-15μs per echo |
| **io_uring (C++)** | Submission queue batching, zero-copy possible | Better — 2-3x throughput vs epoll at lower latency |
| **Tokio async (Rust)** | `tokio::net::TcpListener`, work-stealing runtime | Comparable to epoll/io_uring, but with safe async/await ergonomics |
| **Java NIO (Netty-style)** | `Selector`-based event loop, `ByteBuffer` | Counter-example: good throughput but GC pressure from ByteBuffer allocation |

**What to measure**:
- Round-trip latency: p50, p99, p999 at 100/1K/10K concurrent connections
- Throughput: messages/sec at saturation
- `perf stat -e context-switches` — blocking server will show millions, event-driven near-zero
- CPU utilization per variant
- Scaling curve: latency vs number of concurrent connections

**Load testing**: Use `wrk2` or a custom client that measures RTT per message with HdrHistogram
**Produce**: Latency vs concurrency graph, throughput comparison table, context switch counts, flame graphs
