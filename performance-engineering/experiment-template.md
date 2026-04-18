# Experiment Template

Every experiment in this repo follows this structure. No exceptions. Code without measurement is just code — not engineering.

## Template

```markdown
# Experiment: [Title]

## Hypothesis
What do you expect to happen and why?
> e.g., "SoA layout will have 3-5x fewer L1 cache misses than AoS because
> iterating over one field loads 16 values per cache line instead of 1."

## Setup
- **Hardware**: CPU model, cores, cache sizes, RAM, OS
- **Compiler/Runtime**: GCC version + flags, JDK version + GC, rustc version
- **Environment**: `isolcpus` if used, CPU governor set to `performance`, Docker if used
- **Warm-up**: JMH warmup iterations, Google Benchmark --benchmark_min_time, Criterion sample count

## What We're Measuring
- **Primary metric**: latency (p50/p99/p999) OR throughput (ops/sec) OR hardware counter (cache-misses)
- **Secondary metrics**: memory usage, CPU utilization, IPC, branch-misses
- **Tool**: perf stat / JMH / Google Benchmark / Criterion / HdrHistogram / async-profiler

## Code
- `baseline.java` / `baseline.cpp` / `baseline.rs` — the unoptimized version
- `optimized.java` / `optimized.cpp` / `optimized.rs` — the optimized version
- Same logic, same input, different implementation strategy

## Results

### Raw Numbers
| Variant | p50 | p99 | p999 | ops/sec | cache-misses | IPC |
|---------|-----|-----|------|---------|-------------|-----|
| Java baseline | | | | | | |
| C++ optimized | | | | | | |
| Rust optimized | | | | | | |

### perf stat Output
```
(paste actual perf stat output here)
```

### Flame Graphs
- `flamegraph_java.svg`
- `flamegraph_cpp.svg`
- `flamegraph_rust.svg`

## Analysis
- What happened? Did it match the hypothesis?
- If not, why? What was the actual bottleneck?
- What is the speedup? Break it down by factor (cache, GC, allocation, contention, etc.)
- What surprised you?

## Conclusions
- One paragraph: what we learned and when to apply this technique
```

## Rules

1. **Always measure before and after.** No "I think this is faster." Prove it.
2. **Use the right tool.** JMH for Java, Google Benchmark for C++, Criterion for Rust. Never `System.nanoTime()` in a loop.
3. **Record percentiles, not averages.** p50, p99, p999, max. Averages hide tail latency.
4. **Control the environment.** Set CPU governor to `performance`. Disable turbo boost for consistency. Use `isolcpus` for latency-sensitive tests.
5. **Show the hardware counters.** `perf stat` output tells you WHY something is fast or slow — cache misses, branch misses, IPC.
6. **Include flame graphs.** For anything that takes more than microseconds, a flame graph shows WHERE time goes.
7. **Run enough iterations.** Statistical significance matters. Criterion and JMH handle this automatically. For Google Benchmark, use `--benchmark_repetitions=10`.

## Environment Setup for Reproducible Benchmarks

```bash
# Set CPU governor to performance (disable frequency scaling)
sudo cpupower frequency-set --governor performance

# Disable turbo boost (consistent clock speed)
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# Check NUMA topology
numactl --hardware

# Isolate cores for benchmarking (add to /etc/default/grub GRUB_CMDLINE_LINUX)
# isolcpus=4-7 nohz_full=4-7 rcu_nocbs=4-7

# Docker for reproducible environments (optional)
# Build a benchmark container with pinned compiler versions, same OS, same flags
# docker build -t perf-bench .
# docker run --privileged --cpuset-cpus=4-7 perf-bench ./run_benchmarks.sh
```
