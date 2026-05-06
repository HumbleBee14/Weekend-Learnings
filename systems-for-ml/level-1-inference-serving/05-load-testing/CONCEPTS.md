# 05 — Load Testing

## Why `asyncio.gather` isn't a load test

The script in topics 03–04 sends N requests and waits for them. That's a *concurrency test*, not a load test. Real production load looks different:

- **Poisson arrivals** — requests come in randomly, not in batches of N
- **Variable prompt length** — 50-token prompts mixed with 5000-token prompts
- **Slow clients** — some requests die mid-stream, or take 30s to read the response
- **Retry storms** — when one request fails, clients hammer the endpoint trying to recover
- **Connection churn** — TCP connections open/close constantly

Real load testing tools simulate all of this. The two standard ones:

- **Locust** — Python, scriptable, easy to start with. The right tool for this curriculum.
- **k6** — Go, faster, JS scripting. What you'll see at infra-heavy companies.

We use Locust here. Both produce the same kind of data; Locust is friendlier.

## What Locust does

Locust runs a fleet of "users." Each user repeats a task: pick a prompt, send a request, wait for the response, sleep a random interval. You configure:

- **Number of users** (concurrent virtual clients)
- **Spawn rate** (how fast to ramp up)
- **Tasks** (Python functions that send requests)
- **Wait time** between tasks (constant, exponential, etc.)

Output: throughput, latency CDF (50/95/99/99.9), failure rate. Real-time charts in a web UI.

## The latency CDF

Cumulative Distribution Function. For each latency value X, what fraction of requests were ≤ X?

```
percentile (%)  latency (ms)
50              1500
75              1700
90              1900
95              2050
99              2400
99.9            5000   ← outliers
```

This is more informative than mean+stddev. The mean lies — a few requests at 30s pull it up dramatically. The 99th percentile tells you what 1% of users are seeing.

If your SLO is "p99 ≤ 2 seconds," look at the p99 row. Don't look at mean.

## Saturation point

The interesting question: at what concurrency does p99 break your SLO?

Method:
1. Start with 1 user. Measure p99.
2. Ramp users: 5, 10, 25, 50, 100. Measure p99 at each.
3. Find the highest user count where p99 still meets the SLO.
4. That's your saturation point — the capacity of one server replica.
5. To handle more load, you need more replicas (autoscaling).

## Failure modes at saturation

When you push past saturation, the server doesn't gracefully slow down — it falls off a cliff. Watch for:

- **Queue overflow** — requests rejected with 503
- **Timeouts** — requests sit so long the client gives up (default httpx timeout: 30s)
- **OOM** — KV cache grows past memory limit, server crashes
- **Memory pressure** — CUDA OOM, server returns 500s

Document which failure mode your server hits *first*. It's never random — there's a real reason your specific bottleneck is your specific bottleneck.

## What you'll produce

**G2 of Project 1**: latency CDF at fixed concurrency (16 users, 5 minutes).

Include the median, p95, p99, p99.9, and failure rate. Note the saturation point and the dominant failure mode.

## Pitfalls

1. **Locust web UI metrics include Locust overhead.** For tight numbers, use `--csv` output and post-process.
2. **One Locust instance is one Python process.** At very high RPS (>5000) you need distributed Locust workers. For this curriculum, single-instance is fine.
3. **`time.sleep(1)` between tasks is not "1 user."** It's "1 user that sends 1 request per second on average." Read the Locust docs on `wait_time` carefully.
4. **Reporting mean latency.** Don't. Always report p50/p95/p99 plus failure rate.
