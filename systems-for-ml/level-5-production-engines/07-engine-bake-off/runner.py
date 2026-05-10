"""
07 - The bake-off runner. Drives multiple OpenAI-compatible engines through
identical workloads and writes per-(engine, workload) results.

Inputs:
  - engines.yaml: list of {name, base_url, model_id, quant, label} entries
  - workloads/*.jsonl: each line {prompt, max_tokens}
  - per-engine warmup count, per-workload concurrency

Outputs:
  - results/{engine}_{workload}.json with TTFT/ITL/throughput/p50p95p99

The runner is intentionally engine-agnostic: it only speaks OpenAI HTTP. All
engine-specific startup lives in configs/ shell scripts (start each server
out of band before running this).

Usage:
    python runner.py --engines engines.yaml --out results/

Then `plot.py` reads results/*.json and emits G6-G9.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from openai import AsyncOpenAI

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # fall back to JSON if PyYAML missing


@dataclass
class Engine:
    name: str
    base_url: str
    model_id: str
    quant: str = "unknown"
    label: str = ""


@dataclass
class Workload:
    name: str
    path: str
    concurrency: int = 8
    duration_s: int = 60


@dataclass
class ReqResult:
    ttft_s: float
    itl_s: list[float] = field(default_factory=list)
    n_tokens: int = 0
    total_s: float = 0.0


@dataclass
class RunResult:
    engine: str
    workload: str
    quant: str
    n_requests: int
    wall_s: float
    total_tokens: int
    agg_throughput_tok_s: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    itl_p50_ms: float


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p / 100 * (len(s) - 1)))))
    return s[k]


async def one_request(client: AsyncOpenAI, model: str, prompt: str, max_tokens: int) -> ReqResult:
    t0 = time.perf_counter()
    t_last: float | None = None
    res = ReqResult(ttft_s=0.0)
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
        temperature=0.7,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if not delta or not delta.content:
            continue
        now = time.perf_counter()
        if t_last is None:
            res.ttft_s = now - t0
        else:
            res.itl_s.append(now - t_last)
        t_last = now
        res.n_tokens += 1
    res.total_s = time.perf_counter() - t0
    return res


async def warmup(client: AsyncOpenAI, model: str, n: int = 10) -> None:
    print(f"  warmup x{n} ...")
    for _ in range(n):
        await one_request(client, model, "warm up the engine", 16)


async def drive(engine: Engine, wl: Workload) -> RunResult:
    prompts = [json.loads(line) for line in Path(wl.path).read_text().splitlines() if line.strip()]
    client = AsyncOpenAI(base_url=engine.base_url, api_key="EMPTY")
    await warmup(client, engine.model_id)

    sem = asyncio.Semaphore(wl.concurrency)

    async def bounded(p: dict) -> ReqResult:
        async with sem:
            return await one_request(client, engine.model_id, p["prompt"], p.get("max_tokens", 128))

    # Run for ~duration_s of steady state by repeating the prompt set.
    t_start = time.perf_counter()
    deadline = t_start + wl.duration_s
    results: list[ReqResult] = []
    i = 0
    pending: set[asyncio.Task[ReqResult]] = set()
    while time.perf_counter() < deadline or pending:
        if time.perf_counter() < deadline and len(pending) < wl.concurrency * 2:
            p = prompts[i % len(prompts)]
            pending.add(asyncio.create_task(bounded(p)))
            i += 1
            continue
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for d in done:
            results.append(await d)
    wall = time.perf_counter() - t_start

    ttfts = [r.ttft_s * 1000 for r in results]
    itls = [statistics.median(r.itl_s) * 1000 for r in results if r.itl_s]
    total_tokens = sum(r.n_tokens for r in results)

    return RunResult(
        engine=engine.name,
        workload=wl.name,
        quant=engine.quant,
        n_requests=len(results),
        wall_s=wall,
        total_tokens=total_tokens,
        agg_throughput_tok_s=total_tokens / wall if wall else 0.0,
        ttft_p50_ms=pct(ttfts, 50),
        ttft_p95_ms=pct(ttfts, 95),
        ttft_p99_ms=pct(ttfts, 99),
        itl_p50_ms=pct(itls, 50),
    )


def load_config(path: Path) -> dict:
    text = path.read_text()
    if path.suffix in (".yml", ".yaml") and yaml is not None:
        return yaml.safe_load(text)
    return json.loads(text)


async def main_async(engines_path: Path, workloads_dir: Path, out_dir: Path) -> None:
    cfg = load_config(engines_path)
    engines = [Engine(**e) for e in cfg["engines"]]
    workloads = [Workload(**w) for w in cfg["workloads"]]
    out_dir.mkdir(parents=True, exist_ok=True)

    for e in engines:
        for w in workloads:
            print(f"\n=== {e.name} :: {w.name} (concurrency={w.concurrency}, duration={w.duration_s}s) ===")
            try:
                result = await drive(e, w)
            except Exception as ex:
                print(f"  ERROR: {ex}")
                continue
            out = out_dir / f"{e.name}__{w.name}.json"
            out.write_text(json.dumps(asdict(result), indent=2))
            print(
                f"  -> {result.n_requests} reqs  "
                f"agg {result.agg_throughput_tok_s:.0f} tok/s  "
                f"TTFT p50/p95/p99 {result.ttft_p50_ms:.0f}/{result.ttft_p95_ms:.0f}/{result.ttft_p99_ms:.0f} ms"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="engines.yaml")
    ap.add_argument("--workloads-dir", default="workloads/")
    ap.add_argument("--out", default="results/")
    args = ap.parse_args()
    asyncio.run(main_async(Path(args.engines), Path(args.workloads_dir), Path(args.out)))


if __name__ == "__main__":
    main()
