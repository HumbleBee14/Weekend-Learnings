"""
mini-platform KV-cache-aware router (Python reference).

The interesting bits:
  - Block-hash chain (SHA-256, 16-token blocks) — vLLM 0.11+ default.
  - PrefixStore: hash -> Set[pod].
  - Pod state polled from /kv-blocks endpoints (stand-in for vLLM kv-events stream).
  - Multi-objective scorer: prefix length + load + KV headroom.

Run:
    pip install fastapi uvicorn httpx tiktoken
    python router.py --pods http://pod-a:8000 http://pod-b:8000

Bench:
    python bench.py --shared-prefix 4096 --requests 200 --policy prefix
    python bench.py --shared-prefix 4096 --requests 200 --policy random
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Iterable

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

try:
    import tiktoken
    ENC = tiktoken.get_encoding("cl100k_base")

    def tokenize(text: str) -> list[int]:
        return ENC.encode(text)
except ImportError:  # fallback so the file runs without tiktoken
    def tokenize(text: str) -> list[int]:
        return [ord(c) for c in text]


BLOCK_SIZE = 16


def block_hashes(tokens: list[int], block_size: int = BLOCK_SIZE) -> list[str]:
    """Rolling SHA-256 chain. Each block hash includes the parent."""
    hashes = []
    parent = b""
    for i in range(0, len(tokens), block_size):
        chunk = tokens[i : i + block_size]
        if len(chunk) < block_size:
            break  # don't hash a partial trailing block
        h = hashlib.sha256()
        h.update(parent)
        for t in chunk:
            h.update(t.to_bytes(4, "little", signed=False))
        parent = h.digest()
        hashes.append(parent.hex())
    return hashes


@dataclass
class Pod:
    url: str
    blocks: set[str] = field(default_factory=set)
    inflight: int = 0
    capacity: int = 32
    kv_usage: float = 0.0   # 0..1
    last_seen: float = 0.0


class PrefixStore:
    def __init__(self):
        self.idx: dict[str, set[str]] = {}   # hash -> set(pod_url)

    def add(self, pod_url: str, hashes: Iterable[str]):
        for h in hashes:
            self.idx.setdefault(h, set()).add(pod_url)

    def remove(self, pod_url: str):
        for s in self.idx.values():
            s.discard(pod_url)

    def longest_prefix(self, hashes: list[str]) -> dict[str, int]:
        """Return {pod: matched_block_count} for the longest contiguous prefix per pod."""
        per_pod: dict[str, int] = {}
        candidates = None
        for i, h in enumerate(hashes):
            holders = self.idx.get(h, set())
            if candidates is None:
                candidates = set(holders)
            else:
                candidates &= holders
            if not candidates:
                break
            for p in candidates:
                per_pod[p] = i + 1
        return per_pod


def score(
    pod: Pod, matched: int, total_blocks: int,
    w_p: float = 0.6, w_l: float = 0.3, w_t: float = 0.1,
) -> float:
    prefix_score = matched / max(1, total_blocks)
    load = pod.inflight / max(1, pod.capacity)
    load_score = max(0.0, 1.0 - load)
    kv_score = max(0.0, 1.0 - pod.kv_usage)
    return w_p * prefix_score + w_l * load_score + w_t * kv_score


class Router:
    def __init__(self, pod_urls: list[str], policy: str = "prefix"):
        self.pods: dict[str, Pod] = {u: Pod(u) for u in pod_urls}
        self.store = PrefixStore()
        self.policy = policy
        self._rr_idx = 0

    async def refresh_loop(self, period_s: float = 1.0):
        async with httpx.AsyncClient(timeout=2.0) as client:
            while True:
                await asyncio.gather(*[self._refresh_one(client, p) for p in self.pods.values()])
                await asyncio.sleep(period_s)

    async def _refresh_one(self, client: httpx.AsyncClient, pod: Pod):
        try:
            r = await client.get(f"{pod.url}/kv-blocks")
            data = r.json()
            new_blocks = set(data.get("blocks", []))
            pod.blocks = new_blocks
            pod.kv_usage = float(data.get("kv_usage", 0.0))
            pod.inflight = int(data.get("inflight", 0))
            pod.capacity = int(data.get("capacity", pod.capacity))
            pod.last_seen = time.time()
            # Rebuild index for this pod (cheap for small clusters; use deltas in prod).
            self.store.remove(pod.url)
            self.store.add(pod.url, new_blocks)
        except Exception:
            pass

    def pick(self, prompt: str) -> Pod:
        pods = list(self.pods.values())
        if not pods:
            raise RuntimeError("no pods")

        if self.policy == "random":
            self._rr_idx = (self._rr_idx + 1) % len(pods)
            return pods[self._rr_idx]

        toks = tokenize(prompt)
        hashes = block_hashes(toks)
        matches = self.store.longest_prefix(hashes)

        # All pods get a baseline score (matched=0); prefix winners get more.
        best, best_s = pods[0], -1.0
        for pod in pods:
            m = matches.get(pod.url, 0)
            s = score(pod, m, len(hashes))
            if s > best_s:
                best, best_s = pod, s
        return best


def make_app(router: Router) -> FastAPI:
    app = FastAPI()

    @app.on_event("startup")
    async def _start():
        asyncio.create_task(router.refresh_loop())

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        prompt = "".join(m.get("content", "") for m in body.get("messages", []))
        pod = router.pick(prompt)
        pod.inflight += 1

        async def gen():
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream(
                        "POST", f"{pod.url}/v1/chat/completions",
                        json=body,
                    ) as upstream:
                        async for chunk in upstream.aiter_raw():
                            yield chunk
            finally:
                pod.inflight = max(0, pod.inflight - 1)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"x-mini-platform-pod": pod.url})

    @app.get("/state")
    def state():
        return {
            "policy": router.policy,
            "pods": [
                {
                    "url": p.url, "inflight": p.inflight, "capacity": p.capacity,
                    "kv_usage": p.kv_usage, "blocks": len(p.blocks),
                }
                for p in router.pods.values()
            ],
        }

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pods", nargs="+", required=True)
    p.add_argument("--policy", choices=["prefix", "random"], default="prefix")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    r = Router(args.pods, policy=args.policy)
    uvicorn.run(make_app(r), host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
