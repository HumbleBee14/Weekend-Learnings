"""
Skeleton mini-RLXF loop.

This file is intentionally a sketch — it shows the *shape* of the loop and
where each Level 7 system plugs in. It is not a working trainer; the trainer
side is handed off to your Level 6 FSDP2 setup, and the rollout side talks
to a real vLLM via the OpenAI API.

Wire up:
    - rollout: vLLM serving the current base model on http://vllm:8000
    - reward:  rule-based math grader (reward_rule.py)
    - sync:    NCCL broadcast + engine.update_weights() (weight_sync.py)
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import dataclass

import httpx


@dataclass
class Rollout:
    prompt: str
    answer: str          # ground-truth
    completion: str
    logprobs: list[float]
    reward: float = 0.0
    advantage: float = 0.0


# ---------- 2. ROLLOUT ----------

async def generate_rollouts(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    batch: list[tuple[str, str]],
    n_per_prompt: int = 8,
    max_tokens: int = 512,
) -> list[Rollout]:
    out: list[Rollout] = []

    async def one(prompt, gt):
        rs: list[Rollout] = []
        # vLLM's OpenAI-compatible endpoint: ask for n samples and logprobs.
        r = await client.post(f"{base_url}/v1/completions", json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "n": n_per_prompt,
            "temperature": 0.8,
            "logprobs": 1,
        })
        r.raise_for_status()
        data = r.json()
        for ch in data["choices"]:
            rs.append(Rollout(
                prompt=prompt, answer=gt,
                completion=ch["text"],
                logprobs=ch["logprobs"]["token_logprobs"] or [],
            ))
        return rs

    for prompts in await asyncio.gather(*[one(p, gt) for p, gt in batch]):
        out.extend(prompts)
    return out


# ---------- 3. REWARD (rule-based) ----------

def reward_for(rollout: Rollout) -> float:
    # Replace with reward_rule.grade(rollout.completion, rollout.answer)
    return 1.0 if rollout.answer.strip() in rollout.completion else 0.0


# ---------- 4. ADVANTAGE (GRPO group-normalisation) ----------

def compute_advantages(rollouts: list[Rollout]) -> None:
    """Group by prompt; advantage = (r - mean) / (std + eps)."""
    by_prompt: dict[str, list[Rollout]] = {}
    for r in rollouts:
        by_prompt.setdefault(r.prompt, []).append(r)
    eps = 1e-6
    for group in by_prompt.values():
        rs = [g.reward for g in group]
        m = statistics.mean(rs)
        s = statistics.pstdev(rs)
        for g in group:
            g.advantage = (g.reward - m) / (s + eps)


# ---------- 5. TRAINER STEP (handed off to FSDP2) ----------

async def trainer_step(rollouts: list[Rollout]) -> dict:
    """
    Stub: in real use, push rollouts + advantages to the FSDP2 trainer
    process and await its step result. Return metrics for logging.
    """
    avg_r = statistics.mean(r.reward for r in rollouts) if rollouts else 0.0
    return {"step_reward": round(avg_r, 4), "n": len(rollouts)}


# ---------- 6. WEIGHT SYNC (NCCL + engine.update_weights) ----------

async def maybe_sync_weights(step: int, sync_every: int):
    if step > 0 and step % sync_every == 0:
        # Stub: in real use, call into weight_sync.broadcast_and_update().
        print(f"[step {step}] NCCL broadcast + vLLM engine.update_weights()")
        await asyncio.sleep(0.1)


# ---------- the loop ----------

async def main():
    base_url = "http://localhost:8000"
    model = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    sync_every = 4
    n_steps = 12

    # Tiny example dataset: (prompt, ground_truth_answer).
    dataset = [
        ("Q: 12 + 7 = ?\nA:", "19"),
        ("Q: 8 * 6 = ?\nA:", "48"),
        ("Q: 100 - 37 = ?\nA:", "63"),
    ]

    async with httpx.AsyncClient(timeout=120) as client:
        for step in range(n_steps):
            t0 = time.perf_counter()
            rollouts = await generate_rollouts(client, base_url, model, dataset)
            for r in rollouts:
                r.reward = reward_for(r)
            compute_advantages(rollouts)
            metrics = await trainer_step(rollouts)
            await maybe_sync_weights(step + 1, sync_every)
            metrics.update({"step": step, "wall_s": round(time.perf_counter() - t0, 2)})
            print(json.dumps(metrics))


if __name__ == "__main__":
    asyncio.run(main())
