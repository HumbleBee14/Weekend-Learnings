"""
Example reward function for GRPO on math prompts.

mlx-tune (and similar libraries) load this file and call score(prompt, completion).
Return a float; higher is better. Sparse 0/1 rewards work but high variance —
prefer denser shaping where possible.
"""
from __future__ import annotations
import re


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_FINAL = re.compile(r"(?:answer|final|=)\s*[:=]?\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def _extract(text: str) -> float | None:
    m = _FINAL.search(text)
    if m:
        return float(m.group(1))
    nums = _NUMBER.findall(text)
    return float(nums[-1]) if nums else None


def _expected(prompt: str) -> float | None:
    """Prompts in this demo carry their answer like '(answer=42)'."""
    m = re.search(r"\(answer=(-?\d+(?:\.\d+)?)\)", prompt)
    return float(m.group(1)) if m else None


def score(prompt: str, completion: str) -> float:
    target = _expected(prompt)
    pred = _extract(completion)
    if target is None or pred is None:
        # Mild penalty for unparseable to discourage non-answers.
        return -0.1
    if abs(pred - target) < 1e-6:
        return 1.0
    # Soft credit for being close (helps gradient signal early in training).
    err = abs(pred - target) / (abs(target) + 1.0)
    return max(-0.5, 1.0 - 2.0 * err)
