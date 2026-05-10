"""
Synthetic warmup. Run after the engine reports ready but before live traffic.
Triggers CUDA graph capture across multiple shapes so live requests don't
pay autotune cost.

    python warmup_requests.py --base http://localhost:8000 \\
        --model meta-llama/Llama-3.2-1B-Instruct
"""

import argparse
import time
import httpx


SHAPES = [
    (64, 8),
    (256, 8),
    (1024, 8),
    (4096, 8),
    (8192, 8),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://localhost:8000")
    p.add_argument("--model", required=True)
    args = p.parse_args()

    with httpx.Client(timeout=120) as c:
        for in_len, out_len in SHAPES:
            t0 = time.perf_counter()
            r = c.post(f"{args.base}/v1/completions", json={
                "model": args.model,
                "prompt": "x " * in_len,   # crude but enough to exercise graphs
                "max_tokens": out_len,
                "temperature": 0.0,
            })
            r.raise_for_status()
            print(f"warmup in={in_len:>5} out={out_len:>3} dt={time.perf_counter()-t0:.2f}s")


if __name__ == "__main__":
    main()
