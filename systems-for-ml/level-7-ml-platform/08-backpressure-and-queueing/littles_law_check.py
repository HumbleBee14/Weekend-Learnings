"""
Validate Little's Law against your own metrics.

Pulls L (in-system count), lambda (arrival rate), and W (mean latency)
from a Prometheus endpoint over a window, computes L_pred = lambda * W,
prints the relative error.

Usage:
    python littles_law_check.py \
        --prom http://localhost:9090 \
        --window 5m
"""

import argparse
import sys
from urllib.parse import urlencode
import urllib.request
import json


def query(prom: str, q: str) -> float | None:
    url = f"{prom}/api/v1/query?{urlencode({'query': q})}"
    with urllib.request.urlopen(url, timeout=10) as r:
        data = json.loads(r.read())
    res = data.get("data", {}).get("result", [])
    if not res:
        return None
    val = res[0]["value"][1]
    return float(val)


def check(prom: str, window: str) -> dict:
    # L = mean(running + waiting) over window
    L = query(
        prom,
        f"avg_over_time((sum(vllm:num_requests_running) "
        f"+ sum(vllm:num_requests_waiting))[{window}:5s])",
    )
    # lambda = req/s arrival rate
    lam = query(
        prom,
        f"sum(rate(vllm:e2e_request_latency_seconds_count[{window}]))",
    )
    # W = mean end-to-end latency over window
    W = query(
        prom,
        f"sum(rate(vllm:e2e_request_latency_seconds_sum[{window}])) "
        f"/ sum(rate(vllm:e2e_request_latency_seconds_count[{window}]))",
    )

    if None in (L, lam, W) or lam == 0:
        return {"error": "missing data; ensure vLLM has served traffic in window"}

    L_pred = lam * W
    err = (L_pred - L) / L if L else float("nan")
    return {
        "L_observed": round(L, 4),
        "lambda_rps": round(lam, 4),
        "W_seconds": round(W, 4),
        "L_predicted": round(L_pred, 4),
        "relative_error": round(err, 4),
        "verdict": "PASS" if abs(err) < 0.10 else "INVESTIGATE (>10% drift)",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prom", default="http://localhost:9090")
    p.add_argument("--window", default="5m")
    args = p.parse_args()
    out = check(args.prom, args.window)
    print(json.dumps(out, indent=2))
    if out.get("verdict", "").startswith("INVESTIGATE"):
        sys.exit(2)


if __name__ == "__main__":
    main()
