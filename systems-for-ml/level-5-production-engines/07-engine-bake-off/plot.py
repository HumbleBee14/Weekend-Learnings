"""
07 - Plot G6-G9 from the bake-off results/ directory.

Reads results/{engine}__{workload}.json files emitted by runner.py and
produces the four required figures for reports/bakeoff.md:

  G6  TTFT bar chart per engine, split by W1 (short) vs W2 (long)
  G7  Throughput (tok/s) per engine on identical workload (agg)
  G8  GPU memory usage vs context length per engine
        (memory numbers must be added to results JSON manually — read from
         /metrics or nvidia-smi during the run; the runner doesn't capture
         them by default)
  G9  Cost per million tokens per engine x quant
        (cost-per-hour comes from configs/instance_costs.yaml; pass via
         --instance-cost-yaml)

Usage:
    pip install matplotlib pandas
    python plot.py --results results/ --out figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import pandas as pd  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:
    print("pip install pandas matplotlib")
    raise SystemExit(1)


def load_results(results_dir: Path) -> "pd.DataFrame":
    rows = []
    for p in results_dir.glob("*.json"):
        rows.append(json.loads(p.read_text()))
    return pd.DataFrame(rows)


def g6_ttft_short_vs_long(df: "pd.DataFrame", out: Path) -> None:
    sub = df[df["workload"].isin(["w1_short", "w2_long"])]
    pivot = sub.pivot(index="engine", columns="workload", values="ttft_p99_ms")
    pivot.plot(kind="bar", figsize=(8, 4))
    plt.ylabel("TTFT p99 (ms)")
    plt.title("G6 — TTFT p99: short (128) vs long (4K) prompts")
    plt.tight_layout()
    plt.savefig(out / "G6_ttft.png", dpi=150)
    plt.close()


def g7_throughput(df: "pd.DataFrame", out: Path) -> None:
    pivot = df.pivot(index="engine", columns="workload", values="agg_throughput_tok_s")
    pivot.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("Aggregate throughput (tok/s)")
    plt.title("G7 — Throughput per engine, per workload")
    plt.tight_layout()
    plt.savefig(out / "G7_throughput.png", dpi=150)
    plt.close()


def g8_memory_vs_context(df: "pd.DataFrame", out: Path) -> None:
    if "gpu_mem_gb" not in df.columns:
        print("G8 skipped: no gpu_mem_gb column. Add memory readings to results JSON manually.")
        return
    sub = df[df["workload"].isin(["w1_short", "w2_long"])]
    pivot = sub.pivot(index="engine", columns="workload", values="gpu_mem_gb")
    pivot.plot(kind="bar", figsize=(8, 4))
    plt.ylabel("Peak GPU memory (GB)")
    plt.title("G8 — GPU memory at short vs long context")
    plt.tight_layout()
    plt.savefig(out / "G8_memory.png", dpi=150)
    plt.close()


def g9_cost_per_mtok(df: "pd.DataFrame", out: Path, cost_yaml: Path | None) -> None:
    if cost_yaml is None or not cost_yaml.exists():
        print("G9 skipped: pass --instance-cost-yaml with $/hr per engine instance.")
        return
    import yaml  # type: ignore

    costs = yaml.safe_load(cost_yaml.read_text())  # {engine_name: usd_per_hour}
    df = df.copy()
    df["usd_per_hr"] = df["engine"].map(costs)
    df["usd_per_mtok"] = (df["usd_per_hr"] / df["agg_throughput_tok_s"]) * 1e6 / 3600
    pivot = df.pivot(index="engine", columns="workload", values="usd_per_mtok")
    pivot.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("USD per million tokens")
    plt.title("G9 — Cost per million tokens per engine x workload")
    plt.tight_layout()
    plt.savefig(out / "G9_cost.png", dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/")
    ap.add_argument("--out", default="figures/")
    ap.add_argument("--instance-cost-yaml", default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_results(Path(args.results))
    if df.empty:
        print("No results found. Run runner.py first.")
        return

    g6_ttft_short_vs_long(df, out)
    g7_throughput(df, out)
    g8_memory_vs_context(df, out)
    g9_cost_per_mtok(df, out, Path(args.instance_cost_yaml) if args.instance_cost_yaml else None)

    print(f"Figures written to {out}/")


if __name__ == "__main__":
    main()
