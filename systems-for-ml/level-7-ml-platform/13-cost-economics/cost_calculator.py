"""
$/Mtok calculator and (engine x quant x hardware) cost matrix builder.

Inputs (per cell):
    - hardware $/hr
    - input tokens/s, output tokens/s (from your Project 2 bake-off)
    - warm-pool replicas (cost of idle)
    - KV tier $/hr (Redis/Mooncake amortised)
    - observability $/hr (amortised)

Outputs:
    $/Mtok_input, $/Mtok_output, total $/Mtok at a given input/output ratio.

Run:
    python cost_calculator.py --config matrix.yaml
"""

import argparse
import json
import sys
from dataclasses import dataclass

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


@dataclass
class Cell:
    name: str
    engine: str
    quant: str
    hardware: str
    gpu_dollar_per_hour: float
    input_tok_per_s: float
    output_tok_per_s: float
    warm_pool_replicas: float = 0.0    # extra replicas you keep idle
    kv_tier_dollar_per_hour: float = 0.0
    obs_dollar_per_hour: float = 0.0


def cost_per_mtok(cell: Cell, input_share: float = 0.7) -> dict:
    sec_per_hour = 3600
    input_per_hour = cell.input_tok_per_s * sec_per_hour
    output_per_hour = cell.output_tok_per_s * sec_per_hour

    # Active replica cost.
    active_hourly = (
        cell.gpu_dollar_per_hour
        + cell.kv_tier_dollar_per_hour
        + cell.obs_dollar_per_hour
    )
    # Warm pool overhead (idle replicas).
    warm_overhead = cell.warm_pool_replicas * cell.gpu_dollar_per_hour

    # $/Mtok separated for input and output.
    if input_per_hour > 0:
        usd_per_mtok_in = (active_hourly + warm_overhead) / (input_per_hour / 1_000_000)
    else:
        usd_per_mtok_in = float("inf")
    if output_per_hour > 0:
        usd_per_mtok_out = (active_hourly + warm_overhead) / (output_per_hour / 1_000_000)
    else:
        usd_per_mtok_out = float("inf")

    blended = input_share * usd_per_mtok_in + (1 - input_share) * usd_per_mtok_out

    return {
        "cell": cell.name,
        "engine": cell.engine,
        "quant": cell.quant,
        "hardware": cell.hardware,
        "usd_per_mtok_input":  round(usd_per_mtok_in,  4),
        "usd_per_mtok_output": round(usd_per_mtok_out, 4),
        "usd_per_mtok_blended": round(blended, 4),
        "warm_pool_share_pct":
            round(100 * warm_overhead / (active_hourly + warm_overhead), 1)
            if (active_hourly + warm_overhead) > 0 else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--input-share", type=float, default=0.7,
                   help="Share of total tokens that are input. Defaults to 0.7.")
    args = p.parse_args()

    text = open(args.config).read()
    if yaml is not None:
        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)

    rows = []
    for c in cfg["cells"]:
        rows.append(cost_per_mtok(Cell(**c), input_share=args.input_share))

    rows.sort(key=lambda r: r["usd_per_mtok_blended"])
    fmt = ("{cell:<22}  {engine:<10}  {quant:<8}  {hardware:<10}  "
           "in=${usd_per_mtok_input:>7.4f}  out=${usd_per_mtok_output:>7.4f}  "
           "blend=${usd_per_mtok_blended:>7.4f}  warm={warm_pool_share_pct:>4.1f}%")
    print(fmt.replace(":>7.4f", ":>7s").replace(":>4.1f", ":>4s")
              .format(cell="cell", engine="engine", quant="quant", hardware="hw",
                      usd_per_mtok_input="$/Mtok_in",
                      usd_per_mtok_output="$/Mtok_out",
                      usd_per_mtok_blended="blend",
                      warm_pool_share_pct="warm%"))
    print("-" * 110)
    for r in rows:
        print(fmt.format(**r))


if __name__ == "__main__":
    main()
