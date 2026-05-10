"""Plan a pipeline-parallel shard of a transformer LLM across N Macs.

Given a model name, target quant, and per-Mac RAM, prints a viable shard plan
or a clear failure reason. Also estimates per-token activation traffic over
Thunderbolt 5 so you can tell whether the link is the bottleneck.

This is a planner, not a runtime. Useful before paying for hardware or
spending an hour debugging an exo cluster that was never going to fit.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass


@dataclass
class ModelSpec:
    name: str
    params_b: float       # billions of parameters (total, not active)
    n_layers: int
    hidden: int
    is_moe: bool = False
    active_params_b: float | None = None  # for MoE


# A handful of widely-shared 2026-era models. Add your own.
MODELS: dict[str, ModelSpec] = {
    "llama-3.1-70b":   ModelSpec("llama-3.1-70b",   70.6,  80, 8192),
    "llama-3.1-405b":  ModelSpec("llama-3.1-405b", 405.8, 126, 16384),
    "llama-4-scout":   ModelSpec("llama-4-scout",  109.0,  48,  6144,
                                  is_moe=True, active_params_b=17.0),
    "qwen3-next-80b":  ModelSpec("qwen3-next-80b",  80.0,  64,  6144,
                                  is_moe=True, active_params_b=3.0),
    "deepseek-v3":     ModelSpec("deepseek-v3",    671.0,  61,  7168,
                                  is_moe=True, active_params_b=37.0),
}


# Bytes per param at a given quant level. Approximate — real quantized models
# include scale and zero-point overhead per group, ~2-5% on top.
def bytes_per_param(quant_bits: int) -> float:
    return quant_bits / 8.0 * 1.04   # 4% overhead


# Assume the OS, KV cache, runtime overhead want ~15% of each Mac's RAM free.
USABLE_RAM_FRACTION = 0.85

TB5_GBPS = 80.0   # gigabits per second, full duplex; we treat one direction


def plan(model: ModelSpec, quant: int, mac_ram_gb: list[int]) -> int:
    weight_bytes = model.params_b * 1e9 * bytes_per_param(quant)
    weight_gb = weight_bytes / 1e9

    print(f"model: {model.name}")
    print(f"  total params : {model.params_b:.1f} B")
    if model.is_moe:
        print(f"  active params: {model.active_params_b:.1f} B (MoE)")
    print(f"  layers       : {model.n_layers}")
    print(f"  hidden dim   : {model.hidden}")
    print(f"  weights @ {quant}-bit: ~{weight_gb:.1f} GB")
    print()

    n_macs = len(mac_ram_gb)
    total_ram = sum(mac_ram_gb)
    usable_total = total_ram * USABLE_RAM_FRACTION
    print(f"macs available: {n_macs} with {mac_ram_gb} GB RAM "
          f"({total_ram} GB total, ~{usable_total:.0f} GB usable)")
    print()

    if weight_gb > usable_total:
        print(f"VERDICT: does not fit. need ~{weight_gb:.0f} GB, "
              f"have ~{usable_total:.0f} GB usable.")
        print(f"  reduce quant (try {quant - 1}-bit) or add Macs.")
        return 1

    # Even split across Macs. In real life mlx.distributed and exo do
    # weighted splits when RAM is heterogeneous; we keep it simple here
    # and assume homogeneous Macs for the math.
    layers_per_mac = model.n_layers // n_macs
    remainder = model.n_layers % n_macs
    weight_per_mac = weight_gb / n_macs

    biggest_mac = max(mac_ram_gb)
    biggest_usable = biggest_mac * USABLE_RAM_FRACTION

    if weight_per_mac > biggest_usable:
        print(f"VERDICT: tight or infeasible.")
        print(f"  layers/mac : {layers_per_mac} (+{remainder} on the head)")
        print(f"  weight memory/mac: ~{weight_per_mac:.1f} GB"
              f"  <-- exceeds the largest Mac's ~{biggest_usable:.0f} GB usable")
        print(f"  fix: use {quant - 1}-bit, add a Mac, or use larger-RAM Macs.")
        ret = 1
    else:
        print(f"VERDICT: viable.")
        print(f"  layers/mac : {layers_per_mac} (+{remainder} on the head)")
        print(f"  weight memory/mac: ~{weight_per_mac:.1f} GB")
        ret = 0

    # activation traffic per layer-boundary hop
    # prefill: [batch=1, seq=512, hidden] fp16
    seq_prefill = 512
    bytes_prefill = 2 * seq_prefill * model.hidden
    bytes_decode = 2 * 1 * model.hidden
    ms_prefill = (bytes_prefill * 8) / (TB5_GBPS * 1e9) * 1000
    us_decode = (bytes_decode * 8) / (TB5_GBPS * 1e9) * 1e6

    print()
    print("per-token activation traffic over a single TB5 hop:")
    print(f"  prefill (seq={seq_prefill}, hidden={model.hidden}, fp16): "
          f"{bytes_prefill/1e6:.1f} MB / hop  (~{ms_prefill:.1f} ms @ TB5 80 Gb/s)")
    print(f"  decode (seq=1):                        "
          f"{bytes_decode/1e3:.1f} KB / hop  (~{us_decode:.0f} us latency-bound)")
    print()
    print(f"with {n_macs - 1} hops in a {n_macs}-Mac pipeline, decode adds "
          f"~{us_decode * (n_macs - 1):.0f} us per token.")

    return ret


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS),
                    help="known model name")
    ap.add_argument("--quant", type=int, default=4,
                    help="quant bits per weight (2/3/4/6/8/16)")
    ap.add_argument("--macs", type=int, nargs="+", required=True,
                    help="RAM in GB for each Mac, e.g. --macs 64 64 64")
    args = ap.parse_args()

    return plan(MODELS[args.model], args.quant, args.macs)


if __name__ == "__main__":
    sys.exit(main())
