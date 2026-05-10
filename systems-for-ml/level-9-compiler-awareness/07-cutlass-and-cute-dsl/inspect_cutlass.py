"""Guided reading order for the CUTLASS repository.

Not a benchmark, not a build. This script takes a path to a CUTLASS
checkout and prints the ten files most worth reading first, in order,
with a one-line note on why.

Usage:
    git clone --depth=1 https://github.com/NVIDIA/cutlass.git ~/cutlass
    python inspect_cutlass.py --root ~/cutlass
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List


@dataclass
class ReadingItem:
    relpath: str
    why: str


# Order matters. This is a 1-hour reading plan.
READING_ORDER: List[ReadingItem] = [
    ReadingItem(
        "README.md",
        "Project overview. Skim. Note the supported architectures section.",
    ),
    ReadingItem(
        "media/docs/cute/00_quickstart.md",
        "CuTe in 10 minutes. The Layout / Tensor / TiledMMA vocabulary.",
    ),
    ReadingItem(
        "media/docs/cute/01_layout.md",
        "Layout algebra. Read until 'composition'. The core abstraction.",
    ),
    ReadingItem(
        "media/docs/cute/03_tensor.md",
        "Tensors as Layout + iterator. How CuTe views memory.",
    ),
    ReadingItem(
        "media/docs/cute/0x_gemm_tutorial.md",
        "A GEMM rebuilt from CuTe primitives. The canonical example.",
    ),
    ReadingItem(
        "examples/cute/tutorial/sgemm_sm80.cu",
        "Concrete: an Ampere SGEMM written in CuTe. Read, do not modify.",
    ),
    ReadingItem(
        "examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu",
        "Hopper warp-specialization pattern. The 2024 perf unlock.",
    ),
    ReadingItem(
        "include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp",
        "Production Hopper mainloop. Heavy. Skim only.",
    ),
    ReadingItem(
        "python/README.md",
        "CuTe DSL (Python) status. Where the 2025-2026 work is heading.",
    ),
    ReadingItem(
        "media/docs/dependent_kernel_launch.md",
        "Programmatic Dependent Launch (PDL) — the kernel-launch overlap trick.",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="path to cloned CUTLASS repo")
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"not a directory: {args.root}")

    print(f"CUTLASS reading plan — root: {args.root}\n")
    print("Read these in order. Skip nothing on first pass.\n")

    missing = []
    for i, item in enumerate(READING_ORDER, 1):
        full = os.path.join(args.root, item.relpath)
        exists = os.path.exists(full)
        marker = "[ok] " if exists else "[??] "
        print(f"  {marker}{i:2d}. {item.relpath}")
        print(f"           {item.why}")
        if not exists:
            missing.append(item.relpath)
        print()

    if missing:
        print(
            "Some files were not found. Path layout has changed across CUTLASS\n"
            "versions. Check the repo's current README for renamed docs.\n"
            "Missing:"
        )
        for m in missing:
            print(f"  - {m}")

    print("\nAfter this plan:")
    print("  - Read one production kernel: FlashInfer's prefill or DeepGEMM's main loop.")
    print("  - https://github.com/flashinfer-ai/flashinfer")
    print("  - https://github.com/deepseek-ai/DeepGEMM")


if __name__ == "__main__":
    main()
