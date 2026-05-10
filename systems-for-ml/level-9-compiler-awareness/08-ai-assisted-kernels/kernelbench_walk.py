"""Walk a local KernelBench checkout and summarize the problem set.

Not a runner. Just a structural look at what the benchmark covers, so
you can decide which problems to read or attempt without scrolling
through the whole repo.

Usage:
    git clone --depth=1 https://github.com/ScalingIntelligence/KernelBench.git ~/kernelbench
    python kernelbench_walk.py --root ~/kernelbench
"""

import argparse
import os
import sys
from collections import defaultdict


def walk(root: str) -> None:
    levels = ["level1", "level2", "level3"]
    by_level = defaultdict(list)

    # The repo layout is roughly KernelBench/level{1,2,3}/<problem_name>/<files>.
    candidates = [
        os.path.join(root, "KernelBench"),
        root,
    ]
    base = next((c for c in candidates if os.path.isdir(os.path.join(c, "level1"))), None)
    if base is None:
        sys.exit("could not find KernelBench/level1 under the given root")

    for lv in levels:
        lv_path = os.path.join(base, lv)
        if not os.path.isdir(lv_path):
            continue
        for entry in sorted(os.listdir(lv_path)):
            full = os.path.join(lv_path, entry)
            if os.path.isdir(full):
                by_level[lv].append(entry)

    print("KernelBench problem inventory")
    print("=============================\n")
    descriptions = {
        "level1": "Single-op kernels. Elementwise, reductions, simple matmul.",
        "level2": "Fused or compound ops. Layernorm, softmax+matmul, attention pieces.",
        "level3": "Whole-architecture kernels. Full attention, full MLP blocks.",
    }
    for lv in levels:
        items = by_level.get(lv, [])
        print(f"{lv}  ({len(items)} problems)")
        print(f"  {descriptions.get(lv, '')}")
        for name in items[:8]:
            print(f"    - {name}")
        if len(items) > 8:
            print(f"    ... and {len(items) - 8} more")
        print()

    print("Reading suggestion:")
    print("  Pick three problems from level1, two from level2, one from level3.")
    print("  Open the reference PyTorch op + the test harness for each.")
    print("  That's enough to understand what an LLM has to produce to score on the bench.\n")
    print("Paper: https://arxiv.org/abs/2502.10517")
    print("Repo : https://github.com/ScalingIntelligence/KernelBench")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="path to cloned KernelBench repo")
    args = parser.parse_args()
    if not os.path.isdir(args.root):
        sys.exit(f"not a directory: {args.root}")
    walk(args.root)


if __name__ == "__main__":
    main()
