"""Dump Dynamo + FX + Inductor output for a tiny model using depyf.

Run:
    python tiny_model_dump.py --out dump
    # then uncomment the print() line below and re-run with --out dump_broken
    python tiny_model_dump.py --out dump_broken

Open the dump directories and read every file. See ./README.md for what to look for.
"""

from __future__ import annotations

import argparse
import os
import shutil

import torch
import torch.nn as nn

try:
    import depyf
except ImportError:
    raise SystemExit("pip install depyf")


class TinyModel(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden, hidden, bias=False)
        self.w2 = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.w1(x)
        h = torch.relu(h)
        # ----- BREAK LINE: uncomment to add a graph break -----
        # print("activation mean:", h.mean().item())
        # -----------------------------------------------------
        return self.w2(h)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dump", help="Dump directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(0)
    model = TinyModel().to(args.device)
    x = torch.randn(8, 64, device=args.device)

    compiled = torch.compile(model)

    with depyf.prepare_debug(args.out):
        # depyf must wrap the call sites that trigger compilation.
        y = compiled(x)
        # call again so you can see guards passing on the second call (no recompile)
        y = compiled(x)

    # Sanity check the output is reasonable
    print(f"output mean: {y.mean().item():.4f} (device={args.device})")
    print(f"dump written to: {os.path.abspath(args.out)}")
    print("Files to read:")
    for root, _, files in os.walk(args.out):
        for f in sorted(files):
            print(" ", os.path.join(root, f))


if __name__ == "__main__":
    main()
