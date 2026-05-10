"""Export a tiny PyTorch model to StableHLO via iree-turbine, then compile it.

Two artifacts emerge:
  model.mlir  — human-readable StableHLO. Open it.
  model.vmfb  — IREE compiled artifact for the chosen backend.

Run:
    pip install iree-turbine torch
    python import_and_inspect.py [--device cpu|vulkan|cuda|metal]
"""

import argparse
import os
import sys

import torch
import torch.nn as nn


class TinyBlock(nn.Module):
    """Two linears with a silu in between. Small enough to read the IR."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.up = nn.Linear(dim, dim * 2, bias=False)
        self.down = nn.Linear(dim * 2, dim, bias=False)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.up(x)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "vulkan", "cuda", "metal", "rocm"])
    parser.add_argument("--mlir-out", default="model.mlir")
    parser.add_argument("--vmfb-out", default="model.vmfb")
    args = parser.parse_args()

    try:
        import iree.turbine.aot as aot
    except ImportError:
        sys.exit("iree-turbine is not installed. `pip install iree-turbine`.")

    model = TinyBlock().eval()
    example = torch.randn(4, 16)

    # Sanity: eager forward should produce a (4, 16) tensor.
    out = model(example)
    assert out.shape == (4, 16), out.shape

    print("Exporting to StableHLO via iree-turbine ...")
    exported = aot.export(model, example)

    # Dump the StableHLO MLIR. Open this — it's the portable IR that
    # is the input to IREE's lowering pipeline.
    exported.print_readable()  # to stdout
    exported.save_mlir(args.mlir_out)
    print(f"  wrote {args.mlir_out}")

    # Compile to a backend-specific artifact. The artifact format is the
    # same (.vmfb, IREE bytecode + per-target kernel blobs); only the
    # embedded kernels differ per device.
    print(f"\nCompiling for device={args.device} ...")
    compile_kwargs = {"save_to": args.vmfb_out}
    if args.device != "cpu":
        # IREE's compile() accepts target backends via target_backends.
        # Names: llvm-cpu, vulkan-spirv, cuda, metal-spirv, rocm.
        backend_map = {
            "vulkan": "vulkan-spirv",
            "cuda":   "cuda",
            "metal":  "metal-spirv",
            "rocm":   "rocm",
        }
        compile_kwargs["target_backends"] = [backend_map[args.device]]
    else:
        compile_kwargs["target_backends"] = ["llvm-cpu"]

    exported.compile(**compile_kwargs)
    size = os.path.getsize(args.vmfb_out)
    print(f"  wrote {args.vmfb_out}  ({size} bytes)")

    print(
        "\nNext steps:\n"
        f"  cat {args.mlir_out}                     # read the StableHLO\n"
        f"  iree-dump-module {args.vmfb_out}        # inspect compiled kernels\n"
        "  iree-run-module --module=model.vmfb \\   \n"
        "    --function=main --input='4x16xf32=...'\n"
    )


if __name__ == "__main__":
    main()
