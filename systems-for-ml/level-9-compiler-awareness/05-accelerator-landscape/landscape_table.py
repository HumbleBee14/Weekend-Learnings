"""Print the 2026 accelerator landscape as a study reference.

Not a benchmark. Just a structured dump of what each compiler stack is
called, what IR it speaks, and where its kernel-author DSL lives.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Accelerator:
    vendor: str
    part: str
    exec_model: str
    primary_compiler: str
    ir_layer: str
    kernel_dsl: str
    open_source: bool


LANDSCAPE: List[Accelerator] = [
    Accelerator("NVIDIA",       "Blackwell B200/B300",  "SIMT + tensor cores",   "CUDA / Inductor",      "PTX (via Triton/LLVM)", "Triton, CUTLASS/CuTe DSL", True),
    Accelerator("AMD",          "MI300X / MI350X",      "SIMT + matrix cores",   "ROCm / Inductor",      "AMDGCN (via Triton/LLVM)", "Triton-AMD, Composable Kernel", True),
    Accelerator("Google",       "TPU v5p / v6 / v7",    "Systolic + VPU",        "XLA",                  "StableHLO / HLO",       "Pallas",                    True),
    Accelerator("AWS",          "Trainium2",            "Systolic + scalar",     "Neuron SDK",           "StableHLO / HLO (XLA-derived)", "NKI",               False),
    Accelerator("Groq",         "LPU",                  "Static dataflow",       "GroqWare (closed)",    "internal",              "(none public)",             False),
    Accelerator("Cerebras",     "CS-3 / CS-4",          "Wafer-scale dataflow",  "Cerebras SDK",         "internal",              "Cerebras SDK kernels",      False),
    Accelerator("Tenstorrent",  "Wormhole / Blackhole", "Tensix grid",           "TT-Buda / TT-NN",      "TT-MLIR",               "TT-Metal",                  True),
    Accelerator("SambaNova",    "SN40L",                "Reconfigurable dataflow","SambaFlow",           "MLIR-based",            "(limited public)",          False),
    Accelerator("Modular",      "(targets all above)",  "(varies)",              "MAX engine",           "MLIR / Mojo IR",        "Mojo",                      True),
]


FIVE_QUESTIONS = [
    "1. What is the execution model? (SIMT, systolic, dataflow, wafer, Tensix)",
    "2. Where does the user write kernels? (DSL name, or 'none — graph compiler does it')",
    "3. What IR does the high-level frontend lower to? (Triton, StableHLO, custom)",
    "4. Who controls scheduling — runtime hardware, or compiler?",
    "5. What is the on-chip memory hierarchy, and who places data in it?",
]


def print_table() -> None:
    headers = ["Vendor", "Part", "Exec model", "Primary compiler", "IR layer", "Kernel DSL", "OSS"]
    rows = [
        [a.vendor, a.part, a.exec_model, a.primary_compiler, a.ir_layer, a.kernel_dsl, "yes" if a.open_source else "no"]
        for a in LANDSCAPE
    ]
    widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = " | "

    def fmt(row):
        return sep.join(c.ljust(widths[i]) for i, c in enumerate(row))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(fmt(r))


def print_five_questions() -> None:
    print("\nThe five questions to ask of any new accelerator:\n")
    for q in FIVE_QUESTIONS:
        print("  " + q)


if __name__ == "__main__":
    print_table()
    print_five_questions()
    print(
        "\nReferences:\n"
        "  Triton          https://triton-lang.org/\n"
        "  StableHLO       https://openxla.org/stablehlo\n"
        "  Pallas          https://docs.jax.dev/en/latest/pallas/\n"
        "  AWS NKI         https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/\n"
        "  TT-Metal        https://github.com/tenstorrent/tt-metal\n"
        "  Mojo / MAX      https://www.modular.com/mojo\n"
    )
