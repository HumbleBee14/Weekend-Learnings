"""Probe the host CPU for SME / SME2 / NEON-dot-product support and report
recommended thread counts for llama.cpp on Apple Silicon.

macOS only. Reads sysctl keys exposed by Darwin since macOS 14.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys


SYSCTL_KEYS = [
    "machdep.cpu.brand_string",
    "hw.perflevel0.physicalcpu",   # P-cores
    "hw.perflevel1.physicalcpu",   # E-cores
    "hw.optional.arm.FEAT_SME",
    "hw.optional.arm.FEAT_SME2",
    "hw.optional.arm.FEAT_SVE",
    "hw.optional.arm.FEAT_DotProd",
    "hw.optional.arm.FEAT_I8MM",
    "hw.optional.arm.FEAT_BF16",
    "hw.optional.arm.FEAT_FP16",
]


def sysctl(key: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", key], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except subprocess.CalledProcessError:
        return None


def main() -> int:
    if platform.system() != "Darwin":
        print("This probe targets macOS. Other OSes need /proc/cpuinfo parsing.")
        return 1

    values: dict[str, str | None] = {k: sysctl(k) for k in SYSCTL_KEYS}

    print(f"{'chip':32s}: {values['machdep.cpu.brand_string']}")
    p_cores_raw = values["hw.perflevel0.physicalcpu"]
    e_cores_raw = values["hw.perflevel1.physicalcpu"]
    p_cores = int(p_cores_raw) if p_cores_raw else 0
    e_cores = int(e_cores_raw) if e_cores_raw else 0
    print(f"{'P-cores':32s}: {p_cores}")
    print(f"{'E-cores':32s}: {e_cores}")

    feature_keys = [k for k in SYSCTL_KEYS if k.startswith("hw.optional.arm")]
    for k in feature_keys:
        v = values[k] or "missing"
        short = k.replace("hw.optional.arm.", "")
        print(f"{short:32s}: {v}")

    print()
    print(f"recommended llama.cpp threads (-t)      : {p_cores}")
    print(f"  rationale: P-cores only. Adding E-cores hurts throughput on")
    print(f"  Apple Silicon because the scheduler migrates threads.")

    sme = values.get("hw.optional.arm.FEAT_SME") == "1"
    sme2 = values.get("hw.optional.arm.FEAT_SME2") == "1"
    print()
    if sme2:
        print("SME2 is present. A recent llama.cpp build picks it up automatically")
        print("via ggml/src/ggml-cpu/arch/arm/. Prompt processing should benefit.")
    elif sme:
        print("SME (without SME2) is present. Apple M4 typical configuration.")
    else:
        print("No SME. Likely M1/M2/M3 — falls back to NEON + Apple AMX coprocessor")
        print("through Accelerate. Still fast, just not the SME path.")

    if shutil.which("llama-bench"):
        print()
        print("llama-bench is on PATH. Confirm the runtime feature set with:")
        print("  llama-bench --help | head -n 20")
        print("and check the build banner from any llama.cpp invocation; it lists")
        print("CPU features compiled in.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
