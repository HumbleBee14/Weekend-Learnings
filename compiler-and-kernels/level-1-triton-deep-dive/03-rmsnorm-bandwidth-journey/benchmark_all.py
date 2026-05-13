"""
Run all five versions plus eager, torch.compile, and Liger-Kernel (if installed).
Produce one table comparing them on the same shape.

Usage:
  python benchmark_all.py

If you don't have liger-kernel installed:
  pip install liger-kernel
"""

import importlib.util, os, sys, torch, triton

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


v01 = _load("v01", "01_naive.py")
v02 = _load("v02", "02_vectorized.py")
v03 = _load("v03", "03_single_pass.py")
v04 = _load("v04", "04_autotuned.py")
v05 = _load("v05", "05_persistent.py")


def reference(x, w, eps=1e-6):
    x_fp32 = x.to(torch.float32)
    rms = torch.sqrt((x_fp32 * x_fp32).mean(dim=-1, keepdim=True) + eps)
    return ((x_fp32 / rms) * w.to(torch.float32)).to(x.dtype)


def eager_pt(x, w, eps=1e-6):
    return reference(x, w, eps)


@torch.compile(mode="max-autotune", fullgraph=True)
def compiled_pt(x, w, eps=1e-6):
    return reference(x, w, eps)


def try_liger(x, w):
    try:
        from liger_kernel.transformers.rms_norm import LigerRMSNorm
    except ImportError:
        return None
    n_cols = x.shape[1]
    layer = LigerRMSNorm(n_cols, eps=1e-6).to(x.device).to(x.dtype)
    with torch.no_grad():
        layer.weight.copy_(w)
    return lambda: layer(x)


def main():
    torch.manual_seed(0)
    n_rows, n_cols = 4096, 4096
    dtype = torch.float16
    x = torch.randn(n_rows, n_cols, device="cuda", dtype=dtype) * 0.1
    w = torch.randn(n_cols, device="cuda", dtype=dtype) * 0.5 + 1.0

    # Theoretical-minimum bytes moved (single-pass): load x + load w + store out.
    min_bytes = (n_rows * n_cols + n_cols + n_rows * n_cols) * x.element_size()
    peak_bw_gbs = None
    try:
        # Rough GPU peak HBM bandwidth lookup — adjust the dict for your hardware.
        name = torch.cuda.get_device_name(0).lower()
        peak_map = {
            "t4": 320,
            "a100": 1555 if "40gb" in name or "80gb" not in name else 2039,
            "h100": 3350,
            "h200": 4800,
            "b200": 8000,
            "rtx 4090": 1008,
            "rtx 5090": 1792,
            "mi300x": 5300,
        }
        for k, v in peak_map.items():
            if k in name:
                peak_bw_gbs = v
                break
    except Exception:
        pass

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Shape: ({n_rows}, {n_cols}) fp16")
    if peak_bw_gbs:
        print(f"Approx HBM peak: {peak_bw_gbs} GB/s")
    print()

    # Warm up torch.compile outside timing
    _ = compiled_pt(x, w)

    impls = [
        ("01 naive (2-pass)",   lambda: v01.rmsnorm_naive(x, w)),
        ("02 vectorized",       lambda: v02.rmsnorm_vec(x, w)),
        ("03 single-pass",      lambda: v03.rmsnorm_1pass(x, w)),
        ("04 autotuned",        lambda: v04.rmsnorm_auto(x, w)),
        ("05 persistent",       lambda: v05.rmsnorm_persistent(x, w)),
        ("eager torch",         lambda: eager_pt(x, w)),
        ("torch.compile",       lambda: compiled_pt(x, w)),
    ]

    liger_fn = try_liger(x, w)
    if liger_fn is not None:
        impls.append(("liger-kernel", liger_fn))
    else:
        print("(liger-kernel not installed; skipping. Install with: pip install liger-kernel)")

    print(f"{'impl':<25}{'ms':>10}{'GB/s':>12}{'% peak':>10}")
    print("-" * 60)
    for name, fn in impls:
        ms = triton.testing.do_bench(fn)
        gbps = min_bytes / (ms * 1e-3) / 1e9
        pct = f"{gbps / peak_bw_gbs * 100:.0f}%" if peak_bw_gbs else "—"
        print(f"{name:<25}{ms:>10.3f}{gbps:>12.1f}{pct:>10}")

    print()
    print("Your steps 01 → 05 should show a clear upward staircase.")
    print("Liger-Kernel should be roughly at parity with your step 05.")
    print("If you beat Liger, double-check (warmup? same dtype? same shape? correctness?).")


if __name__ == "__main__":
    main()
