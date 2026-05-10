"""
13 - Benchmark a small embedding model across runtimes.

Compares:
    - PyTorch eager
    - PyTorch + torch.compile
    - ONNX Runtime (CPU EP)
    - ONNX Runtime (CUDA EP)         — if a CUDA GPU is present
    - ONNX Runtime (TensorRT EP)     — if TRT EP is installed

Same model, same inputs, same batch sizes. Reports embeddings/sec.

Prereqs (minimum):
    pip install torch transformers
    pip install onnxruntime           # or onnxruntime-gpu for CUDA
    pip install "optimum[onnxruntime]"

Run:
    python bench_embedding.py
"""

from __future__ import annotations

import time

import numpy as np

MODEL = "BAAI/bge-small-en-v1.5"
BATCH_SIZES = [1, 8, 32, 128]
SEQ_LEN = 128
N_WARMUP = 5
N_MEASURE = 50


def make_inputs(batch_size: int) -> dict:
    rng = np.random.default_rng(0)
    input_ids = rng.integers(low=0, high=30000, size=(batch_size, SEQ_LEN), dtype=np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def bench_pytorch_eager() -> dict[int, float]:
    import torch
    from transformers import AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(MODEL).to(device).eval()
    out: dict[int, float] = {}
    for bs in BATCH_SIZES:
        ins = make_inputs(bs)
        ids = torch.from_numpy(ins["input_ids"]).to(device)
        mask = torch.from_numpy(ins["attention_mask"]).to(device)
        for _ in range(N_WARMUP):
            with torch.no_grad():
                model(input_ids=ids, attention_mask=mask)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_MEASURE):
            with torch.no_grad():
                model(input_ids=ids, attention_mask=mask)
        if device == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        out[bs] = (bs * N_MEASURE) / wall
    return out


def bench_pytorch_compile() -> dict[int, float] | None:
    try:
        import torch
        from transformers import AutoModel
    except ImportError:
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(MODEL).to(device).eval()
    model = torch.compile(model, mode="reduce-overhead")
    out: dict[int, float] = {}
    for bs in BATCH_SIZES:
        ins = make_inputs(bs)
        ids = torch.from_numpy(ins["input_ids"]).to(device)
        mask = torch.from_numpy(ins["attention_mask"]).to(device)
        for _ in range(N_WARMUP):
            with torch.no_grad():
                model(input_ids=ids, attention_mask=mask)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_MEASURE):
            with torch.no_grad():
                model(input_ids=ids, attention_mask=mask)
        if device == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        out[bs] = (bs * N_MEASURE) / wall
    return out


def export_to_onnx(out_path: str = "bge.onnx") -> str:
    from pathlib import Path

    if Path(out_path).exists():
        return out_path
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModel.from_pretrained(MODEL).eval()
    sample = tok("hello world " * 10, return_tensors="pt", padding="max_length", max_length=SEQ_LEN, truncation=True)
    torch.onnx.export(
        m,
        (sample["input_ids"], sample["attention_mask"]),
        out_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "last_hidden_state": {0: "batch"},
        },
        opset_version=17,
    )
    return out_path


def bench_ort(provider: str) -> dict[int, float] | None:
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        return None
    if provider not in ort.get_available_providers():
        return None
    onnx_path = export_to_onnx()
    session = ort.InferenceSession(onnx_path, providers=[provider])
    out: dict[int, float] = {}
    for bs in BATCH_SIZES:
        ins = make_inputs(bs)
        for _ in range(N_WARMUP):
            session.run(None, ins)
        t0 = time.perf_counter()
        for _ in range(N_MEASURE):
            session.run(None, ins)
        wall = time.perf_counter() - t0
        out[bs] = (bs * N_MEASURE) / wall
    return out


def main() -> None:
    print("Benchmarking BGE-small across runtimes (embeddings/sec, higher better)")
    print(f"  seq_len={SEQ_LEN}  warmup={N_WARMUP}  measure={N_MEASURE}\n")

    runs = {
        "pytorch eager": bench_pytorch_eager(),
        "pytorch compile": bench_pytorch_compile(),
        "ORT CPU": bench_ort("CPUExecutionProvider"),
        "ORT CUDA": bench_ort("CUDAExecutionProvider"),
        "ORT TensorRT": bench_ort("TensorrtExecutionProvider"),
    }

    header = f"{'runtime':20s} " + " ".join(f"{bs:>10d}" for bs in BATCH_SIZES)
    print(header)
    for name, res in runs.items():
        if res is None:
            print(f"{name:20s}  (not available in this environment)")
            continue
        row = f"{name:20s} " + " ".join(f"{res[bs]:>10.0f}" for bs in BATCH_SIZES)
        print(row)


if __name__ == "__main__":
    main()
