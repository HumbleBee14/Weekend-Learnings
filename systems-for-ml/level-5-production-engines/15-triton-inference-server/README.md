# 15 — NVIDIA Triton Inference Server

> Position in the stack: the **framework-agnostic outer wrapper**. vLLM/SGLang/TRT-LLM are *engines* (they own one model's compute path); Triton Inference Server is a *server* (it owns the HTTP/gRPC surface, model lifecycle, batching dispatch, and multi-model routing). Most production stacks have both: vLLM doing the LLM forward pass, Triton handling everything around it.

## Files

- `CONCEPTS.md` — what Triton IS is, where it fits, model repository layout, dynamic batching, ensemble graphs, the vLLM-backend pattern.
- `model_repository/` — example layout for a real Triton deployment (LLM + embedding + reranker).
- `bench_triton_vs_vllm.py` — same model, vLLM-native endpoint vs Triton+vLLM-backend — measure the wrapper overhead.

## Why this matters in 2026

Every JD that lists vLLM also lists Triton Inference Server. The reason is **production stacks are rarely one model.** A real RAG-shaped service has:

```
   request ──► [Triton] ──┬──► embedding model (ORT or TRT)
                          ├──► retrieval index  (external)
                          ├──► reranker         (ORT, INT8)
                          └──► LLM              (vLLM backend)
```

You can wire this with N separate processes and your own router, or you can hand all of it to Triton, which knows how to:
- Run each model on the right runtime (ONNX, TRT, PyTorch, Python, vLLM, custom C++ backend).
- Batch dynamically per-model with per-model batch policies.
- Compose them with **ensemble graphs** (a DAG executed server-side, no round-trips).
- Hot-load and version models without redeploying.
- Expose Prometheus metrics out of the box.

## Where Triton fits next to vLLM

| Concern | vLLM (engine) | Triton IS (server) |
|---|---|---|
| LLM forward pass | ✅ owns it | delegates to vLLM backend |
| Paged KV cache | ✅ owns it | delegates |
| Continuous batching for LLMs | ✅ owns it | delegates |
| HTTP / gRPC surface | basic (OpenAI-compatible) | full (REST, gRPC, KServe v2) |
| Multi-model serving | one model per process | many models, one server |
| Non-LLM models (embedding, vision, ASR) | not its job | ✅ runs them too |
| Dynamic batching for *non-LLM* models | not its job | ✅ its bread and butter |
| Ensemble / pipeline DAGs | no | ✅ (`ensemble` model type) |
| Model versioning / hot reload | manual | ✅ built-in |
| Multi-GPU model placement | per-process | ✅ instance groups across GPUs |

**Rule of thumb in 2026:**
- If your service is **one LLM, OpenAI-compatible**: vLLM alone is enough.
- If your service is **multiple models + an LLM**: Triton IS with a vLLM backend for the LLM.
- If you need **enterprise features** (KServe integration, advanced auth, audit logs): Triton (often via NVIDIA NIM, which wraps Triton+TRT-LLM as the customer-facing product).

## Quickstart

```bash
# 1. Install (Docker — recommended; bare-metal Triton is a maintenance burden)
docker pull nvcr.io/nvidia/tritonserver:24.10-vllm-python-py3

# 2. Lay out a model repository (see model_repository/ in this folder)
#    Each model is its own folder with config.pbtxt + the model artifact.

# 3. Start the server
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $PWD/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.10-vllm-python-py3 \
  tritonserver --model-repository=/models

# 4. Hit it (HTTP)
curl -X POST localhost:8000/v2/models/llama-vllm/generate \
  -d '{"text_input":"Hello","parameters":{"max_tokens":32}}'

# 5. Compare to vLLM-direct
python bench_triton_vs_vllm.py
```

## Expected output (illustrative, single L40S)

```
Triton+vLLM-backend  vs  vLLM-native, Llama-3-8B, fp16, 50 concurrent users

metric                  vllm-native       triton+vllm        delta
─────────────────────────────────────────────────────────────────────
TTFT p50 (ms)               142               158           +11%
TTFT p99 (ms)               298               341           +14%
throughput (tok/s)         4250              4180           -1.6%
multi-model (LLM+emb)      separate procs    one container  ✓
```

The headline: **Triton wraps vLLM with a small overhead** (~10-15% on TTFT, ~2% on throughput). You pay that price for everything Triton gives you (multi-model, dynamic batching for the embedding model, ensemble graphs, versioning). For a single-LLM service it's not worth it. For a real-world multi-model service it pays for itself in days.

## Ensemble graphs — the killer feature

A common production pattern: classify → route → generate. With separate services you pay 3× HTTP round-trips. With a Triton ensemble:

```
# config.pbtxt for the ensemble
ensemble_scheduling {
  step [
    { model_name: "classifier"  input_map { ... }  output_map { ... } }
    { model_name: "router"      input_map { ... }  output_map { ... } }
    { model_name: "llama-vllm"  input_map { ... }  output_map { ... } }
  ]
}
```

One request, one response, all hops in-process on the GPU node. This is also how VLM serving (Topic 14) is often deployed: vision encoder (TRT) → LLM (vLLM) as a two-step ensemble.

## Try

- **Deploy two models, one Triton.** Llama-3-8B (vLLM backend) and BGE-small (ONNX Runtime backend). Hit each independently.
- **Build an ensemble.** Add a reranker after retrieval. Measure round-trip count and total latency vs N separate services.
- **Hot-reload.** Drop a v2 of the embedding model into `model_repository/embedding/2/` and watch Triton pick it up without restart. Compare to vLLM's "restart the process to swap weights" flow.
- **Read the NIM source.** [NVIDIA NIM](https://docs.nvidia.com/nim/) is essentially Triton + TRT-LLM + a customer-facing OpenAI-compatible shim. Understanding Triton is most of understanding NIM.

## Where this goes

- Project 2 (`engine-bakeoff`): add `triton+vllm-backend` as one of the bake-off entries. Quantify the wrapper overhead on your hardware.
- Level 7's `mini-platform`: if you have multiple models (LLM + embedding + reranker for a RAG-shaped deployment), Triton is the natural host. The router topic (Level 7 Topic 06) sits *in front of* Triton, not in place of it.

## References

- [Triton Inference Server docs](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/)
- [Triton vLLM backend](https://github.com/triton-inference-server/vllm_backend)
- [NVIDIA NIM overview](https://docs.nvidia.com/nim/) — the productized wrapper

## Conceptual link to Level 5 Topic 13 (ONNX Runtime + TensorRT)

Topic 13 covered the *runtimes* that host non-LLM models (ORT, TRT). This topic shows how Triton **composes** those runtimes alongside an LLM engine into one production service. The progression: write a kernel (Level 2) → optimize one model (Level 4) → serve one model (Level 5 Topics 1–13) → serve a whole stack (this topic).
