# Prompt 05 — Worked Solution

> Open this only after attempting the prompt cold. This is one defensible design, not the only one.

## 1. Clarifying questions (the first 3 minutes)

A senior candidate asks these *before* drawing anything. They scope the design and signal you've shipped multi-model pipelines before:

1. **Per-query token shape.** What's the mean/p99 length of a question, of a retrieved doc chunk, of the synthesis prompt after top-k assembly? (Affects whether the LLM step is the bottleneck — usually yes — and how aggressive prefix-caching pays off.)
2. **Top-k for retrieval and rerank.** Is it `retrieve 100 → rerank top 100 → keep top 5 for LLM`, or `retrieve 20 → keep top 5 without rerank`? (The reranker's batch shape is fixed by this; it's the difference between needing a GPU reranker and not.)
3. **Streaming or buffered?** Are we streaming the LLM tokens to a UI, or returning the full answer in one JSON response? (Streaming hides the LLM latency behind TTFT; without it, p95 < 3s is much tighter.)
4. **Doc-level access control.** Does the answer need to be filtered by who-can-see-what at query time, or is the vector store partitioned by tenant up front? (Affects whether we run a post-retrieval ACL filter — which can require *over-retrieving* to keep the post-filter top-k stable.)
5. **What fraction of queries include a screenshot?** 5% or 50%? (Tells me whether to provision the vision encoder as a small always-on pool or as a separate scale-to-zero pool.)

**Reasonable assumptions to bake in if the interviewer waves off:**
- Query mean 25 tokens, doc chunk mean 400 tokens, final synthesis prompt ~2500 tokens (5 chunks + question + system prompt), output mean 350 tokens
- Retrieve top 50 → rerank → keep top 5 for LLM
- Streaming SSE to the browser; TTFT is the user-perceived metric, p95 end-to-end is the contract
- Tenant-partitioned vector store (each Fortune 500 division has its own namespace); ACL at the namespace level, not row level
- 15% of queries include a screenshot

## 2. The right answer in one sentence

**NVIDIA Triton Inference Server hosting a single ensemble graph — embedding (ORT) → vector-fetch (Python backend) → reranker (TRT) → vision encoder (TRT, conditional) → LLM (vLLM backend) — colocated on the same GPU node so there are zero internal HTTP hops, with per-model dynamic batching, prefix-cache on the LLM, and a Redis query→embedding cache in front.**

The sentence that separates this from the bluff answer: **the pipeline is one server-side DAG, not three microservices.** The naive design is "an embedding service, a reranker service, an LLM service, all talking HTTP to a Python orchestrator." That adds 4 network round-trips (~5–15ms each plus tail-latency dragons), prevents cross-model batching, forces independent scaling decisions on each, and turns a 2.5s pipeline into a 3.5s pipeline at p95 with a worse failure surface. Triton's ensemble feature was built for exactly this shape — and the candidate who knows that is the one you want serving production RAG.

## 3. The architecture (whiteboard)

```
                          Internet (employees)
                                 │
                                 ▼
                       ┌────────────────────┐
                       │   API gateway      │   ─ SSO (Okta/Azure AD)
                       │   (Envoy)          │   ─ per-user rate limit
                       │                    │   ─ stamps user_id, dept,
                       │                    │     OTel trace context
                       └─────────┬──────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │  Query cache       │   ─ Redis cluster
                       │  (Redis)           │   ─ key = sha256(query + acl)
                       │                    │   ─ TTL 1h; ~20% hit rate
                       └─────────┬──────────┘
                                 │ miss
                                 ▼
                  ┌──────────────────────────────┐
                  │   Triton Inference Server    │   ONE process per node
                  │   (NVIDIA NIM-style image)   │
                  │   ── ensemble DAG: ─────────│
                  │                              │
                  │   ┌─────────────┐            │
                  │   │ embed_model │  ORT-INT8  │   bge-large, dynamic
                  │   │  (CPU pool) │  on CPU    │   batch up to 64
                  │   └──────┬──────┘            │
                  │          │                   │
                  │          ▼                   │
                  │   ┌─────────────────┐        │
                  │   │ vision_encoder  │  TRT   │   conditional —
                  │   │  (15% of reqs)  │  FP16  │   skipped if no image
                  │   └────────┬────────┘        │
                  │            │                 │
                  │            ▼                 │
                  │   ┌─────────────────┐        │
                  │   │ vec_fetch       │  Python│   Pinecone/pgvector
                  │   │  (Python BE)    │  BE    │   client; top-50;
                  │   │                 │        │   honors namespace ACL
                  │   └────────┬────────┘        │
                  │            │                 │
                  │            ▼                 │
                  │   ┌─────────────────┐        │
                  │   │ rerank_model    │  TRT   │   bge-reranker-v2-m3
                  │   │ (GPU pool)      │  FP16  │   pairs (q, doc) × 50;
                  │   │                 │        │   batch in one fwd
                  │   └────────┬────────┘        │
                  │            │                 │
                  │            ▼                 │
                  │   ┌─────────────────┐        │
                  │   │ llm_synthesize  │  vLLM  │   Llama-3-70B FP8,
                  │   │ (GPU pool)      │  BE    │   TP=2 across H100s,
                  │   │                 │        │   prefix-cache ON,
                  │   │                 │        │   SSE streaming out
                  │   └────────┬────────┘        │
                  └────────────┼─────────────────┘
                               │ streamed tokens
                               ▼
                         to client (SSE)

         ┌─────────────────────────────────────┐
         │  Control plane                      │
         │  ─ Triton model-control-mode=EXPLICIT│
         │    + model_repository on S3-fuse    │
         │  ─ Prometheus → Grafana             │
         │    nv_inference_request_duration_us │
         │    per-model batch fill ratio       │
         │  ─ OTel GenAI spans per stage       │
         │  ─ KEDA on nv_inference_queue_duration│
         └─────────────────────────────────────┘
```

### The five-box mapping
- **Gateway:** Envoy with SSO auth, per-user rate limit (employees can't DDoS each other), OTel trace context propagation. Stamps `user_id`, `dept_id`, `acl_scope` into headers — passes through to Triton.
- **Router:** Redis query cache (cheap win, ~20% hit on common queries inside a Fortune 500) then a Kubernetes Service in front of Triton replicas. No KV-cache-aware routing needed at 50 QPS; round-robin is fine because every replica has identical model state.
- **Scheduler:** Triton's per-model dynamic batcher (embedding batches at ~30ms window, reranker at ~10ms, LLM via vLLM's continuous batcher). The ensemble scheduler routes intermediate tensors between models *in-process*, never hitting the network.
- **Worker:** A Triton node has CPU cores for the embedding model (ORT INT8) and 2 H100s for {vision encoder, reranker, LLM}. LLM gets ~80% of GPU mem; reranker + vision share the rest.
- **Control plane:** Model repository on an S3-mounted volume so model versions are git-managed; Triton in `EXPLICIT` mode so deploys are atomic; KEDA scales replica count on `nv_inference_queue_duration_us` (Triton's queue-wait metric); OTel spans per stage so you can attribute latency to embed vs. rerank vs. LLM.

**The senior signal:** drawing one Triton box with the ensemble DAG inside, not four boxes connected by HTTP arrows. Bonus: naming the right runtime per model (ORT-INT8 for embedding on CPU, TRT-FP16 for small encoders, vLLM-FP8 for the LLM) — different models want different engines, and Triton is the one server that hosts them all under one process.

## 4. The capacity math

```
Workload:
  50 QPS sustained, p95 end-to-end < 3s.
  Each query: 1 embed + 1 vec-fetch + 50 rerank pairs + 1 LLM synth (~2500 in / 350 out).
  15% of queries also: 1 vision-encoder pass on ~512×512 screenshot.

LATENCY BUDGET (the load-bearing analysis):
  embed (bge-large, 24 layers, query is short):
      CPU ORT-INT8, batch 32 @ 30ms window
      → per-query 180-220ms wall (batched)
  vec_fetch (Pinecone HTTP, p95):
      ~50ms (in-region, with connection pool)
  vision_encoder (conditional — assume worst case):
      TRT FP16, batch 1, ViT-L/14 @ 512×512
      → ~80ms on H100
  rerank (bge-reranker-v2-m3, 50 pairs):
      TRT FP16, one batched fwd
      → ~180ms on H100 (50 pairs of ~512 tok each)
  llm_synthesize (Llama-3-70B FP8, vLLM, TP=2):
      prefill 2500 tokens: ~120ms (60% of an H100 sec at 21K tok/s prefill)
      decode 350 tokens at ~50 tok/s/req (TP=2): ~7s end-to-end
      TTFT: prefill + 1st-token decode ≈ 140ms
      End-to-end LLM step ≈ 7s — TOO SLOW.

WAIT. Re-check decode envelope.
  Llama-3-70B FP8, TP=2 on 2×H100 80GB, vLLM 0.7:
      decode at batch 16: ~95 tok/s/req (memory-bandwidth bound)
      decode at batch 1:  ~55 tok/s/req
  350 tokens at 95 tok/s = 3.7s          ← acceptable if we batch
  350 tokens at 55 tok/s = 6.4s          ← not acceptable alone

So end-to-end at batch-friendly load:
  embed 200 + fetch 50 + rerank 180 + synth 3700 = ~4.1s
  TTFT: 200 + 50 + 180 + 140 = 570ms — perceived snappy via SSE
  But p95 contract is "end-to-end < 3s" — we miss without help.

THE FIX — what makes p95 < 3s achievable:
  (a) Prefix-cache the system prompt + few-shot exemplars (~800 tokens fixed)
      → vLLM prefix-cache hit rate ~85% → prefill drops to ~50ms
  (b) Spec decode (EAGLE-3 draft) for LLM → ~1.7× decode speedup
      → 350 tokens at ~160 tok/s effective = 2.2s
  (c) Embed + rerank + vec-fetch can overlap with LLM prefill kick-off
      (Triton ensemble executes the LLM as soon as inputs are ready;
       the small steps' work is amortized inside the batching windows)

Revised: embed 200 + fetch 50 + rerank 180 + synth 2200 = 2.63s p95
  TTFT (first SSE token) ≈ 530ms
  Within 3s budget. Comfortable headroom for the 15% vision-encoder branch.

GPU COUNT from throughput:
  LLM (the only meaningful GPU cost):
    50 QPS × (2500 in + 350 out) = 142K tok/s aggregate
    Llama-3-70B FP8 TP=2 on 2×H100: ~13K tok/s prefill + ~3K tok/s decode at batch 16
    Decode dominates: 50 QPS × 350 = 17.5K tok/s
    GPUs needed: 17.5K / (3K × 0.7 util) = 8.3 → need ~5 TP=2 replicas = 10 H100s
  Reranker (bge-reranker-v2-m3):
    50 QPS × 50 pairs = 2500 pair-evals/s
    H100 envelope: ~25K pair-evals/s at batch 50, FP16
    → 0.1 H100 → colocate on the same nodes as the LLM (uses spare cycles)
  Vision encoder (15% of QPS):
    7.5 QPS × ~80ms = 0.6 H100-second/sec ≈ 0.6 H100
    → colocate on the same nodes
  Embedding (CPU):
    50 QPS × ~4 cores/QPS at batch 32 → ~200 cores → ~25 CPU pods
    OR move to GPU (one H100 handles 50 QPS easily) — depends on $/QPS math

Binding: 10 H100s (5 nodes of 2×H100 each) for LLM-dominant.
Round to 6 nodes for headroom. + 25 small CPU pods for embedding.
```

### Cost-to-serve, blended

```
Option                                                   $/Mtok blended   Monthly @ 50 QPS
─────────────────────────────────────────────────────────────────────────────────────────
Triton ensemble, 6× 2×H100 nodes, ORT-CPU embed (this)   $0.95            $11,800
3 separate HTTP services, naive routing                  $1.40            $17,400
Triton but everything on GPU (embed on GPU too)          $1.10            $13,700
Bedrock for LLM + own retrieval                          $2.30            $28,500
```

The Triton + CPU-embed design wins on two axes simultaneously: zero internal HTTP hops (lower p95) and CPU-cheap embedding (lower $/Mtok). Putting the embedding model on GPU is a common mistake — at 50 QPS with batch-32 windows, bge-large on a 64-core Sapphire Rapids node with ORT INT8 will cost a fraction of an H100 slice.

## 5. The hard parts — what actually breaks

### 5a. The Triton ensemble config

This is the load-bearing artifact. Triton ensembles are defined in `config.pbtxt`. The config below routes embedding output to vec-fetch, vec-fetch output to rerank, rerank output to the LLM — all inside one server-side request.

```protobuf
# /models/rag_pipeline/config.pbtxt
name: "rag_pipeline"
platform: "ensemble"
max_batch_size: 8

input [
  { name: "query"        data_type: TYPE_STRING  dims: [ 1 ] },
  { name: "screenshot"   data_type: TYPE_FP32    dims: [ -1, 3, 512, 512 ] optional: true },
  { name: "acl_namespace" data_type: TYPE_STRING dims: [ 1 ] }
]

output [
  { name: "answer_stream" data_type: TYPE_STRING dims: [ -1 ] }
]

ensemble_scheduling {
  step [
    {
      model_name: "embed_model"        # ORT-INT8 on CPU
      input_map  { key: "text" value: "query" }
      output_map { key: "vec" value: "_query_vec" }
    },
    {
      model_name: "vision_encoder"     # TRT FP16, conditional
      model_version: -1
      input_map  { key: "image" value: "screenshot" }
      output_map { key: "img_vec" value: "_img_vec" }
    },
    {
      model_name: "vec_fetch"          # Python backend
      input_map  { key: "qvec" value: "_query_vec"
                   key: "ivec" value: "_img_vec"
                   key: "ns"   value: "acl_namespace" }
      output_map { key: "docs" value: "_top50_docs" }
    },
    {
      model_name: "rerank_model"       # TRT FP16
      input_map  { key: "query" value: "query"
                   key: "docs"  value: "_top50_docs" }
      output_map { key: "top5"  value: "_top5_docs" }
    },
    {
      model_name: "llm_synthesize"     # vLLM backend, streaming
      input_map  { key: "query" value: "query"
                   key: "ctx"   value: "_top5_docs" }
      output_map { key: "tokens" value: "answer_stream" }
    }
  ]
}
```

Two non-obvious wins from this layout:
- **Per-model dynamic batching independently configured.** The embedding model batches at a 30ms window (high tolerance, gather-many wins); the LLM uses vLLM's continuous batcher (no fixed window). Triton's scheduler honors both simultaneously without the candidate having to think about it.
- **`max_batch_size: 8` at the ensemble level** lets up to 8 concurrent end-to-end pipelines flight, which is what feeds the LLM enough concurrency to hit batch-16 internally (each ensemble request often has multiple reranker pairs etc.).

### 5b. Why dynamic batching matters per-model

```
embed_model:        latency_p95 vs. batch_size
  batch 1   →  35ms
  batch 8   →  60ms      (8× throughput, only 1.7× latency)
  batch 32  →  180ms     (32× throughput at the 30ms window cost)

rerank_model:       latency_p95 vs. batch_size
  batch 10  →  60ms
  batch 50  →  180ms     (5× throughput, 3× latency)
  batch 100 →  340ms     (10× throughput, 5.7× latency) — diminishing returns

llm_synthesize:     decode tok/s vs. concurrent reqs (Llama-70B FP8 TP=2)
  1 req    →   55 tok/s
  4 reqs   →   78 tok/s/req
  16 reqs  →   95 tok/s/req
  32 reqs  →  102 tok/s/req (only marginal beyond 16 — KV memory becomes the wall)
```

Each model has a different optimal batching policy. Per-model knobs in Triton:
```protobuf
dynamic_batching {
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 30000   # 30ms — embedding tolerates this
}
```

### 5c. Prefix caching on the LLM

The system prompt + few-shot exemplars are ~800 tokens of fixed text on every query. vLLM's automatic prefix caching detects the shared prefix and reuses the KV blocks — first request pays full prefill, subsequent share. At 50 QPS, hit rate stabilizes ~85% within minutes of warmup. Effect: prefill drops from ~120ms to ~20ms for the cached portion.

```python
# vLLM backend config (passed to Triton's vLLM backend)
{
  "model": "meta-llama/Meta-Llama-3-70B-Instruct",
  "quantization": "fp8",
  "tensor_parallel_size": 2,
  "enable_prefix_caching": true,
  "max_num_batched_tokens": 8192,
  "gpu_memory_utilization": 0.90,
  "speculative_config": {
      "model": "eagle3-llama-70b",        # spec decode draft
      "num_speculative_tokens": 5
  }
}
```

EAGLE-3 spec decode gets us the 1.7× decode speedup that makes the p95 budget achievable. Level 5 Topic 12 covers the production setup.

### 5d. Vector store as a Python backend (not a separate service)

The "vec_fetch" step is a Python-backend Triton model wrapping the Pinecone client. This keeps it inside the ensemble — Triton can batch concurrent fetches, and the OTel span sits in the same trace as the LLM call. The naive alternative — a Lambda or a Go microservice — adds 15–30ms of network + cold-start risk per query.

```python
# /models/vec_fetch/1/model.py
import triton_python_backend_utils as pb_utils
from pinecone import Pinecone
import numpy as np

class TritonPythonModel:
    def initialize(self, args):
        self.pc = Pinecone(api_key=os.environ["PINECONE_KEY"])
        self.index = self.pc.Index("corp-docs")

    def execute(self, requests):
        responses = []
        for req in requests:
            qvec = pb_utils.get_input_tensor_by_name(req, "qvec").as_numpy()
            ns   = pb_utils.get_input_tensor_by_name(req, "ns").as_numpy()[0].decode()
            ivec = pb_utils.get_input_tensor_by_name(req, "ivec")  # may be None

            # If image vector exists, do hybrid fusion (weighted avg)
            if ivec is not None:
                qvec = 0.7 * qvec + 0.3 * ivec.as_numpy()

            res = self.index.query(
                vector=qvec.tolist(),
                top_k=50,
                namespace=ns,                    # ACL boundary
                include_metadata=True,
            )
            docs_tensor = pb_utils.Tensor(
                "docs",
                np.array([d["metadata"]["text"] for d in res["matches"]], dtype=object)
            )
            responses.append(pb_utils.InferenceResponse([docs_tensor]))
        return responses
```

The `namespace=ns` parameter is the ACL boundary — Fortune 500 means strict per-division isolation, and the Pinecone namespace per division gives that for free.

### 5e. The screenshot branch — conditional execution

15% of queries include a screenshot. Triton ensembles handle this elegantly: the `screenshot` input is marked `optional: true`. If absent, the `vision_encoder` step receives a zero-element tensor and is short-circuited; `vec_fetch` sees `ivec=None` and skips fusion. No separate code path, no separate deployment. The vision encoder pool is sized for the 15% load (about 7.5 QPS), shared on the same 2×H100 nodes as the reranker.

## 6. The break-it list

| Failure | What happens | Your mitigation |
|---|---|---|
| Pinecone region outage | All RAG queries fail at vec_fetch | Local pgvector replica as fallback, fed by nightly snapshot; degraded mode is "older doc index" not "no answers" |
| Embedding model OOM on CPU pod | Embeds queue, pipeline TTFT spikes | KEDA on `nv_inference_queue_duration_us{model="embed_model"}` scales the CPU pool; per-pod max-concurrent enforced |
| Reranker disagrees with embedding (low overlap) | Top-5 to LLM is irrelevant; bad answers | Eval gate — bi-encoder + cross-encoder pair quality-tested together on internal eval set before release; cross-encoder/embedding model version pinning |
| LLM hallucinates outside the retrieved context | Wrong answer with confidence | Citation enforcement in prompt (must quote chunk_id); post-gen grounding check (NLI model verifies span overlap); display "low-confidence" badge when grounding < 0.6 |
| ACL leak — user sees docs from another division | Catastrophic compliance event | ACL enforced at vec_fetch (Pinecone namespace), AND re-checked at LLM prompt assembly (chunk's `dept_id` must match user's allowed set); audit log on every retrieval |
| GPU node loses 1 of 2 GPUs (NVLink failure) | TP=2 LLM dies | Triton health-checks the LLM model; KEDA marks node unhealthy; traffic drains; node restarted with 1×H100 in degraded TP=1 (slower, but serves) |
| Long-doc query → 30K-tok context blows LLM batch | KV mem OOM in vLLM, other requests evicted | Hard cap on chunks-to-LLM (top 5, each <= 800 tok); rerank emits `truncated=True` flag; observable via OTel |
| Query cache stale after doc reindex | Users see old answers | Cache key includes index version (`sha256(query + acl + index_epoch)`); index publish bumps epoch atomically |
| New employee onboarding → cold cache 100% miss | Latency spikes for the day | Cache miss is still 2.6s p95, well within budget — accept; only an issue at 10× scale where we'd add a warming job |
| Triton model_repository S3 stall on deploy | New model version stuck loading | Triton model-control-mode=EXPLICIT — only loads on explicit API call; rollout pipeline waits for `READY` per replica before flipping traffic |

## 7. What changes at 10× scale

```
At 500 QPS (10× sustained):

Sharding:
  - Vector store: shard the Pinecone index by department for true parallelism
    (or partition pgvector across nodes by namespace)
  - Triton ensemble: replicate across regions (us-east, us-west, eu-west)
    for latency to global Fortune 500 offices
  - Embedding model: dedicated pool with its own autoscaler (it batches well —
    larger pool means bigger batches means cheaper $/Mtok)

Caching:
  - Add Redis embedding cache (query_text → vector) for queries with high repeat
    rate — a Fortune 500 internal search sees ~25% repeat queries day-over-day
  - LMCache (Level 7 Topic 12) for cross-replica KV reuse of common doc-chunk
    prefixes — the same KB articles are retrieved across many queries
  - System prompt prefix-cache already on, but now warmed across replicas via
    a shared KV layer

Compute:
  - ~100 H100s for LLM-dominant path at 500 QPS
  - Justifies disaggregated prefill/decode (Level 5 Topic 08) — prefill on a
    small fast pool, decode on the bigger pool
  - NVIDIA Dynamo as the orchestrator (Level 5 Topic 09)

Reliability:
  - Multi-region active-active, request-level routing on user's home region
  - Per-region failure domain — losing us-east-1 doesn't take down the product
  - Read-replica vector stores in each region; writes go to primary, replicate
    async (eventual consistency for doc index, which is fine for RAG)

Team-shape:
  - Now justifies a dedicated retrieval-quality eng (separate from LLM eng);
    these are distinct skill sets at scale
  - Per-doc-corpus eval pipeline becomes continuous (Level 7 Topic 03)
```

**The axis of change:** at 10×, the LLM is no longer the only thing worth optimizing — the retrieval side becomes a real engineering investment. Embedding pool sizing, vector store sharding, query caching, and cross-region replication all become first-class concerns. The Triton ensemble pattern stays; you just run more of them.

## 8. The 30-second summary you give the panel

> "RAG is a multi-model pipeline, so the right substrate is NVIDIA Triton with an ensemble graph — embedding on CPU with ORT-INT8, vision encoder and reranker on TRT, Llama-3-70B FP8 on the vLLM backend, all colocated, zero internal HTTP. Sized at ~10 H100s for 50 QPS, LLM-dominant. The p95 < 3s budget needs three things together: prefix-cache for the 800-token system prompt, EAGLE-3 spec decode on the LLM, and Triton's dynamic batching tuned per-model. Vector store is Pinecone, accessed via a Python-backend Triton model so the fetch sits inside the ensemble trace and respects ACL via namespace. At 10× I'd shard by department, replicate Triton across regions, and add LMCache. The biggest mistake here is shipping three HTTP services with a Python orchestrator — that's a 30% latency tax and a worse failure surface for no benefit."

## What this prompt is really testing

- **Triton Inference Server as the right tool for multi-model pipelines** (Level 5 Topic 15) — the candidate either knows ensembles or doesn't
- **Per-model runtime selection** — ORT for CPU embedding, TRT for small encoders, vLLM for the LLM. Real production RAG uses three engines, not one
- **Latency budgeting** — decomposing the p95 contract across stages and finding which stage owns the budget
- **Prefix caching + spec decode as a budget-saver** (Level 4 + Level 5 Topic 12) — not nice-to-have, load-bearing at 50 QPS with 70B
- **Multi-tenancy and ACL** in retrieval — the Fortune 500 framing is a signal that this is enterprise, and ACL leak is the worst-case failure
- **What you don't do** — naming things you skip (HTTP between models, GPU embedding, separate vector microservice) is a senior signal
- **Migration thinking** — knowing the single-region Triton ensemble holds until ~500 QPS, then you shard, is the seniority signal

## References

- [Level 5 Topic 15 — Triton Inference Server](../../../level-5-production-engines/15-triton-inference-server/)
- [Level 5 Topic 12 — production spec decode (EAGLE-3)](../../../level-5-production-engines/12-spec-decode-prod/)
- [Level 5 Topic 14 — vision-language models](../../../level-5-production-engines/14-vlm/)
- [Level 5 Topic 13 — ONNX / TRT runtime choice](../../../level-5-production-engines/13-onnx-trt/)
- [Level 5 Topic 08 — disaggregated prefill/decode](../../../level-5-production-engines/08-disaggregated-prefill-decode/)
- [Level 7 Topic 03 — eval pipelines](../../../level-7-ml-platform/03-eval/)
- [Level 7 Topic 05 — OTel GenAI semconv](../../../level-7-ml-platform/05-otel/)
- [Level 7 Topic 12 — KV-tiering / LMCache](../../../level-7-ml-platform/12-kv-tiering/)
- NVIDIA Triton ensemble docs: `https://docs.nvidia.com/deeplearning/triton-inference-server/`
- bge-reranker-v2-m3 model card (BAAI, 2024) — the reranker most production RAG ships with
- Kiely *Inference Engineering* Ch 4 §4.3 — multi-model serving and ensemble patterns
