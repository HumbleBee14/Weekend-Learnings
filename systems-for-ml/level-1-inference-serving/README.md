# Level 1 — Inference Serving (Build Your Own)

> Outer reference: [`systems-for-ml/README.md`](../README.md) · Project: **Project 1 — `mini-serve`** (first half)
>
> Textbook companion (academic): [Reddi Vol 1 — *Serving* chapter](https://mlsysbook.ai/) — canonical concept framing.
>
> Practitioner companion: [Kiely, *Inference Engineering*](../references/Inference-Engineering-Kiely-2025.pdf) **Ch 0–1** — the discipline (Ch 0) and the prerequisites every inference engineer needs (Ch 1: scale, online vs offline, latency percentiles, end-to-end metrics). Strongest single read for this level. Open the PDF, ~30 pages.

## How to study this level

```
  Day 0 (15m)  ──►  Read this README — the Week goal + Where this fits
  Day 1 (45m)  ──►  Kiely Ch 0 + Ch 1  (references/Inference-Engineering-Kiely-2025.pdf)
                    ── strongest single read; gives you the vocabulary
  Day 1 → 5    ──►  Topics 01 → 06, in order. For each topic:
                       1. Open the topic folder's  README.md  (the launcher)
                       2. Read its  CONCEPTS.md   (the depth)
                       3. Run/extend the code in the folder
                       4. Mark it done; move on
  Day 5-7      ──►  Project 1 first half — start the capstone
                    at  _capstone-mini-serve/   (~450 LOC scaffold, extend it)
                    Generate G1 + G2 from the project graph list
```

**Reference order when you get stuck:**
1. The topic's own `CONCEPTS.md` (closest to what you're doing)
2. Kiely Ch 0–1 (practitioner framing)
3. [Reddi *Serving* chapter](https://mlsysbook.ai/) (academic framing)
4. The actual engine source code if you need ground truth (vLLM, etc.)

**Don't skip ahead.** The whole point of Topics 03–04 is to feel naive batching break. If you reach for vLLM now, you'll skip the lesson Level 4 needs you to have already learned.

## Week goal

Walk out able to serve an LLM yourself, end-to-end, with no inference framework underneath. By Friday you should have:

- A FastAPI server that loads a transformer (your MiniGPT from `python-pytorch/level-4` is fine, or a small HF model) and answers `/generate` requests.
- Token streaming over Server-Sent Events.
- A naive batching layer you wrote (not vLLM's, not anyone else's).
- A load test that produces real numbers — TTFT, ITL, throughput, p50/p95/p99 — under concurrent users.
- Hands-on contact with `ollama` and `llama.cpp` for the local-serving end of the spectrum, so you know the shape of what comes in Week 8.

The reason we don't touch vLLM yet is pedagogical: every flag in vLLM exists because someone hit a wall. You're going to hit those walls yourself this week.

## Where this fits

- **Comes after:** `python-pytorch` Levels 1–7. You should already be comfortable with `nn.Module`, `model.generate()`, KV caching at the API level, and basic profiling.
- **Comes before:** Level 2 (CUDA), Level 3 (profiling), Level 4 (where you'll replace your naive batching and KV cache with paged versions).
- **Project this feeds:** **Project 1 (`mini-serve`)** — the server you build this week is the artifact you keep extending through Level 4.

## Topic-by-topic deep dive

| # | Topic | What you build |
|---|-------|---------------|
| 01 | inference-server-basics | FastAPI endpoint that loads a model and serves `/generate` |
| 02 | streaming-tokens | SSE — stream tokens as they generate |
| 03 | request-batching | Batch multiple user requests into one forward pass |
| 04 | latency-vs-throughput | Measure and graph the tradeoff |
| 05 | load-testing | Locust / k6 — generate real concurrent load |
| 06 | local-first-touch | Ollama + llama.cpp — see the local-serving end |

### 01 — `inference-server-basics`

**What it is.** A FastAPI app with one route, `POST /generate`, that takes `{prompt, max_tokens, temperature}`, runs `model.generate()`, and returns the completion. No streaming yet. No batching yet. Single-request, single-thread.

**Why it matters.** This is the strawman. Every later optimization is measured against this baseline. You will compare *every* number — your batched version, your paged-KV version, vLLM, SGLang — back to "single-request blocking server." If you skip this you have nothing to compare against.

**Build steps.**
1. Pick a model. Either your MiniGPT from PyTorch Level 4, or `Qwen/Qwen2.5-0.5B-Instruct` (tiny, runs on CPU). Don't pick anything bigger than 1.5B for this week — you want fast iteration, not big numbers.
2. `model = AutoModelForCausalLM.from_pretrained(...)`, `tokenizer = AutoTokenizer.from_pretrained(...)`. Move to device.
3. FastAPI route that calls `model.generate(input_ids, max_new_tokens=...)`, decodes, returns JSON.
4. Run with `uvicorn main:app --workers 1`. One worker on purpose — multi-worker hides batching problems behind OS-level concurrency.

**Common confusion.** "Why is the second request slow even though the first finished?" Because Python's GIL serializes the forward pass. This is the first lesson — model inference is not free-threaded.

**What to measure.** Send 1 request, time it. Send 10 sequentially, time them. Send 10 concurrently (with `asyncio.gather` or `curl &`), time them. The 10-concurrent case will not be 10× faster than 10-sequential — that gap is the motivation for everything that follows.

### 02 — `streaming-tokens`

**What it is.** Server-Sent Events. Client opens an HTTP connection, server sends `data: {"token": "hello"}\n\n` chunks as tokens are generated. This is how every chat UI works (ChatGPT, Claude, etc.).

**Why it matters.** Latency the user perceives ≠ end-to-end latency. The metric that matters is **TTFT — time to first token**. Users will tolerate slow generation if the first token shows up fast. Measuring TTFT separately from total latency is the first step toward thinking like a serving engineer.

**Build steps.**
1. Use HuggingFace's `TextIteratorStreamer` or roll your own with a thread + queue.
2. FastAPI's `StreamingResponse` with `media_type="text/event-stream"`.
3. Yield each token as `f"data: {json.dumps({'token': tok})}\n\n"`.
4. Test with `curl -N` — `-N` disables buffering so you see tokens arrive live.

**Two metrics to start logging now.**
- **TTFT** — clock starts when request hits server, stops when first token leaves.
- **ITL — inter-token latency** — time between consecutive tokens, average and p99.

These two will show up in every graph for the rest of the curriculum.

### 03 — `request-batching`

**What it is.** Two requests arrive within 50ms of each other. Instead of running `model()` twice, you stack their input tensors and run one forward pass. GPU utilization roughly doubles; per-request latency goes up slightly.

**Why it matters.** GPUs hate small batches. A forward pass on batch=1 leaves the SMs (streaming multiprocessors) mostly idle waiting on memory. Batching is *the* lever for throughput on GPU inference. But naive batching has two failure modes you'll discover firsthand:

1. **Padding waste.** If one request is 50 tokens and another is 5000, you pad the short one to 5000 and waste compute.
2. **Head-of-line blocking.** If you wait for a batch to fill, fast users wait for slow ones. If you don't wait, you give up the throughput.

You will hit both of these. That pain is exactly what continuous batching (Week 4 / Week 5) solves.

**Build steps.**
1. Add a queue in front of `model.generate`.
2. A background task drains the queue: collect requests for up to 10ms or until 8 are present, then batch.
3. Pad input_ids to the max length in the batch. Track the original lengths so you can split outputs back out.
4. Re-measure TTFT, ITL, throughput. Compare to Step 01.

**What you will discover.** Throughput went up. p99 latency *also* went up. This is the throughput-vs-latency tradeoff in its purest form, and it's about to become a graph.

### 04 — `latency-vs-throughput`

**What it is.** The single most important graph in serving systems. X-axis: batch size. Y-axis-left: throughput (tokens/sec). Y-axis-right: p99 latency (ms). Two curves, one rising, one rising-then-exploding.

**Why it matters.** Every serving decision — batch size, autoscaling threshold, engine choice — is a point on this curve. If you can't read this graph for your own system, you can't make those decisions for someone else's.

**Build steps.**
1. Pick batch sizes: 1, 2, 4, 8, 16, 32, (64 if memory allows).
2. For each, drive the server at saturation (more concurrent users than the batch can hold).
3. Record throughput and full latency distribution.
4. Plot. This is **G1** in the Project 1 deliverables.

**Insight to extract.** There is a knee in the latency curve. To the left of the knee, more batching is free throughput. To the right, you're trading user-perceived latency for throughput. The knee location depends on model size, sequence length, and hardware. Find yours.

### 05 — `load-testing`

**What it is.** Real load generators — `locust` (Python, scriptable) or `k6` (Go, faster but JS scripts) — driving your server with realistic concurrent users.

**Why it matters.** `curl` in a loop is not a load test. Real load has Poisson arrival distributions, variable sequence lengths, slow clients, retries, connection churn. None of those show up in `for i in $(seq 1 100); do curl ... done`.

**Build steps.**
1. Pick one tool. `locust` is easier to start with; `k6` is what you'll see at infra-heavy companies.
2. Define a "user" that sends a `/generate` request every N seconds with a random prompt from a small corpus.
3. Ramp users from 1 → 100 over 5 minutes.
4. Capture: total throughput, latency CDF (p50/p95/p99/p999), error rate, dropped connections.
5. Plot the latency CDF. This is **G2** in Project 1.

**What to measure beyond the obvious.**
- **Saturation point** — at what concurrency does p99 break your SLA target (pick one — say 2s)?
- **Failure mode** — when you push past saturation, do requests queue, drop, or 5xx? You should know which.

### 06 — `local-first-touch`

**What it is.** A 2-hour detour to install Ollama and run llama.cpp directly. Same prompts, same model family, but on your laptop with no GPU.

**Why it matters.** Half the field runs on consumer hardware. You will meet `llama.cpp` and `Ollama` again in Week 5 (as serving engines) and Week 8 (as the foundation of local-first AI). Get a feel for them now while expectations are low.

**Build steps.**
1. `brew install ollama` (Mac) or download for your OS.
2. `ollama pull qwen2.5:0.5b` — same model family as your server, GGUF-quantized.
3. `ollama run qwen2.5:0.5b "hello"` — see TTFT on CPU.
4. Hit `http://localhost:11434/api/generate` from a script — Ollama exposes an API just like yours.
5. Quick contrast: same prompt to your server, same prompt to Ollama. Note TTFT and tokens/sec for both. Don't over-analyze yet.

**Insight to carry forward.** A 0.5B model on CPU via llama.cpp is *not slow*. The local stack is genuinely competitive for small models. This shapes Week 8.

## Project 1 work this week

You're building the first half of `mini-serve`. By end of Friday you should have:

```
mini-serve/
├── server.py              # FastAPI app, routes, streaming
├── batcher.py             # Naive request batcher
├── model.py               # Model loading, tokenizer setup
├── loadtest/
│   ├── locustfile.py      # or k6 script
│   └── corpus.txt         # ~50 prompts of varied length
└── reports/
    └── week1.md           # Your writeup so far
```

**Required graphs from Project 1, due this week:**
- **G1** — batch size (1 → 32) vs throughput vs p99 latency.
- **G2** — request latency CDF at fixed concurrency (start with 16 users).

Both go in `reports/week1.md` with the **Setup → Observation → Insight** template:

```
Setup: Qwen2.5-0.5B on M2 Mac, CPU-only, 50-prompt corpus, 16 concurrent users for 3 min.
Observation: Throughput rises from 12 tok/s (b=1) to 78 tok/s (b=16); p99 rises from 380ms to 1.4s.
Insight: 6.5× throughput at the cost of 3.7× p99. The knee is around b=8 for this workload.
```

**Break-it list to start (full list closes in Level 4):**
- Static batch with mixed sequence lengths (50 tok and 5000 tok in same batch). Watch padding waste.
- 100 concurrent users with no batching. Watch p99 explode past 30s.

Save the broken-state numbers. The fix comes in Level 4 when you build the paged KV cache and continuous batching.

## Definition of done

You don't move to Level 2 until all of these are true:

- [ ] `mini-serve` runs locally and answers `/generate` requests with streaming.
- [ ] You can articulate, without notes, the difference between TTFT, ITL, and total request latency.
- [ ] You have G1 and G2 in `reports/week1.md` with Setup/Observation/Insight captions.
- [ ] You have hard numbers (not "feels slow") for the two break-it scenarios.
- [ ] You ran the same prompt through Ollama and your server, and you know which was faster and approximately why.

## Resources (canonical only)

- **FastAPI streaming** — [official docs on `StreamingResponse`](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse).
- **HuggingFace `TextIteratorStreamer`** — [transformers docs](https://huggingface.co/docs/transformers/internal/generation_utils#transformers.TextIteratorStreamer).
- **Locust** — [locust.io quickstart](https://docs.locust.io/en/stable/quickstart.html).
- **k6** — [k6.io HTTP basics](https://grafana.com/docs/k6/latest/using-k6/http-requests/).
- **What is TTFT/ITL** — vLLM's [benchmarking guide](https://docs.vllm.ai/en/stable/usage/benchmarking.html) defines these consistently with the rest of the field.
- **Ollama** — [ollama.com/docs](https://github.com/ollama/ollama/blob/main/docs/api.md).

Skip the "build a chatbot in 5 minutes" tutorials. They teach you to call APIs, not to *be* the API.

## Common pitfalls

1. **Using a model that's too big.** A 7B model on a laptop will make iteration painful and you'll skip experiments. Use 0.5B–1.5B all week. The systems lessons don't depend on model size.
2. **Multi-worker uvicorn.** Running with `--workers 4` will look like batching is working when actually you're just running 4 separate models. Stay on 1 worker for this week.
3. **Measuring with `curl` in a shell loop.** Synchronous sequential requests are not concurrent load. Use `locust`/`k6` or at minimum `asyncio.gather` from a Python script.
4. **Not separating TTFT from total latency.** If you only log total latency you miss the metric users actually feel. Log both from day one.
5. **Skipping the broken-state measurements.** "It crashed at 100 users" is not a measurement. Capture *exactly* what broke (queue overflow? OOM? timeout?) and at what number.

## What you'll be able to do after this week

> Build a FastAPI LLM inference server with token streaming and naive request batching; drive it under realistic concurrent load with Locust or k6; characterize the throughput-vs-latency tradeoff curve; identify the saturation knee for a small transformer on a single host.

Each of those clauses is concrete enough that you can defend the numbers behind it — which is the difference between *having read about serving* and *having served*.
