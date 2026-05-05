# Level 9 — Rust for ML Infrastructure

> Outer reference: [`compiler-and-kernels/README.md`](../README.md) · Project: Rust tokenizer + prefix-routing layer; P50/P99 latency vs Python equivalent

## Week goal

Rust has moved from interesting to standard in the ML inference infrastructure layer. Not for model code — for the parts that need zero-GC latency, thread safety without the GIL, and small binary footprint. By Friday you should be able to:

- Understand where Rust shows up in production ML systems and why (not just "it's fast")
- Write a Rust tokenizer pipeline using the `tokenizers` crate (already under your Python `tokenizer.encode()`)
- Write a Rust-based prefix-aware request router using Tokio + HuggingFace Candle for embedding lookup
- Benchmark P50/P99 latency vs the Python equivalent and explain the gap

## Where this fits

- **Comes after:** All of Levels 1–8. Rust is the final layer — the infrastructure shell that the kernel and compiler work runs inside. You don't need to write Rust model code; you need to understand the Rust infrastructure layer that wraps the Python/CUDA code.
- **Comes before:** Nothing. This is the last level. The capstone of the full track.

## Why Rust, not C++

C++ is the kernel language (CUDA, CUTLASS). Rust is the *infrastructure* language. The distinction:
- C++: memory management is manual, concurrency is unsafe, build system is painful, package ecosystem is fragmented
- Rust: memory safety guaranteed by the compiler, concurrency is safe (`Send`/`Sync` traits), `cargo` is excellent, `crates.io` has everything you need for HTTP servers, async runtimes, and tokenization

For ML inference infrastructure specifically: the request handling path, tokenization, routing, and protocol parsing are latency-critical but not GPU-dependent. Rust's zero-GC, async-native model (Tokio) is the right tool. The Python GIL means a Python router at 10K req/s has serious latency variance. Rust doesn't have a GIL.

## 2026 reality check

- **HuggingFace `tokenizers`** (Rust crate) is used in production by virtually every major inference stack. The Rust implementation is 10–43× faster than pure Python tokenizers for tokenization-heavy workloads.
- **vLLM Semantic Router v0.1 "Iris"** (Jan 2026) is production Rust — 25% higher request throughput and 1,200ms lower TTFT vs the Python-based router. It uses Candle for embedding inference and Tokio for async request handling. 600+ PRs merged, contributions from 50+ engineers.
- **Candle** (HuggingFace): minimalist ML inference in Rust. No Python runtime. Supports LLaMA, Mistral, Whisper. Used for embedding inference, small model serving, serverless functions.
- **mistral.rs**: PagedAttention + speculative decoding + continuous batching in Rust, built on Candle. For serving contexts where Python overhead is unacceptable.
- **`safetensors`**: the standard tensor serialization format. Core is Rust; the Python API is a PyO3 wrapper. You're already using this in every PyTorch project.

## Topic-by-topic deep dive

| # | Topic | What you build |
|---|-------|---------------|
| 01 | rust-for-ml-engineers | What Rust offers; the GIL problem; ownership model |
| 02 | tokenizers-crate | Fast tokenization; BPE internals; PyO3 Python binding |
| 03 | tokio-async | Async HTTP server; connection handling; backpressure |
| 04 | candle-inference | Run a small model in pure Rust; no Python required |
| 05 | prefix-aware-router | Hash-based prefix routing; the vLLM Router architecture |
| 06 | pyo3-bridge | Call Rust from Python; hot-path extraction |
| 07 | mistralrs-tour | Read mistral.rs: PagedAttention + spec decode in Rust |

### 01 — `rust-for-ml-engineers`

**The GIL problem.** Python's Global Interpreter Lock means that at any moment, only one Python thread runs Python code. For ML inference infrastructure at high QPS, this is the bottleneck: request parsing, tokenization, routing, and response serialization all run single-threaded despite having many OS threads available. `async/await` in Python (asyncio) sidesteps the GIL for I/O but not for CPU-bound work.

**What Rust gives you.** No GIL. Memory safety without a garbage collector (no GC pauses in the request path). Native `async/await` via Tokio (a battle-tested async runtime). The `Send`/`Sync` traits make concurrency bugs compile-time errors rather than runtime bugs.

**The ownership model — the hard part.** Rust's borrow checker enforces that each value has one owner, and references must not outlive the value they reference. This is the main learning curve. Spend time here; don't rush to write network code until you're comfortable with:
- `String` vs `&str` (owned vs borrowed string)
- `Vec<T>` vs `&[T]` (owned vs slice)
- `Arc<Mutex<T>>` for shared mutable state across threads

**Resources.**
- [The Rust Book](https://doc.rust-lang.org/book/) — Chapters 1–10 are enough for this week.
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/) — faster if you learn by doing.

### 02 — `tokenizers-crate`

**What the `tokenizers` crate is.** The Rust library that HuggingFace's `transformers` Python package wraps. When you call `tokenizer.encode("hello world")` in Python, you're calling into Rust via a PyO3 binding.

**BPE internals.** Byte-Pair Encoding: starts with a vocabulary of individual bytes; iteratively merges the most frequent adjacent pair; repeats until vocabulary size is reached. The Rust implementation uses a trie for fast pair lookup and parallel processing across the vocabulary. This is why it's 43× faster than Python — the merge loop runs without Python object overhead.

**Build steps.**
```toml
# Cargo.toml
[dependencies]
tokenizers = "0.19"
```

```rust
use tokenizers::Tokenizer;

fn main() {
    let tokenizer = Tokenizer::from_pretrained("Qwen/Qwen2.5-0.5B", None).unwrap();
    let encoding = tokenizer.encode("Hello, world!", false).unwrap();
    println!("{:?}", encoding.get_ids());
}
```

1. Write a Rust binary that loads a tokenizer, encodes 10K prompts, and measures throughput.
2. Compare to the Python `tokenizers` package doing the same thing. They should be similar — the Python package is already calling into this Rust code. The difference shows the PyO3 bridge overhead.
3. Write a batch tokenizer that processes a slice of prompts in parallel (use Rayon: `prompts.par_iter().map(|p| tokenizer.encode(p, false))`).

### 03 — `tokio-async`

**Tokio.** The standard Rust async runtime. An async function in Rust is a state machine that can be paused at `.await` points. Tokio's scheduler runs many async tasks on a thread pool — you get high concurrency without threads-per-connection overhead.

**Build an HTTP server in 30 lines.**
```rust
use axum::{routing::post, Router, Json};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct GenerateRequest { prompt: String, max_tokens: usize }

#[derive(Serialize)]
struct GenerateResponse { tokens: Vec<u32> }

async fn tokenize_handler(Json(req): Json<GenerateRequest>) -> Json<GenerateResponse> {
    // tokenize synchronously (fast, no I/O)
    let tokens = TOKENIZER.encode(&req.prompt, false).unwrap().get_ids().to_vec();
    Json(GenerateResponse { tokens })
}

#[tokio::main]
async fn main() {
    let app = Router::new().route("/tokenize", post(tokenize_handler));
    axum::Server::bind(&"0.0.0.0:8080".parse().unwrap())
        .serve(app.into_make_service()).await.unwrap();
}
```

**Build steps.**
1. Write this server with `axum` (the standard Rust HTTP framework).
2. Drive it with `wrk` or `locust` at 1K, 5K, 10K req/s.
3. Measure P50/P99 latency and compare to a FastAPI equivalent.
4. The gap should be significant at P99 — GC pauses in Python show up in the tail.

### 04 — `candle-inference`

**What Candle is.** Minimalist ML inference in Rust. No Python runtime. Uses safetensors for model loading, supports CUDA and Metal backends. The design goal: a tiny binary that loads a model and serves it, with no Python startup overhead.

**Why this matters.** Serverless inference (AWS Lambda, Google Cloud Run) benefits enormously from fast cold starts. A Candle binary cold-starts in <1 second. A Python vLLM instance cold-starts in 30–60 seconds. For low-traffic applications, the Candle path is the right one.

**Build steps.**
```rust
use candle_core::{Device, Tensor};
use candle_transformers::models::llama::{Config, Llama};

// Load Qwen2.5-0.5B in pure Rust
let device = Device::Cpu;  // or Device::Cuda(0) for GPU
let config = Config::qwen2_5_0b5();
let vb = candle_nn::VarBuilder::from_safetensors(&["model.safetensors"], DType::F32, &device)?;
let model = Llama::load(vb, &config)?;
```

1. Clone the `candle` examples. Run the LLaMA example.
2. Measure cold-start time (from binary start to first token): compare to Python vLLM.
3. Measure throughput on CPU (your Mac) vs Python + PyTorch CPU.

### 05 — `prefix-aware-router`

**The project.** Build a Rust-based request router that:
1. Accepts HTTP requests with a `{prompt, model}` body
2. Tokenizes the prompt (using the `tokenizers` crate)
3. Computes a prefix hash (SHA-256 of the first N token IDs, block-aligned to 16 tokens)
4. Routes to the backend (vLLM instance) with the longest prefix match in its prefix store
5. Maintains a prefix store in memory (`HashMap<[u8; 32], BackendAddr>`)

This is exactly what vLLM Router v0.1 does — the 600-line core.

```rust
use sha2::{Sha256, Digest};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

struct PrefixStore {
    // hash -> backend address
    store: RwLock<HashMap<[u8; 32], String>>,
}

impl PrefixStore {
    fn longest_prefix_match(&self, token_ids: &[u32]) -> Option<String> {
        let store = self.store.read().unwrap();
        // Try progressively shorter prefixes until a match is found
        let block_size = 16;
        let mut len = (token_ids.len() / block_size) * block_size;
        while len > 0 {
            let hash = sha256_of(&token_ids[..len]);
            if let Some(backend) = store.get(&hash) {
                return Some(backend.clone());
            }
            len -= block_size;
        }
        None
    }
}
```

**Build steps.**
1. Implement the prefix store with `RwLock<HashMap>` — read lock for routing, write lock for updates.
2. Add a Tokio HTTP server front-end (axum).
3. Simulate two "backends" (just print which backend was selected).
4. Drive with 1K requests — half with a shared 512-token system prompt, half with unique prompts.
5. Measure: what fraction of requests hit the prefix cache? What's the P99 routing latency?
6. Compare to a Python FastAPI equivalent doing the same prefix hashing.

**Reference.** [vLLM Router v0.1 source](https://github.com/vllm-project/vllm/tree/main/router). Read `src/routing.rs` — the actual implementation of the production Rust router.

### 06 — `pyo3-bridge`

**When you want Rust from Python.** You've written a fast Rust tokenizer and router. But your serving stack is Python (vLLM, FastAPI). PyO3 lets you compile Rust into a Python extension module — Python imports it like any other module, calls functions, and gets native-speed execution.

```rust
// lib.rs (compiled to a Python extension via maturin)
use pyo3::prelude::*;

#[pyfunction]
fn tokenize_batch(texts: Vec<String>) -> PyResult<Vec<Vec<u32>>> {
    // parallel Rust tokenization
    let results = texts.par_iter()
        .map(|t| TOKENIZER.encode(t, false).unwrap().get_ids().to_vec())
        .collect();
    Ok(results)
}

#[pymodule]
fn fast_tokenizer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tokenize_batch, m)?)?;
    Ok(())
}
```

```bash
pip install maturin
maturin develop  # compiles Rust + installs as Python package
```

```python
import fast_tokenizer
ids = fast_tokenizer.tokenize_batch(["hello", "world"])  # calls Rust
```

**When to use this vs rewriting in pure Python.** Use PyO3 for: tokenization hot paths, hash computation, any CPU-bound loop that profiling shows taking >10% of request time. Don't use it for: anything that's already I/O-bound, anything already running on GPU.

### 07 — `mistralrs-tour`

**mistral.rs.** A full LLM inference engine in Rust — PagedAttention, continuous batching, speculative decoding, GGUF quantization, all implemented without Python. Built on Candle. Provides an OpenAI-compatible API server.

**Why to read this source.** It's the Rust equivalent of what you built in `systems-for-ml` (mini-serve, mini-vllm) and Level 5 (production engines). Reading it shows you what the full production stack looks like when you remove Python entirely from the critical path.

**What to look for in the source.**
- [`src/pipeline/`](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs-core/src/pipeline): the continuous batching scheduler — compare to vLLM's Python scheduler from systems-for-ml Level 4.
- [`src/paged_attn/`](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs-core/src/paged_attn): PagedAttention in Rust — the block manager, the KV block table, the eviction policy.
- [`src/utils/tokens.rs`](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs-core/src/utils): sampling strategies (top-k, top-p, temperature).

**Build steps.** Clone mistral.rs. Run the getting-started example with Qwen2.5-0.5B. Compare throughput to Python vLLM on the same model. The gap will be smaller than you expect — both end up calling the same CUDA kernels (FlashAttention, cuBLAS). The Rust layer is thin but real for CPU-side latency.

## Project this week

```
compiler-and-kernels/
└── rust_infra/
    ├── fast_tokenizer/            # Rust tokenizer (maturin project)
    │   ├── Cargo.toml
    │   └── src/lib.rs
    ├── prefix_router/             # Rust HTTP router
    │   ├── Cargo.toml
    │   └── src/main.rs
    ├── candle_demo/               # Pure Rust inference
    │   ├── Cargo.toml
    │   └── src/main.rs
    └── reports/
        └── level9-rust.md        # benchmark table + when-to-use guide
```

**Benchmark table:**

| Component | Python | Rust | Speedup |
|---|---|---|---|
| Tokenization (single) | | | |
| Tokenization (batch 1K) | | | |
| HTTP routing P50 | | | |
| HTTP routing P99 | | | |
| Cold start (serve first token) | | | |

## Definition of done

- [ ] You have a working Rust tokenizer using the `tokenizers` crate with benchmark numbers vs Python.
- [ ] You have a working Rust prefix router with the hash-based prefix matching logic.
- [ ] You understand the vLLM Router Rust source well enough to explain its routing algorithm.
- [ ] You have read the mistral.rs PagedAttention implementation.
- [ ] `reports/level9-rust.md` includes the benchmark table and a when-to-use guide.

## Resources

- **The Rust Book** — [doc.rust-lang.org/book](https://doc.rust-lang.org/book/).
- **tokenizers crate** — [github.com/huggingface/tokenizers](https://github.com/huggingface/tokenizers).
- **Candle** — [github.com/huggingface/candle](https://github.com/huggingface/candle).
- **mistral.rs** — [github.com/EricLBuehler/mistral.rs](https://github.com/EricLBuehler/mistral.rs).
- **vLLM Router (Rust)** — [github.com/vllm-project/vllm/tree/main/router](https://github.com/vllm-project/vllm/tree/main/router).
- **vLLM Router v0.1 Iris blog** — [blog.vllm.ai/2026/01/05/vllm-sr-iris.html](https://blog.vllm.ai/2026/01/05/vllm-sr-iris.html).
- **axum** — [github.com/tokio-rs/axum](https://github.com/tokio-rs/axum). The standard Rust HTTP framework.
- **maturin** — [github.com/PyO3/maturin](https://github.com/PyO3/maturin). Build and publish PyO3 extensions.
- **Tokio tutorial** — [tokio.rs/tokio/tutorial](https://tokio.rs/tokio/tutorial).

## What you'll be able to do after this week

> Write production Rust for ML infrastructure: fast batch tokenization, async HTTP routing with prefix-hash logic, and PyO3 bridges for hot-path extraction from Python. Read and navigate the mistral.rs and vLLM Router Rust codebases. Know exactly when Rust adds value in the ML serving stack and when it doesn't.
