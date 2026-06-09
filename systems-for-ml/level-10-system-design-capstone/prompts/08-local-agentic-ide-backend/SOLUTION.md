# Prompt 08 — Worked Solution

> The most differentiated prompt in the set. Most candidates have never thought seriously about local-first AI architecture. If you've done Level 8, you'll stand out hard here — there's almost no public curriculum that covers this depth.

## 1. Clarifying questions (the first 3 minutes)

1. **Target hardware spread.** Is the floor M2 Pro (16-24GB RAM) or M3+ (24-128GB)? The product can serve very different model tiers depending. *Assumed: M3-and-up, 24GB minimum; M4 Max / M5 Max as the "ideal experience."*
2. **Online fallback OK or strictly offline?** Some products allow "if user opts in, occasionally call Claude for the hard cases." Strict offline = much harder. *Assumed: strict offline by default, optional cloud opt-in per user.*
3. **Concurrent model assumption.** Does the user trigger autocomplete while a long chat is generating? Almost certainly yes. *Assumed: 3 models can be active simultaneously; need memory + scheduling story.*
4. **Personalization scope.** QLoRA on user's own code only, or include their reading history / messages / etc.? Affects training data sensitivity and adapter complexity. *Assumed: code only, scoped to the user's repos they explicitly opt in.*
5. **Update channel.** Are model updates user-driven (App Store updates) or sideloaded (Ollama-style pulls)? *Assumed: Ollama-style — the daemon pulls new model versions on a schedule the user controls.*
6. **Privacy/threat model.** What does "private" mean? No data leaves the machine, period? Or "no third-party telemetry; OS-level access by Apple is acceptable"? *Assumed: nothing leaves the device by default; PCC-style opt-in for occasional cloud overflow.*

## 2. The right answer in one sentence

**A local daemon that multiplexes three model tiers — a 3B autocomplete model (MLX, always loaded, sub-100ms TTFT), an 8-13B inline-edit model (MLX, lazily loaded), and a 32-70B chat model (llama.cpp with Metal backend + Q4_K_M quant, lazily loaded) — all sharing UMA via a memory-pressure-aware scheduler that demotes/evicts under OS pressure, with per-user QLoRA personalization fine-tuned nightly as a background job and Apple Foundation Models framework reserved for the system-3B fallback.**

The senior signal: **mixing frameworks deliberately.** MLX for the smaller/medium tier (Apple-native, fastest on M5 Neural Accelerators); llama.cpp Metal for the 70B class (more mature on huge quantized models, GGUF ecosystem). Most candidates pick one framework for everything; production local apps pick per model.

## 3. The architecture (whiteboard)

```
   ┌────────────────────────────────────────────────────────────┐
   │  IDE (Swift / Electron / VS Code extension host)           │
   │  ─ user types → triggers autocomplete request              │
   │  ─ user hits ⌘K → inline-edit request                      │
   │  ─ user opens chat panel → chat request                    │
   └────────────────────────┬───────────────────────────────────┘
                            │ local Unix socket / gRPC over loopback
                            ▼
   ┌────────────────────────────────────────────────────────────┐
   │  Local model daemon (Rust, runs as background process)     │
   │                                                            │
   │  ┌──────────────────────────────────────────────────────┐  │
   │  │  Memory-pressure scheduler                           │  │
   │  │  ─ monitors os_proc_available_memory() every 1s      │  │
   │  │  ─ tier 1 (3B):   always resident, ~2GB             │  │
   │  │  ─ tier 2 (8-13B): resident if memory > 24GB         │  │
   │  │  ─ tier 3 (70B Q4): resident if memory > 48GB        │  │
   │  │  ─ on pressure event: demote tier 3 first            │  │
   │  └──────────────────────────────────────────────────────┘  │
   │                                                            │
   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
   │  │ 3B Qwen2.5   │  │ 13B Qwen-    │  │ 70B-class chat   │  │
   │  │ Coder        │  │ Coder        │  │ (Llama-3 / Qwen) │  │
   │  │ (MLX,        │  │ (MLX, FP8)   │  │ (llama.cpp Metal │  │
   │  │  FP16/M5 NA) │  │              │  │  Q4_K_M GGUF)    │  │
   │  │ ALWAYS HOT   │  │ on-demand    │  │ on-demand        │  │
   │  └──────────────┘  └──────────────┘  └──────────────────┘  │
   │                                                            │
   │  ┌──────────────────────────────────────────────────────┐  │
   │  │  Adapter manager (user QLoRA)                        │  │
   │  │  ─ user's personalization LoRA, hot-swapped via      │  │
   │  │    mlx_lm.unwrap_lora() per request                  │  │
   │  │  ─ nightly fine-tune job (apple/mlx-lm QLoRA)        │  │
   │  └──────────────────────────────────────────────────────┘  │
   │                                                            │
   │  ┌──────────────────────────────────────────────────────┐  │
   │  │  Fallback router                                     │  │
   │  │  ─ Apple Foundation Models (3B, system-provided)     │  │
   │  │    used when our daemon is starting up or evicted    │  │
   │  │  ─ Optional cloud (Claude API, opt-in only) for      │  │
   │  │    very long contexts (>100K tokens)                 │  │
   │  └──────────────────────────────────────────────────────┘  │
   └────────────────────────────────────────────────────────────┘

   ┌────────────────────────────────────────────────────────────┐
   │  Background workers (non-realtime)                         │
   │  ─ Nightly QLoRA fine-tune on user's code (LaunchAgent     │
   │    at 2am if Mac plugged in + idle > 5 min)                │
   │  ─ Model pull / update via signed manifests (Ollama-style) │
   │  ─ Personalization data: code corpus indexed locally       │
   │    (~/.local/agentic-ide/personalize/), never uploaded     │
   └────────────────────────────────────────────────────────────┘
```

### Five-box mapping (adapted for local)

The traditional five boxes still map, but to a single process:

- **Gateway = local Unix socket / gRPC over loopback.** No TLS needed (loopback only); rejects non-local connections.
- **Router = memory-pressure scheduler.** Decides which model tier handles which request. Autocomplete → always tier 1 (latency-critical). Inline-edit → tier 2 if available, fallback tier 1 with degraded quality if not. Chat → tier 3 if available, fallback tier 2.
- **Scheduler = within-model batching.** MLX and llama.cpp both do this internally; we just don't fight it. Autocomplete requests rarely batch (latency-critical, 1-at-a-time); chat requests can batch if multiple panels open.
- **Worker = MLX or llama.cpp inference, on M-series GPU + Neural Accelerators.** UMA is the secret weapon — KV cache lives in the shared 64-128GB pool.
- **Control plane = the daemon's own config/state.** LaunchAgent runs the fine-tune; signed manifests gate model updates; user preferences in `~/Library/Application Support/`.

## 4. Capacity math — but local

```
Hardware target: M5 Max, 64GB unified memory (the "sweet spot" SKU).

Per-model footprint:
  Tier 1: Qwen2.5-Coder-3B, FP16 in MLX:           ~6 GB
          (M5 Neural Accelerators give ~1300 tok/s decode, ~25K tok/s prefill)
  Tier 2: Qwen2.5-Coder-13B, FP8 (mxfp4) in MLX:   ~10 GB
          (~280 tok/s decode, ~8K tok/s prefill on M5 Max)
  Tier 3: Llama-3-70B Q4_K_M in llama.cpp:          ~42 GB
          (~12-18 tok/s decode, ~400 tok/s prefill on M5 Max)
  User QLoRA adapter:                              ~50 MB
  Daemon overhead + buffer:                        ~2 GB

All three loaded: 6 + 10 + 42 + 2 = ~60 GB

  THE CATCH (the detail that separates a real Mac engineer from a bluffer):
  Metal's default GPU-usable budget (recommendedMaxWorkingSetSize) is ~75% of
  unified memory above 32GB — so a 64GB Mac only exposes ~48GB to the GPU by
  default. 60GB of resident models will NOT fully offload. You must raise the
  cap: `sudo sysctl iogpu.wired_limit_mb=57344` (resets on reboot; eats OS/CPU
  headroom). Realistically, all-three-resident is a 96GB+ machine; on 64GB you
  keep tier 1 + tier 2 hot and load tier 3 on demand.

On 32GB Mac (M4/M5 base):
  Can run tier 1 + tier 2 simultaneously (~18 GB)
  Tier 3 only if we evict tier 2 — chat becomes a serial experience

On a 96GB+ Mac (Max/Ultra tier):
  All three loaded with comfortable headroom, even with the raised Metal cap.
  ~64GB of GPU-usable budget after the 75% cap easily holds the ~60GB working set plus three KV caches at usable context.
```

### TTFT budget per tier

```
Tier 1 (3B autocomplete):
  Goal: <100ms TTFT
  Prefill of typical 1K-token context @ 25K tok/s = 40ms
  + first decode token ~5ms
  + IPC roundtrip ~5ms
  Total: ~50-80ms ✓ comfortably under 100ms

Tier 2 (13B inline-edit):
  Goal: sub-second TTFT for a 2K-token context
  Prefill @ 8K tok/s for 2K context = 250ms
  + first decode token ~10ms
  Total: ~280-350ms ✓

Tier 3 (70B chat):
  Goal: sub-2s TTFT for a 4K-token chat history
  Prefill @ 400 tok/s for 4K context = 10 seconds ✗ TOO SLOW

  This is the binding constraint. Fix:
  ─ Prefix cache the conversation history (llama.cpp supports this)
  ─ Only first turn pays the full prefill; subsequent turns ~instant
  ─ Stream output so user sees something within ~200ms regardless
```

The first-turn-of-a-long-chat is the local-first product's biggest UX challenge. Solution: stream a "thinking..." status, then stream tokens. The 10-second prefill is hidden under perceived "the assistant is reading my code."

## 5. The hard parts unique to local

### 5.1 Memory pressure is a first-class constraint, not an edge case

On a fleet of cloud GPUs, OOM is rare and well-handled (kill replica, route around). On a user's Mac, OOM = OS swap thrashing = entire system unusable = user opens Activity Monitor, finds our daemon using 60GB, force-quits, never installs again.

**Mitigation: aggressive pressure-aware demotion.**

```swift
// Pseudocode — Apple's MemoryStatus framework or os_proc_available_memory()
func onMemoryPressure(_ level: PressureLevel) {
    switch level {
    case .normal:    break
    case .warning:   evictTier3IfIdle()       // 70B chat goes first
    case .critical:  evictTier2IfIdle()        // 13B inline-edit next
    case .emergency: evictTier1AndFallToFM()   // last resort: lean on
                                               // Apple Foundation Models
    }
}
```

The 4GB-margin design in the capacity table assumes the user *might* open Lightroom or Final Cut. When they do, we gracefully degrade rather than fight the OS for memory.

### 5.2 MLX vs llama.cpp — knowing where each wins

| Concern | MLX | llama.cpp |
|---|---|---|
| Apple Silicon native | Yes, built for it | Yes, Metal backend |
| Performance on M5 NA | **20-50% faster** than llama.cpp | Slower; doesn't fully exploit Neural Accelerators yet |
| GGUF ecosystem | No (uses safetensors) | **Huge — every 4-bit quant model lives here** |
| Quantization options | FP16, Int8, MXFP4 | **GGUF: Q4_K_M, Q5_K_M, Q6_K, Q8 — much wider** |
| LoRA support | Native | Partial |
| Maturity | Newer, still rapidly evolving | Production-hardened since 2023 |
| Best for | Smaller / FP-precision models on M5+ | Larger Q4-quantized models |

The 2026 reality: **MLX wins on small-medium models on M5+** (20-50% faster per Apple's own benchmarks), but **llama.cpp wins on giant Q4-quantized models** because the GGUF ecosystem has 70B/405B Q4_K_M models ready, and llama.cpp's quantization is more mature.

Pick per model. Don't religiously pick one for everything.

### 5.3 Concurrent inference — autocomplete must not stall behind chat

Autocomplete fires keystroke-by-keystroke (debounced). It cannot wait 5 seconds because a chat is generating in the background. Two ways to handle:

```
Option A: GPU time-slicing (Metal command queue priority)
  ─ Autocomplete kernels submitted at high priority
  ─ Chat kernels submitted at default priority
  ─ Metal interleaves at command-buffer granularity
  ─ Works but autocomplete still sees ~10-15ms tail latency from chat

Option B: Dedicated MLX device per tier (M5 Max has 40 GPU cores)
  ─ Pin tier 1 model to GPU cores 0-7 (always available)
  ─ Tier 2/3 use cores 8-39
  ─ Pure isolation, autocomplete unaffected
  ─ Slightly less throughput on chat (uses 32/40 cores instead of all 40)

Production choice: B. The 20% chat-throughput hit is worth perfect autocomplete latency.
```

### 5.4 On-device QLoRA personalization (nightly job)

Spec: every night at 2am, if Mac is plugged in and idle, run an MLX QLoRA fine-tune on the user's code from the last 7 days.

```python
# Pseudocode using mlx_lm
from mlx_lm import lora

corpus = load_user_code_diffs(last_days=7)  # from local index
                                            # ~10K examples typical

lora.train(
    base_model="Qwen2.5-Coder-13B-FP8",
    train_data=corpus,
    rank=16,
    alpha=32,
    lr=1e-4,
    iters=1000,
    save_path="~/Library/Application Support/agentic-ide/loras/v_N.safetensors"
)
```

On M5 Max, 1000 iters on 10K examples × 13B = ~25-40 min. User wakes up to a model that knows their code style — and **none of their data ever left the machine**.

Catastrophic forgetting check: every nightly LoRA is auto-evaluated on a fixed "general coding" benchmark; if performance drops >5%, the new LoRA is rejected and yesterday's stays active.

### 5.5 Privacy threat model — what "local" actually buys

```
Threat                                  Local-first protection
──────────────────────────────────────────────────────────────────────
Cloud API logging of your code          ✓ Nothing leaves device
Cloud breach exposes your data          ✓ No data in cloud
Network adversary sees prompts          ✓ No network traffic
Insider at your AI vendor reads convos  ✓ No vendor
Your code becomes training data         ✓ Stays on disk
Malware on your Mac reads your code     ✗ Same risk as before — local
                                           AI is no worse than your IDE
                                           already having access
OS-level (Apple) collects telemetry     ✗ Apple's privacy policy applies
                                           (you're already trusting macOS)
Physical seizure of your laptop         ✗ FileVault helps but not perfect
```

The win is real but limited. The talking points: *nothing in transit, nothing in cloud logs, nothing in third-party telemetry, no vendor lock-in for prompts.* The thing it doesn't fix: someone with shell on your machine.

For super-paranoid customers: Apple's Private Cloud Compute (PCC) pattern — *if* we ever need to spill over to cloud (e.g., 100K-token refactor), use PCC-style attested enclave where Apple cryptographically proves the data is processed without persistence.

## 6. Break-it list

| Failure | What happens | Mitigation |
|---|---|---|
| User opens Lightroom while chatting → OOM | OS forces our daemon to swap, latency explodes | Memory-pressure scheduler evicts tier 3 immediately; chat degrades to tier 2; user notified |
| Disk full → can't load 70B GGUF | Tier 3 fails on swap-in | Pre-flight check before loading; graceful fallback to tier 2; surfaced clearly to user |
| User on 16GB M2 | Can't even run our minimum tier 1 + tier 2 | Detect at install; recommend running tier 1 only; offer optional cloud opt-in |
| Nightly QLoRA degrades the user's experience | Personalization makes general queries worse | Auto-eval per nightly LoRA; reject if regression > 5% on general benchmark; keep yesterday's LoRA |
| Model update introduces bug | Bad model ships to all users via auto-update | Staged rollout (1% / 10% / 100% over 7 days); client-side anomaly detection (e.g., output token rate drops) auto-rolls back |
| User's Mac too cold (recently woken) | First request after wake takes seconds (page faults) | Warm-up phase: on app open, fire 1 dummy request per loaded tier to fault-in pages; ~3s startup cost |
| Concurrent inference between chat and autocomplete | Autocomplete stutters | Dedicated GPU core pinning (option B above); validated via Instruments traces |
| Apple Foundation Models framework changes API | Our fallback breaks on new macOS | Pin to a known-good API version; test on every macOS beta; have a non-FM fallback path |
| Researcher injects prompt into a code comment | Local model exfiltrates context to cloud opt-in | Output filter strips network-y outputs; the cloud opt-in is gated by explicit user click per session |
| User's code corpus contains secrets | QLoRA learns to emit secrets | Pre-fine-tune scrubbing: detect & redact API keys, tokens, passwords before training |

## 7. What changes at 10× scale (10× users, OR 10× model size)

Local-first doesn't scale the way cloud does — there's no fleet. "10×" means either *10× users* (millions on different Macs) or *10× model size* (taking advantage of the next Apple Silicon generation).

### 10× users
- **Distribution becomes the hard problem.** Pushing 40GB model files to millions of devices needs P2P (BitTorrent-style) or aggressive CDN regions.
- **Telemetry without privacy violation.** Want to know if model X regresses for users — can only collect *aggregate* metrics (latency histograms, error rates), never user data. Differential privacy on rollups.
- **A/B testing models on-device.** Half of users get model v2; compare aggregate latency/quality without per-user telemetry.
- **Support load.** Each user's Mac is its own broken environment. "It's slow on my M2" is now 1M tickets. Solution: in-app diagnostics that reproduce the issue locally and produce shareable bug reports (no user data, just performance traces).

### 10× model size (700B Q4-class running locally — the M7 Ultra era)
- **Distributed inference across Macs in the same home/office.** Apple's [exo](https://github.com/exo-explore/exo)-style distributed inference over Thunderbolt 5 — multiple Macs split a giant model.
- **The "office cluster" pattern.** A small business has 4 M-series Macs networked over Thunderbolt 5; one of them runs the model server, others contribute compute. This is real today (exo, Petals) and gets more practical with TB5 + M7 generation.
- **Hybrid edge↔cloud for the long tail.** Most queries local; the 5% that need 200K context (refactor across whole codebase) spill to PCC-attested cloud. Per-request opt-in.

### Across both
- **Custom Metal shaders for hot paths.** At 10× users, even a small kernel win matters. Sibling `compiler-and-kernels/` track, Metal/MSL specifically.
- **MLX upstream contributions.** Open-source contributions to MLX core for the patterns our users hit — speculative decoding, batched LoRA, etc. We become a maintainer.
- **Foundation Models framework integration on iOS.** The same daemon, packaged for iPad / iPhone, using the Apple-system models (smaller but free) plus on-device fine-tunes. Now the product is "agentic IDE that works the same on your Mac and your iPad."

## 8. The 30-second summary

> "Local-first agentic IDE backend on Apple Silicon UMA means three model tiers in one daemon: a 3B Qwen-Coder in MLX always hot for autocomplete with sub-100ms TTFT, a 13B in MLX for inline edits, and a 70B Q4 in llama.cpp Metal for chat. Memory-pressure scheduler aggressively demotes the larger tiers when the OS gets pressure — staying under 60GB on a 64GB Mac with 4GB margin. Per-tier GPU core pinning prevents chat from stalling autocomplete. Nightly on-device QLoRA personalizes to the user's code without anything leaving the device. MLX wins on the small/medium tier — 20-50% faster than llama.cpp on M5 Neural Accelerators — but llama.cpp wins on the giant Q4 model because the GGUF ecosystem is more mature. Apple Foundation Models framework is the fallback when our daemon is starting up or evicted. At 10× scale we get into exo-style distributed inference across Macs over Thunderbolt 5."

## What this prompt is really testing

- **Apple Silicon UMA understanding.** Not "small NVIDIA GPU."
- **Mixing frameworks per model tier** — MLX for small/medium, llama.cpp for the giant. Not religious about one framework.
- **Memory pressure as a first-class scheduler input** — the local-first equivalent of KEDA on queue depth.
- **On-device QLoRA workflow** — knowing this is a thing, and how to do it without breaking general performance (catastrophic forgetting check).
- **Concurrent inference design** — GPU core pinning, not just "submit and hope."
- **Privacy threat model honesty** — knowing what local *actually* buys you and what it doesn't.
- **The 10× answer being exo/Thunderbolt-shaped** — not "more cloud."

## References

- [Topic 01 — unified-memory-mental-model](../../../level-8-local-and-on-device/01-unified-memory-mental-model/)
- [Topic 02 — mlx-basics](../../../level-8-local-and-on-device/02-mlx-basics/)
- [Topic 03 — mlx-vs-llama-cpp-vs-mps](../../../level-8-local-and-on-device/03-mlx-vs-llama-cpp-vs-mps/)
- [Topic 04 — m5-neural-accelerators](../../../level-8-local-and-on-device/04-m5-neural-accelerators/)
- [Topic 07 — foundation-models-framework](../../../level-8-local-and-on-device/07-foundation-models-framework/)
- [Topic 08 — local-serving-stack](../../../level-8-local-and-on-device/08-local-serving-stack/)
- [Topic 11 — agentic-ide-backend](../../../level-8-local-and-on-device/11-agentic-ide-backend/) — the most direct parent of this prompt
- [Topic 12 — qlora-on-device](../../../level-8-local-and-on-device/12-qlora-on-device/)
- [Topic 15 — distributed-mac-inference](../../../level-8-local-and-on-device/15-distributed-mac-inference/) — exo, Thunderbolt 5, the 10× answer
- [Topic 16 — privacy-and-pcc](../../../level-8-local-and-on-device/16-privacy-and-pcc/)
- [Apple's MLX research note: *Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU*](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [Reddi Vol 2 *Edge Intelligence* chapter](https://mlsysbook.ai/)
- Kiely §3.5 (Local Inference) — brief but useful framing from the cloud perspective
