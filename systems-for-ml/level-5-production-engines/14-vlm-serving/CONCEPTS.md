# 14 — VLM Serving

By 2026, multimodality is the default. Qwen2.5-VL, Pixtral, LLaVA-OneVision, Gemini, GPT-4o, Claude 3.7 — all multimodal. Most "agentic" workloads (browse-the-web agents, screenshot understanding, document QA) need image input. Serving text-only LLMs and not VLMs covers half the production surface.

## The architecture

```
                ┌─────────────────────────────────────────────────────┐
                │  Vision encoder (ViT-style; SigLIP / CLIP-like)     │
   image  ───►  │  patches the image → emits visual tokens            │
                │  ~256-576 tokens for a 336x336 image                │
                │  (~16K for 1080p at full resolution)                │
                └─────────────────────────────────────────────────────┘
                                    │
                                    ▼
                ┌─────────────────────────────────────────────────────┐
                │  Projector (small MLP or cross-attention)           │
                │  maps visual tokens into the LLM's embedding space  │
                └─────────────────────────────────────────────────────┘
                                    │  visual tokens (LLM-token-shaped)
                                    ▼
                ┌─────────────────────────────────────────────────────┐
                │  LLM decoder (regular causal transformer)           │
   text   ───►  │  prepends/interleaves visual tokens with text       │
                │  autoregressive generation as usual                 │
                └─────────────────────────────────────────────────────┘
                                    │
                                    ▼  output tokens
```

Three forward passes per request: vision encoder, projector, LLM. The first two are *not* batched the way LLM prefill is — different model, different shape, different optimal batch size.

## What's different from text-only serving

### 1. Two-stage prefill

The vision encoder runs first. It's compute-heavy but with no KV cache (it's not autoregressive — it's a single forward pass over the image). Then the LLM prefill consumes the visual tokens + text tokens.

**Implication:** the engine has to schedule two distinct pieces of work per request. vLLM and SGLang both handle this in 2026; older codebases (or naive implementations) treat the vision pass as a separate service.

### 2. Variable visual token counts

Different image sizes produce different numbers of visual tokens. Qwen2.5-VL goes from ~64 visual tokens (small thumbnail) to ~16K (1080p preserved at full resolution). Padding waste is *worse* than text padding waste because the variance is bigger.

```
Image size      Visual tokens (Qwen2.5-VL, native res)
──────────      ──────────────────────────────────────
256 x 256       ~64
448 x 448       ~256
672 x 672       ~576
1024 x 1024     ~1280
1920 x 1080     ~16400
```

The engine usually exposes a `max_pixels` knob to cap this — Qwen2.5-VL recommends `max_pixels=1280*28*28` (~1M pixels) for serving. Without it, one user sending a full-res screenshot can blow your KV budget.

### 3. KV cache for visual tokens looks weird

The LLM's KV cache contains entries for visual tokens that have no real "text" meaning. Prefix caching across requests has to consider whether the **same image** was sent — a hash of the image bytes (or its visual-token output) is needed in addition to the text prefix hash.

vLLM handles this via `extra_keys` in the prefix-cache hash chain: the image hash becomes part of the chain. Two requests with the same image and same text prefix → cache hit. Two requests with different images and same text → no false-positive hit.

### 4. Memory pressure

```
Request:   1 image at 1024x1024 + 256 text tokens, max 512 output
Visual tokens:   ~1280
Text + visual:   ~1536 (prefill)
Output:          ~512
KV cache size:   ~2K tokens × 2 (K and V) × n_layers × hidden_dim × bytes

Rough on a 7B at FP8 KV: ~80 KB per token × 2K tokens = 160 MB per request.
At batch=8 with images: 1.3 GB just for in-flight KV. Doable, but cap max_pixels.
```

### 5. Encoder-decoder placement

In a heterogeneous setup, the vision encoder might run on a smaller GPU (it's compute-light relative to the LLM). This is the **disaggregation pattern from Topic 08, generalized to VLMs**: vision-encoder pool on L4s, LLM-decoder pool on H100s, projected visual tokens transferred between them. NVIDIA Dynamo and llm-d both support this shape; vLLM is gaining native support.

## Production engines in 2026

```
Engine          VLM support
──────          ───────────
vLLM            Native since v0.6 (mid-2024). Qwen2-VL, Qwen2.5-VL,
                Pixtral, LLaVA, MiniCPM-V, Phi-3-Vision, Phi-4-MM, Gemma-3-VL.
                Standard LLM(model=...) API; image input via the OpenAI
                content-list message shape.
                V1 chunked prefill handles 16K-token visual prefills.

SGLang          Added 2024. Ahead on prefix-cache deduplication for
                repeated images (the radix tree extends naturally).

TRT-LLM         Supported but multimodal pipeline is more setup-heavy.
                NIM packages for VLMs available.

llama.cpp       LLaVA family supported via llava.cpp; Qwen2-VL via
                community PRs; less polished than vLLM.

MLC-LLM         WebLLM supports several VLMs in-browser; novel use case.
```

## OpenAI shape for VLM requests

```python
client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            {"type": "text", "text": "What's in this image?"},
        ],
    }],
    max_tokens=256,
)
```

Standard OpenAI multi-content shape. Same client code as text-only requests.

## Pitfalls

1. **No `max_pixels` cap.** A user sending a 4K screenshot can saturate your KV cache. Always cap.
2. **Forgetting image-hash prefix-cache integration.** Two users sending the same image should share KV; without `extra_keys` they don't. And two users sending *different* images must never share — that's a quality bug.
3. **Treating vision-encoder time as decode time.** Profile the three phases separately (encoder / projector / LLM). Encoder is often 30-50% of total prefill cost on small text + big image.
4. **Same batch for vision and LLM.** Different optimal batch sizes. The engine should schedule them separately; if you wrote your own server, this is the trap to avoid.
5. **Skipping multi-image requests.** Most modern VLMs support N images per request (Qwen2.5-VL goes to ~10). Memory budgeting must account for that.

## What to do this topic

1. Serve Qwen2.5-VL-7B-Instruct via vLLM (or smaller — Qwen2-VL-2B if memory is tight).
2. Send a request with one image + one text question.
3. Measure: vision-encoder time, LLM prefill time, decode time. Three numbers, very different.
4. Send a batch of 8 requests with the *same* image, different text prompts. Measure with and without prefix caching enabled. The visual-token portion of the prefix should hit cache.
5. Send 8 requests with *different* images. Confirm no false-positive cache hits (would be a quality bug).
6. Stress test: 8 requests with very different image sizes (256² to 1024²). Watch the visual token count vary; observe padding behavior.

## References

- Qwen2.5-VL paper — https://arxiv.org/abs/2502.13923
- LLaVA family — https://llava-vl.github.io/
- vLLM multimodal docs — https://docs.vllm.ai/en/latest/usage/multimodal_inputs.html
- vLLM supported VLMs — https://docs.vllm.ai/en/latest/models/supported_models.html#multimodal-language-models
- SGLang multimodal — https://docs.sglang.ai/backend/multimodal.html
- Pixtral (Mistral) — https://mistral.ai/news/pixtral-12b/
- LLaVA-OneVision — https://arxiv.org/abs/2408.03326
- Phi-4-multimodal — https://huggingface.co/microsoft/Phi-4-multimodal-instruct
