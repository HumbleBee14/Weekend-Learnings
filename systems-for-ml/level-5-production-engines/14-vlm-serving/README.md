# 14 — VLM Serving

## Files

- `CONCEPTS.md` — VLM architecture, what's different from text-only serving, the five complications, 2026 engine support
- `vlm_demo.py` — three scenarios against a vLLM Qwen2.5-VL server: same-image varied-text (prefix-cache hits), different-images (must not false-share), size sweep (visual-token cost scaling)

## Quickstart

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 \
    --max-model-len 8192 --limit-mm-per-prompt image=2

pip install openai pillow
python vlm_demo.py
```

## Expected output (rough shape)

```
[A] Same image, varied text — visual prefix-cache should kick in after #1
   ttft=720ms  total=950ms  q=What primary color is the image?
   ttft=140ms  total=320ms  q=Describe the image in five words.
   ttft=135ms  total=290ms  q=Could this be a sunset?
   first vs subsequent TTFT: 720 vs mean(rest) 138 ms

[B] Different images, same text — must NOT false-share cache
   image #0  total=720ms  text='What primary color is the image?'
   ...

[C] Image size sweep — visual token count varies dramatically
   256x256px   ttft=380ms
   448x448px   ttft=720ms
   672x672px   ttft=1180ms
   1024x1024px ttft=2400ms
```

The 5× TTFT improvement on the second same-image request is the prefix cache hitting on the visual portion. The size sweep shows why `max_pixels` is the most important VLM-serving flag.

## Try

- **Send the same image with `--enable-prefix-caching=False`.** Watch TTFT stay flat — the cache was doing all the work.
- **Send a 1920×1080 screenshot.** May OOM the KV cache; cap `max_pixels` (e.g., `--mm-processor-kwargs '{"max_pixels": 1280000}'`).
- **Profile the three phases.** vLLM exposes per-phase timing in trace logs (`--otlp-traces-endpoint`); send a single request and inspect.
- **Try multi-image requests.** Two images per request, same prompt. Memory budget grows linearly.
- **Try a different VLM** (Pixtral, LLaVA-OneVision, Phi-4-MM). Visual token counts differ; encoder cost differs; prefix-cache behavior should be similar.

## Where this goes

- Topic 13 — the vision encoder is exactly the kind of model TensorRT or ORT could run on a separate small GPU in a disaggregated VLM stack
- Topic 08 — VLM disaggregation generalizes the prefill/decode split: vision encoder pool + LLM pool + KV transport
- Level 8 — local VLM serving on Apple Silicon via MLX is a parallel track
