# Prompt 03 — Multi-LoRA Serving for 500 Fine-Tunes

Your company sells a "bring your own data, get your own model" service. Each customer fine-tunes their own LoRA adapter on top of a shared Llama-3-8B base.

Current state:
- **500 LoRA adapters** in the catalog, growing 10/week
- ~50 active LoRAs at any moment (long tail is cold)
- Adapter sizes: ~50 MB each (rank 16)
- **Per-customer SLO:** p95 TTFT < 500ms when their adapter is "warm"; < 5s acceptable when cold
- Per-customer rate limits, per-customer billing
- 200 QPS aggregate across all customers

Design it. 45-minute interview format.
