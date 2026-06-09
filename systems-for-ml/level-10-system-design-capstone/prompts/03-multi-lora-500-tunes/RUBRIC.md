# Prompt 03 — Rubric

## Strong signals
- Multi-LoRA hot-swap as a single base model + 500 adapters (not 500 separate services)
- Hot/cold tiering with explicit eviction policy
- Per-customer billing tied to OTel spans + LoRA ID
- Consistent-hash routing for adapter cache locality

## Anti-signals
- "Deploy 500 vLLM instances" — completely misses Topic 10 of Level 5
- No story for the cold-LoRA case
- Confuses LoRA with full fine-tune (different memory budget, different deploy story)

## What's tested
Knowledge of vLLM multi-LoRA mechanics specifically. This is one of the most concrete "did you actually do Level 5" questions.
