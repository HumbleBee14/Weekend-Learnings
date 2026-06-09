# Prompt 08 — Rubric

## Strong signals
- UMA model — explains why local on Apple Silicon ≠ "GPU inference on your laptop"
- Multi-model concurrency with explicit memory budget
- MLX for small/medium, llama.cpp for the largest (knows which framework wins where)
- On-device QLoRA personalization as a separate background workflow
- Privacy threat model — what "local" actually buys

## Anti-signals
- "We'd just run Ollama" — no, you'd use Ollama as the *server* but you still have to design the model topology
- Treats Apple Silicon like a small NVIDIA GPU
- No memory-pressure degradation story
- Single-model design — misses the autocomplete/edit/chat tiering

## What's tested
Level 8 specifically. This is the most differentiated prompt — most candidates have never thought seriously about local-first AI architecture.
