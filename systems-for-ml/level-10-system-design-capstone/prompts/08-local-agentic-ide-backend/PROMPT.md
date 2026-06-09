# Prompt 08 — Local-First Agentic IDE Backend

You're building the local-mode backend for an agentic IDE (think Cursor's local mode, or a privacy-first Continue.dev). Requirements:

- **Runs entirely on the user's Mac** (M-series, 32-64GB RAM)
- **TTFT < 100ms** for the autocomplete/inline-completion case
- **Sub-second** for the chat case
- Multiple models running concurrently:
  - Tiny model (3B) for autocomplete — always loaded
  - Medium model (8B-13B) for inline edits
  - Large model (32B-70B Q4) for chat / deep reasoning
- Zero per-request cost; user owns the hardware
- **Personalization via QLoRA** on the user's own code style
- **Privacy:** nothing leaves the device

Design it. 45-minute interview format.
