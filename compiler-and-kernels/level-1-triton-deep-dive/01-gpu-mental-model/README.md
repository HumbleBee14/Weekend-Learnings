# 01 — GPU mental model

You write no code in this sub-module. You read [`CONCEPTS.md`](CONCEPTS.md), then you answer the eight diagnostic questions in [`diagnostics.md`](diagnostics.md) in your own words in [`notes.md`](notes.md). If you can answer them without looking back at the doc, the rest of Level 1 will make sense. If you cannot, re-read until you can — the time you spend here is paid back many times in the next five sub-modules.

The cost of skipping this and faking it later: you write Triton kernels that "work" but make no sense to you, and when they are slow you have no model for why. The cost of doing it now: about two hours.

## What to do

1. Read [`CONCEPTS.md`](CONCEPTS.md) front to back. It is ~25 minutes of reading.
2. Open [`diagnostics.md`](diagnostics.md). For each of the eight questions, write your answer in [`notes.md`](notes.md) — without scrolling back to `CONCEPTS.md`. Closed-book. The answers should be a few sentences each, sometimes with small arithmetic.
3. Open the answer key in `CONCEPTS.md` (at the bottom — collapsed by default in your mental model, expanded in the file). Compare. If you got fewer than 6/8 confidently right, re-read and try again the next day.
4. Bonus: open NVIDIA's H100 datasheet and your own GPU's spec sheet (or Colab T4 if that's what you have) and find the actual numbers for: number of SMs, SRAM per SM, HBM bandwidth, FP16 tensor-core TFLOPS. Write them in `notes.md`. Knowing the real numbers for your hardware is what turns abstract performance discussion into concrete reasoning.

## Files in this folder

- `CONCEPTS.md` — the read.
- `diagnostics.md` — the eight questions.
- `notes.md` — your answers and hardware-spec table. Template included.

## Where this goes next

Sub-module 02 puts the first Triton kernel on a GPU. Everything you'll write — `tl.program_id`, `tl.load` with `mask`, `tl.dot`, `BLOCK_SIZE` — has a direct hardware meaning. The diagnostic questions here are the meanings you'll need.
