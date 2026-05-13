# notes — flexattention

## What got inlined (three sentences per variant)

### ALiBi
(fill after running read_emitted_kernel.py with the alibi score_mod)

### Sliding window
(fill)

### Document mask
(fill)

## What surprised me

- BlockMask sparsity tracks the actual speedup almost linearly. If `block_mask.sparsity() == 0.12` you see ~8x speedup over dense, not 1/0.12 = 8.3x but close.
- `torch.compile` recompiles when the closure variable's *value* changes if it's a Python scalar; tensor closures don't recompile. Use `WINDOW = torch.tensor(1024)` if you want to sweep window sizes without paying compile cost each time.
- The compiled FlexAttention forward includes the backward kernel generation. You get differentiability for free.

## Should I ever write hand-Triton attention instead?

Yes, if:
- Your `score_mod` requires data-dependent control flow (different math per position based on a runtime check that's not pointwise).
- You need a specific memory layout (e.g., quantized KV with non-uniform scales per block) that BlockMask can't express.
- You're benchmarking FlexAttention vs hand-Triton and the gap is >20% — which is rare for pointwise mods.

Otherwise, FlexAttention is the right answer.
