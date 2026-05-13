# notes — online softmax

## What happens when m_t == m_{t+1}?

Rescale factor = exp(0) = 1. O and ell are unchanged before the tile contribution adds in. FA4 detects this case and skips the rescale entirely — the FMA savings dominate when the running max stabilizes early, which it does on most attention rows.

## What happens with tile size 1?

The recursion still works. You're effectively running per-element online softmax. It will be slower in any real kernel (no SIMD width) but the algebra is identical.

## Pitfalls

- Rescale order: rescale O and ell BEFORE adding the new tile's contribution. Easy to flip; the unit test in `online_attention_np.py` catches it.
- `exp(m - m_new)` when `m == -inf` (first tile): we special-case to `rescale = 0.0` so `ell * rescale = 0`. The Triton version uses `m = -float("inf")` init and the exp underflows cleanly to 0.

## After

You can now read the FA2 paper Algorithm 1 line by line and recognize every term.
