# notes — attention from scratch

## The memory wall

Three sentences explaining what the `memory_wall.py` output means for HBM bandwidth in a real kernel:

(write yours here)

## Sparsity sanity check

- Causal: ~0.5 (lower triangle).
- Sliding causal W=512 at N=8192: ~0.0625 (the diagonal band).
- Document mask at 4 docs of 2048: ~0.25 (block-diagonal).
- Sink+window: a hair above sliding-causal alone.

If the FlexAttention BlockMask in sub-module 06 reports a block-sparsity that roughly matches these fractions, the metadata is wired up correctly.

## Pitfalls hit

(fill as you go)
