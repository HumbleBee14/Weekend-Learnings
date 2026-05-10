"""
Measure dataloader throughput in isolation, with and without sequence packing.
This is the input for G17 — compare to model step throughput from Topic 02.

Run:
    python loader_throughput.py --packing 0
    python loader_throughput.py --packing 1
"""

import argparse
import time
import torch
from torch.utils.data import DataLoader, IterableDataset


class FakeTokenStream(IterableDataset):
    """Simulates pre-tokenized shards. Each item is a sequence of tokens."""

    def __init__(self, n_seqs: int = 50_000, lo: int = 32, hi: int = 1024, vocab: int = 32_000):
        self.n_seqs = n_seqs
        self.lo, self.hi = lo, hi
        self.vocab = vocab

    def __iter__(self):
        gen = torch.Generator().manual_seed(42)
        for _ in range(self.n_seqs):
            length = int(torch.randint(self.lo, self.hi + 1, (1,), generator=gen).item())
            yield torch.randint(0, self.vocab, (length,), generator=gen)


def collate_padded(batch, max_len: int = 1024):
    out = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, t in enumerate(batch):
        out[i, : len(t)] = t
    return out


class PackedCollator:
    def __init__(self, max_len: int = 1024):
        self.max_len = max_len
        self.buf = torch.empty(0, dtype=torch.long)

    def __call__(self, batch):
        flat = torch.cat([self.buf] + list(batch))
        n_full, rem = divmod(flat.numel(), self.max_len)
        out = flat[: n_full * self.max_len].view(n_full, self.max_len)
        self.buf = flat[n_full * self.max_len :]
        return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--packing", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--steps", type=int, default=200)
    args = p.parse_args()

    ds = FakeTokenStream()
    if args.packing:
        collate = PackedCollator(max_len=1024)
    else:
        collate = collate_padded
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=True,
    )

    it = iter(loader)
    # warmup
    for _ in range(5):
        next(it)

    t0 = time.time()
    n_tokens_real = 0
    n_tokens_with_pad = 0
    for _ in range(args.steps):
        b = next(it)
        n_tokens_with_pad += b.numel()
        # for non-packed, real = nonzero count (a stand-in for "useful" tokens)
        n_tokens_real += int((b != 0).sum().item()) if not args.packing else b.numel()
    dt = time.time() - t0

    print(f"packing={args.packing}  batch={args.batch}  workers={args.workers}")
    print(f"  wallclock           : {dt*1000:.0f} ms for {args.steps} batches")
    print(f"  tokens/sec (with pad): {n_tokens_with_pad/dt:,.0f}")
    print(f"  tokens/sec (useful) : {n_tokens_real/dt:,.0f}")
    print(f"  pad overhead        : {(1 - n_tokens_real/n_tokens_with_pad)*100:.1f}%")


if __name__ == "__main__":
    main()
