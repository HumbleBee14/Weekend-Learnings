# 09 — Naive KV Cache

## Files

- `CONCEPTS.md` — what the KV cache is, why it makes decode O(N) instead of O(N²), the four problems with contiguous allocation
- `naive_kv_cache.py` — a standalone naive KV cache demonstrating memory waste, hard caps, and lack of prefix sharing

## Quickstart

```bash
pip install torch
python naive_kv_cache.py
```

## Expected output

```
Created cache. Reserved memory: 705 MB
This is reserved REGARDLESS of how many tokens we actually use.

Allocated 3 slots with [100, 200, 5000] tokens.
Memory actually used:  462 MB
Memory reserved total: 705 MB
Wasted:                243 MB
Utilization:           65.5%

Pain points to feel here:
  1. Memory waste — 5 unused slots...
  2. Hard cap...
  3. No prefix sharing...
  4. Internal fragmentation...
```

The 65% utilization is on a *favorable* test (only 3 slots, the longest near max). Production traffic with shorter requests would be much worse — often 20-40% utilization.

## Try

- **Increase max_seq_len to 32768.** Reserved memory grows 4×. Utilization plummets.
- **Try max_batch=16.** More slots, more potential waste.
- **Add a 9000-token request.** Will fail with the hard-cap error from `append`.
- **Free slot 0, allocate again** — the gap is reusable only if you maintain the slot. Fragmentation in real settings is more subtle (tracker complexity grows).

## What you should walk away with

- Why naive KV cache wastes so much memory in practice
- Why mixed-length workloads especially suffer
- A working baseline implementation to compare against the paged version (Topic 10)

## Where this goes

Topic 10 is paged KV cache — the real-world fix. You'll allocate fixed-size *blocks* (16 tokens each), reference them through a per-request *block table*, and reclaim free blocks dynamically. This is what vLLM does. Same interface (`append`, `read`), completely different storage.
