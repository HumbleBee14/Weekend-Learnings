# 03 — Data Loading and Tokenization

## Files

- `CONCEPTS.md` — the data-pipeline ceiling, MDS / Energon / WebDataset, sequence packing, G17
- `loader_throughput.py` — measures pure-loader throughput with vs without packing; useful tokens / total tokens / pad overhead

## Quickstart

```bash
python loader_throughput.py --packing 0   # padded
python loader_throughput.py --packing 1   # packed
```

## Expected output

```
packing=0  batch=16  workers=4
  wallclock           : 1820 ms for 200 batches
  tokens/sec (with pad): 1,798,000
  tokens/sec (useful) : 920,000
  pad overhead        : 48.8%

packing=1  batch=16  workers=4
  wallclock           : 1980 ms for 200 batches
  tokens/sec (with pad): 1,653,000
  tokens/sec (useful) : 1,653,000
  pad overhead        : 0.0%
```

The packed-tokens-per-second is ~1.8× the useful-tokens-per-second of padded — that is the GPU work you no longer waste on `<pad>`.

## Try

- `--workers 0` — single-process loader. Watch throughput collapse. This is what happens when `num_workers=0` is left in code by accident.
- `--workers 16` — diminishing returns past CPU count. Eventually hurts because of fork overhead.
- `--batch 64` — bigger batches amortize collate cost. Throughput climbs.
- Combine: time `ddp_train.py` from Topic 02 with this loader plugged in. Measure model tok/s and loader tok/s. If loader < model, you have your G17 wall.

## Build steps for full G17

1. Train the Topic 02 transformer with this loader feeding it.
2. Sweep `seq_len` from 256 to 4096. At each, measure loader-only and model-only tok/s.
3. Plot both lines. Crossover is your wall — below it loader can keep up, above it model starves.

## Where this goes

- Topic 04 (FSDP2) trains a real model; the same loader plugs in
- Topic 10 (torchtitan) uses Mosaic StreamingDataset for the actual run; the principles here transfer directly
