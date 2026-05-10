# 03 — Data Loading and Tokenization

The data pipeline is where most "GPU underutilized" diagnoses end. Optimize the model all you want — if the loader can't feed it, nothing matters.

## The data-pipeline ceiling

GPUs in 2026 chew through tokens fast. A 7B BF16 forward+backward at TP=1 on H100 is roughly 0.5 ms per 2K-token batch. That demands ~4M tokens/sec sustained from the dataloader. Reading from a remote object store, decompressing, tokenizing online — none of this hits 4M/s without engineering.

Symptoms of a starved loader:
- GPU utilization (`nvidia-smi`) bounces 0% → 90% → 0%
- `torch.profiler` shows long gaps between steps with the GPU idle
- `torch.cuda.synchronize()` time per step inflates with no kernel changes

Diagnose first by measuring the loader independently:

```python
loader_iter = iter(loader)
t0 = time.time()
total_tokens = 0
for _ in range(100):
    batch = next(loader_iter)
    total_tokens += batch.numel()
print(f"loader-only tok/s: {total_tokens / (time.time() - t0):,.0f}")
```

If that number is below the model's consumption rate, the loader is the wall.

## The 2026 stack

### Mosaic StreamingDataset (MDS)

[github.com/mosaicml/streaming](https://github.com/mosaicml/streaming). Drop-in `IterableDataset`. Shards live on S3/GCS/Azure; client downloads in background and caches locally. Three properties matter:

1. **Elastic determinism**: same global sample order regardless of GPU/node count. Resume on a different cluster size and you get the same data sequence. Critical for reproducibility.
2. **Compression**: zstd or brotli per shard. CPU-time tradeoff; ~5× smaller objects than uncompressed.
3. **Predownload**: configurable in-flight shard count. Prevents loader stalls when a shard takes longer than expected.

```python
from streaming import StreamingDataset, StreamingDataLoader

ds = StreamingDataset(
    remote="s3://my-bucket/tokens",
    local="/tmp/cache",
    shuffle=True,
    batch_size=8,
    predownload=64 * 8,  # 64 batches in flight
)
loader = StreamingDataLoader(ds, batch_size=8, num_workers=8)
```

Dominant choice for LLM pretraining outside NVIDIA stack in 2026.

### NVIDIA Megatron-Energon

[github.com/NVIDIA/Megatron-Energon](https://github.com/NVIDIA/Megatron-Energon). The multimodal-first dataloader inside NeMo / Megatron-Core / Megatron-Bridge. WebDataset-style shards but with multimodal handling (images + text + audio). Used for VLM pretraining.

### WebDataset

[github.com/webdataset/webdataset](https://github.com/webdataset/webdataset). Tar-shard format; iterable; predates MDS. Still alive but losing ground for pure-text LLM pretraining. Common in vision/multimodal codebases.

### HuggingFace `datasets` IterableDataset

Fine for fine-tuning and small-scale pretraining. Not engineered for the throughput a frontier-scale loader needs. The `interleave_datasets` + `to_iterable_dataset` pattern works for runs up to a few hundred GPUs; beyond that, MDS or Energon.

## Tokenization at scale

Online tokenization (tokenize at `__getitem__` time) is the easy default. It bottlenecks above ~70B-scale runs because the tokenizer becomes the per-worker bottleneck.

The 2026 default is **pre-tokenized shards**. Tokenize the corpus once, offline, into MDS or WebDataset shards storing `int32` token IDs. The dataloader then just decodes and slices, no tokenizer call on the hot path.

Tools:
- **HuggingFace `tokenizers`** (Rust): the standard for offline tokenization. Multi-GB/s on a beefy node.
- **RAPIDS-tokenizer / cuDF**: GPU-accelerated tokenization for very large corpora. Used by NVIDIA Curator.
- **NeMo Curator**: full-pipeline data-prep including tokenization.

For fine-tuning runs, online tokenization is fine.

## Sequence packing

Variable-length sequences waste compute when padded to a fixed length. Sequence packing concatenates short sequences into one packed `seq_len` slot, with cross-sequence attention masks preventing leak.

```
without packing                    with packing
batch[0]: <s1><pad><pad><pad>      batch[0]: <s1><eos><s2><eos><s3>
batch[1]: <s2><eos><pad><pad>      attention mask is block-diagonal:
batch[2]: <s3><eos><eos><pad>          ┌─s1─┐ 0    0
                                       │   │
                                       0   ┌s2┐ 0
                                       0    0 ┌s3──┐
```

Standard in 2026 across torchtitan, Megatron, Axolotl, etc. Implementation knobs:
- **THD layout**: tokens-then-headdim contiguous; FlashAttention 2/3 has efficient kernels for packed THD.
- **Document boundaries via `cu_seqlens`**: one int32 per sequence boundary. FlashAttention's `varlen` interface consumes this directly.

## G17 of Project 3

The graph: tokenization throughput (tokens/sec from dataloader) vs training step throughput (tokens/sec consumed by model). Plot both on the same axes vs batch size or sequence length.

```
tokens/sec
   ▲
   │     ┌─── model consumption (compute-bound)
   │    ╱
   │   ╱  ┌─── loader supply (with packing)
   │  ╱  ╱
   │ ╱  ╱   ┌─── loader supply (without packing)
   │╱──╱───╱
   └────────────────►  batch size
        ↑       ↑
        │       │
   loader the wall (no packing)
            loader stops being the wall
```

Where the loader curve crosses the model curve is your wall. Below that batch size, the loader can keep up. Above, the model is starved.

## Build steps

1. Pick a small dataset (TinyStories, 1B tokens). Tokenize offline using HF `tokenizers` into MDS shards.
2. Train a 100M-param model with sequence packing on, then off. Measure tokens/sec, GPU utilization.
3. Plot G17.

## Reference

- Mosaic StreamingDataset: [github.com/mosaicml/streaming](https://github.com/mosaicml/streaming)
- Streaming docs: [docs.mosaicml.com/projects/streaming/](https://docs.mosaicml.com/projects/streaming/)
- Megatron-Energon: [github.com/NVIDIA/Megatron-Energon](https://github.com/NVIDIA/Megatron-Energon)
- HF tokenizers (Rust): [github.com/huggingface/tokenizers](https://github.com/huggingface/tokenizers)
- FlashAttention varlen: [github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- NVIDIA NeMo Curator: [github.com/NVIDIA/NeMo-Curator](https://github.com/NVIDIA/NeMo-Curator)
- Sequence packing in transformers: [huggingface.co/docs/trl/main/en/sft_trainer#packing-dataset-constantlengthdataset](https://huggingface.co/docs/trl/main/en/sft_trainer)
