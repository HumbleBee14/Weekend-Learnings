# Capstone — notes

## Hardware
GPU model, PyTorch version, dtype, hidden dim used.

## Audit
Did `fullgraph=True` pass on the first run? If not, what broke and how did you fix it?

## Benchmark (ms per iter, mean of 50)
| Variant | Prefill (B=1, S=128) | Decode (B=1, S=1) |
|---|---|---|
| Eager | | |
| compile(default) | | |
| compile(fullgraph=True) | | |
| compile(reduce-overhead) | | |
| **Piecewise (yours)** | | |

## Compile times
| Variant | Cold start (s) | First-call latency |
|---|---|---|
| compile(default) | | |
| compile(reduce-overhead) | | |
| Piecewise (yours) | | |

## Where the piecewise win came from
Three sentences. Be specific about which kernels were launched fewer times, or which capture was avoided.

## Where you fell short of vLLM
Reference vLLM's CUDA graphs design doc. What does their dispatcher do that yours doesn't? (Examples: batch>1, decode/prefill coexistence, FP8 KV cache, multi-step scheduling.)

## What you'd change to handle batch > 1
The current wrapper keys graphs on (B, S). For variable batch you'd need either bucketing or a uniform-batch fast-path. Sketch your design.

## One thing you would actually ship from this code
Be honest. Is the wrapper good enough to lift into a real project, or is there a piece missing?
