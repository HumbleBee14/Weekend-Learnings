# Notes — 05

## Hardware
GPU model, dtype used.

## Recompile counts (from TORCH_LOGS=recompiles output)
| mode | recompiles | fell back to eager? |
|---|---|---|
| naive | | |
| mark_dynamic | | |
| dynamic=True | | |
| bucket | | |

## Latency table (ms per call, mean over 20 iters)
| seqlen | naive | mark_dynamic | dynamic | bucket |
|---|---|---|---|---|
| 1 | | | | |
| 16 | | | | |
| 64 | | | | |
| 128 | | | | |
| 256 | | | | |

## One concrete SymInt expression you saw in the FX graph (from TORCH_LOGS=aot_graphs)
[paste the line, explain what s0 represents]

## Which mode you would deploy and why
