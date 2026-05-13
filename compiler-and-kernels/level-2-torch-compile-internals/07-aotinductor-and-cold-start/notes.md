# Notes — 07

## Hardware
GPU, PyTorch version.

## Cold-start times
| path | cold start (s) |
|---|---|
| eager | |
| torch.compile JIT | |
| AOTInductor load + first | |

## Archive size
`packaged_model.pt2`: ___ MB
Comparison to weights size (model weights as fp16 / bf16): ___ MB
Implication: are weights bundled?

## What would change for a real 7B model
Estimated cold-start scaling, archive size, what you'd need to externalize.
