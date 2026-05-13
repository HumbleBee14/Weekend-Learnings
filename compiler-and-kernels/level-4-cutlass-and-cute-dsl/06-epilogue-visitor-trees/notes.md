# Notes — 06

Capture:

- Your unfused / `torch.compile` / EVT-fused numbers on the LLaMA FFN-1 shape.
- The HBM bandwidth gap between unfused and fused. The accumulator round-trip is what fusion saves; how big is it as a fraction of total bandwidth?
- For the NVFP4 epilogue: explain in two sentences why the block-max reduction has to happen in the epilogue and can't be done as a separate pass. (Hint: the FP32 accumulator only exists in registers/TMEM; once it's stored, you've lost the precision you need for scale derivation.)
- A custom epilogue node you'd write for a different op. E.g. "fused linear + RMSNorm for the residual stream — RMSNorm needs a per-row variance which is a row reduction the EVT can express."
