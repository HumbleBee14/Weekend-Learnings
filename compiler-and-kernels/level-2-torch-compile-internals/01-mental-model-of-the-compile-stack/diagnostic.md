# Diagnostic — six questions

Answer in your own words in [`notes.md`](notes.md). No looking back at CONCEPTS.md while you write the first pass. Then check yourself.

1. A user reports their `torch.compile`'d model recompiles every batch. Which of the four layers is responsible for deciding to recompile, and what is the specific mechanism — what data structure does the runtime consult on each call?

2. Inductor decides to fuse RMSNorm with the immediately following GeLU into one Triton kernel. Why is that fusion legal — what property of the two ops makes it safe? What would make it illegal (give one example)?

3. A model has `if x.sum() > 0:` in its forward. Walk through what Dynamo does, instruction by instruction, when it hits that line. Where does it cut the graph? What does the generated code look like?

4. Your friend says "I set `dynamic=True` and now my model is faster because there are no recompiles." This is sometimes true and sometimes false. Give one scenario for each.

5. You compile a model in training mode and the backward produces wrong gradients. Which layer's output do you read to debug? What kind of bug would live there that wouldn't live in your forward code?

6. vLLM wraps attention as a custom op. Attention is GPU-safe and would fit in a CUDA graph fine — why deliberately exclude it? Answer in terms of recompile cost and shape variability.

## Self-check

If you wrote >3 sentences for each and named the right layer in each, you are ready for sub-module 02. If you wrote one-line answers or named the wrong layer, re-read [`CONCEPTS.md`](CONCEPTS.md) and try again. There is no point continuing without this.
