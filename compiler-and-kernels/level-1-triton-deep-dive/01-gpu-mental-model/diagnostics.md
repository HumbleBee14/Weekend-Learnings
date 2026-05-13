# Diagnostic questions

Closed-book. Answer in [`notes.md`](notes.md) in your own words. A few sentences each, sometimes with small arithmetic. If you can answer 6+ confidently, you're ready for sub-module 02.

1. What is the smallest unit of execution on an NVIDIA GPU, and why does it matter for the code you write?

2. An H100 has 132 SMs. If I launch a Triton kernel with `grid = (8,)`, what's the problem? How would you fix it?

3. RMSNorm reads N floats from a row, computes one reduction over them, writes N floats back. Roughly what's its arithmetic intensity in FLOPs per byte? Is it memory-bound or compute-bound on an H100, and why?

4. What does warp divergence cost in clock cycles, qualitatively? Give an example of code that would cause it and one way to avoid it.

5. What is TMA (Tensor Memory Accelerator) and what kernel pattern does it enable? Why does that pattern beat the pre-Hopper way?

6. What does `tl.dot` lower to on H100? Why does it matter that a matmul-shaped op uses it instead of a hand-written loop of multiplies?

7. A memory-bound kernel runs at 15% of peak HBM bandwidth on your hardware. List three plausible reasons. Which one would you check first and how?

8. Compare a CPU's L1 cache (~32 KB, ~5 cycles) to an H100's shared memory (~228 KB per SM, ~30 cycles). Where does the analogy hold and where does it break?

When you've answered all eight, open `CONCEPTS.md` and expand the answer key at the bottom. Compare. Anywhere you missed substantially, re-read the relevant section.
