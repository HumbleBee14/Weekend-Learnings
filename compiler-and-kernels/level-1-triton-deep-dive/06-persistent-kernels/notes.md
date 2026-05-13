# Observations

Record your numbers and what you noticed. One paragraph per file is enough.

## Hardware

- GPU:
- Driver / CUDA / PyTorch / Triton versions:
- SM count:

## File 01 — Static persistent matmul

Decode (M=1):
- non-persistent: ___ us
- persistent:     ___ us
- ratio:          ___ x

Small batch (M=8):
- non-persistent: ___ us
- persistent:     ___ us
- ratio:          ___ x

Square (M=N=K=2048):
- non-persistent: ___ us
- persistent:     ___ us
- ratio:          ___ x

What did you notice? Where does persistence help, where doesn't it?


## File 02 — Dynamic persistent

- coefficient of variation of tile cost: ___
- static persistent : ___ us
- dynamic persistent: ___ us
- ratio:             ___ x

If you bump `skew` up or down in `build_ragged_workload`, where does the crossover happen?


## File 03 — CUDA graph capture

At M=1 N=4096 K=4096:
- eager persistent : ___ us
- graph replay     : ___ us
- speedup:          ___ x

Estimate your driver's launch overhead: (eager - graph) ≈ ___ us per launch. Compare to the ~7us figure from the PyTorch CUDA graphs blog.

Did the "wrong pattern" demo (fresh allocation + replay) show a large diff for you? If not, why might that be on your hardware/driver?


## File 04 — Piecewise graph

At M=1, two-matmul-plus-argmax:
- all eager  : ___ us
- piecewise  : ___ us
- speedup:    ___ x

Bump M to 32. Does piecewise still win? By how much?


## Three things

1. Something you didn't expect:
2. Something that confirmed what `CONCEPTS.md` said:
3. Something you'd change about your benchmark methodology:
