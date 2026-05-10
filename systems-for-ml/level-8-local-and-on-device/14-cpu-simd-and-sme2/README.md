# 14 — CPU SIMD and SME2

## Files

- `CONCEPTS.md` — AMX, SME/SME2, AVX-512, NEON; Accelerate; llama.cpp CPU backend; when CPU beats GPU.
- `bench_cpu_vs_gpu.sh` — drives `llama-bench` across `-ngl 0` and `-ngl 99`, sweeps thread counts, prints a table.
- `sme_probe.py` — detects SME/SME2 on the host via `sysctl` and reports llama.cpp's chosen CPU features.

## Quickstart

Install llama.cpp (Homebrew ships an Apple-Silicon-aware build):

```bash
brew install llama.cpp
# pull a small GGUF for the bench
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir ./gguf
```

Probe the host:

```bash
python sme_probe.py
```

Run the comparison:

```bash
MODEL=./gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf bash bench_cpu_vs_gpu.sh
```

## Expected output

`sme_probe.py` prints something like:

```
chip                : Apple M4 Max
cpu.optional.arm.FEAT_SME  : 1
cpu.optional.arm.FEAT_SME2 : 1
cpu.optional.arm.FEAT_SVE  : 0   # Apple does not ship SVE outside SME streaming mode
P-cores             : 12
E-cores             : 4
recommended -t      : 12
```

`bench_cpu_vs_gpu.sh` prints a table:

```
model           backend    threads    pp512        tg128
3B Q4_K_M       Metal      -          ~3200 t/s    ~140 t/s
3B Q4_K_M       CPU        12         ~280 t/s     ~32 t/s
3B Q4_K_M       CPU        16         ~250 t/s     ~28 t/s    <- E-cores hurt
3B Q4_K_M       CPU        8          ~210 t/s     ~30 t/s
```

`pp512` is prompt-processing tok/s (compute-bound). `tg128` is decode tok/s (bandwidth-bound). GPU wins both. The CPU regime to remember: low-power steady state where 30 tok/s with the fans off is the right product choice.

## Try

- Rebuild llama.cpp from source with `-DGGML_METAL=OFF -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple` and rerun. Prompt processing should jump on the CPU side because Accelerate dispatches to SME.
- Run with `taskpolicy -c utility` (macOS) to pin to E-cores and watch tok/s collapse — confirms why the recommended thread count equals P-core count.
- Repeat on a 7B Q4. The CPU/GPU gap widens. The CPU is for small models.
- If you have access to an Intel Xeon SPR+ box, run the same model with AMX bf16 enabled and compare. AMX is the x86 equivalent story.

## Where this goes

Topic 15 climbs back up from one box to many: distributing one model across multiple Macs over Thunderbolt 5. Topic 16 closes the level with the privacy threat model — the answer to "why was I doing all this on-device in the first place?"
