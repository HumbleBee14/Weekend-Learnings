# Per-vendor cheat sheet (2026)

One paragraph each. Use as a desk reference when reading vendor blogs or hiring posts.

## NVIDIA

Blackwell (B200/B300) is the 2026 datacentre part. FP4 (NVFP4) and FP8 are first-class. Compiler entry points: CUDA (driver), Triton (Inductor's GPU codegen target), CUTLASS 4.x with CuTe DSL (custom kernels). All ultimately emit PTX → SASS via `ptxas`. The Triton compiler itself is MLIR-based (Triton dialect, then `triton-gpu` dialect, then LLVM). Docs: https://docs.nvidia.com/cuda/, https://github.com/NVIDIA/cutlass.

## AMD

MI300X (CDNA3) shipped 2024; MI350X (CDNA4, FP4-capable) in 2025. ROCm 7 + Triton's AMD backend is the path PyTorch users take. Composable Kernel (CK) is AMD's CUTLASS analog. The MI300X's 192 GB HBM3e per package is the architectural draw — many models that need 4xH100 fit on 1xMI300X. Docs: https://rocm.docs.amd.com/, https://github.com/ROCm/composable_kernel.

## Google TPU

v5p, v6 (Trillium), v7 announced 2025. JAX → StableHLO → HLO → TPU ISA, all driven by XLA. Pallas is the kernel DSL when XLA's auto-codegen isn't enough — used inside JAX for FlashAttention-on-TPU. PyTorch users reach TPUs via PyTorch/XLA, which lowers FX to StableHLO. Docs: https://docs.jax.dev/en/latest/pallas/, https://openxla.org/xla.

## AWS Trainium2

Shipping at scale on AWS in 2025–2026. Neuron SDK is XLA-derived: PyTorch / JAX programs lower through StableHLO into Neuron's compiler. NKI (Neuron Kernel Interface) is the Pallas-equivalent kernel DSL — Python-fronted, emits Neuron ISA. Docs: https://awsdocs-neuron.readthedocs-hosted.com/.

## Groq LPU

Inference-only. Static dataflow: the compiler emits a complete cycle-accurate schedule for every functional unit. Result: deterministic latency, very fast token generation, but every supported model has been compiled by Groq's team. Compiler is closed source. Architecture overview: https://groq.com/.

## Cerebras

CS-3 and CS-4 are wafer-scale (one chip = the whole 300mm wafer). Programming model: weight streaming — weights live in MemoryX, stream into wafer per layer; activations stay on-chip. The compiler has to do layer-pipeline placement across ~900K cores. The 2025 surprise: Cerebras inference for Llama-405B-class models is faster than any GPU rack at the same dollar cost, because there's no HBM bottleneck. Docs: https://www.cerebras.ai/.

## Tenstorrent

Tensix-core architecture, RISC-V control, explicit data movement. TT-Metal (low level, like CUDA) is open source and the centre of the developer story. TT-NN is a PyTorch-like op library on top. TT-Buda is the autoplacement compiler. The bet is on an open ecosystem that gathers community kernels the way Triton has. Docs: https://github.com/tenstorrent/tt-metal, https://docs.tenstorrent.com/.

## SambaNova SN40L

Reconfigurable dataflow ASIC for enterprise inference of large models. SambaFlow compiler is MLIR-based. Smaller community than the above; primary distribution is appliance + service. Docs: https://sambanova.ai/.

## Etched Sohu

Transformer-only ASIC. The compiler is *narrower by design* — only attention + MLP + a few activations. The bet: by giving up generality, push throughput far past general-purpose silicon. Early customer deployments through 2025; broader availability 2026.

## Modular (cross-cutting)

Mojo is a Python-superset built on MLIR. MAX is the runtime that compiles Mojo (and imported PyTorch/ONNX models) and dispatches to NVIDIA / AMD / CPU backends. The "compile once, run on many accelerators" story for closed-source-but-commercial users; IREE is the open-source equivalent (Topic 06). Docs: https://www.modular.com/.
