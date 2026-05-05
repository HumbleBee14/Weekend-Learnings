# Level 7 — StableHLO and XLA

> Outer reference: [`compiler-and-kernels/README.md`](../README.md) · Project: Flax model → StableHLO export → IREE GPU + CPU benchmark

## Week goal

StableHLO is the portability layer that connects ML frameworks (JAX, PyTorch/XLA, TensorFlow) to hardware compilers (XLA GPU, IREE, TT-Forge, NeuronSDK). Understanding it means you can move models across hardware targets. By Friday you should be able to:

- Export a JAX model to StableHLO and read the op graph
- Understand XLA's compilation pipeline: StableHLO → HLO → LLO → backend codegen
- Deploy via IREE as an alternative to XLA (useful for non-Google hardware)
- Understand why TT-Forge and NeuronSDK chose StableHLO as their ingestion format

## Where this fits

- **Comes after:** Level 6 (MLIR — StableHLO is an MLIR dialect; after writing passes in Level 6 you can read StableHLO ops as just more MLIR).
- **Comes before:** Level 8 (AI-assisted kernels) and Level 9 (Rust). This is the last compiler-focused level.

## 2026 reality check

- **StableHLO has a 5-year backward + 2-year forward compatibility guarantee.** This makes it viable for long-lived production deployments — unlike MHLO (which changes with XLA internals). The commitment matters for custom hardware teams.
- **PJRT (Pluggable XLA Runtime)** is the hardware plugin interface. Any framework that supports PJRT (JAX, PyTorch/XLA) can run on any hardware that implements PJRT — including Tenstorrent, AWS Trainium, and future accelerators. StableHLO is what PJRT transfers between the framework and the compiler.
- **OpenXLA** is the GitHub organization (openxla.org) that maintains StableHLO, XLA GPU backend, and IREE collectively. It's the neutral home for cross-company ML compiler work (Google, NVIDIA, AMD, Apple, Meta contributors).
- **For most practitioners outside Google:** StableHLO matters as an *exchange format*, not a compiler to extend. The use cases are: (1) deploying JAX models to non-TPU hardware, (2) understanding what PyTorch/XLA does under the hood, (3) reading TT-Forge's ingestion path.

## Topic-by-topic deep dive

| # | Topic | What you build |
|---|-------|---------------|
| 01 | stablehlo-op-set | What ops exist; how they differ from MLIR linalg |
| 02 | export-from-jax | jax.export → StableHLO; read the graph |
| 03 | xla-compilation-pipeline | StableHLO → HLO → LLO → backend |
| 04 | pjrt-interface | How hardware plugins talk to JAX/PyTorch |
| 05 | iree-as-xla-alternative | Deploy StableHLO via IREE instead of XLA |
| 06 | stablehlo-in-tt-forge | How Tenstorrent ingests StableHLO |
| 07 | pytorch-xla-path | torch.export → StableHLO on TPU |

### 01 — `stablehlo-op-set`

**What StableHLO is.** StableHLO is an MLIR dialect (namespace: `stablehlo`) that defines a portable set of ML operations. It's derived from MHLO (Meta High-Level Operations), which was XLA's internal dialect, but with a stability guarantee. The op set covers: elementwise math (`stablehlo.add`, `stablehlo.multiply`, `stablehlo.exp`), linear algebra (`stablehlo.dot_general`, `stablehlo.convolution`), data movement (`stablehlo.reshape`, `stablehlo.transpose`, `stablehlo.gather`), control flow (`stablehlo.while`, `stablehlo.if`), and collective communication (`stablehlo.all_reduce`, `stablehlo.all_gather`).

**How it differs from linalg.** `linalg.matmul` is generic — it describes the computation structure (two nested parallel loops + one reduction loop) and lets the compiler decide how to execute it. `stablehlo.dot_general` is semantic — it describes *what* the contraction is (which dimensions are batch, which are contracting) but also carries XLA-specific semantics around type promotion, rounding, and batching. Linalg is more compiler-friendly; StableHLO is more semantically precise.

**Read the spec.** The [StableHLO spec](https://openxla.org/stablehlo/spec) documents each op's semantics. Spend one session reading the `dot_general`, `reduce`, and `scatter` specs — these are the three ops that map to the most interesting computations (matmul, softmax/sum, attention scatter).

### 02 — `export-from-jax`

**`jax.export`.** The clean API for producing StableHLO from JAX:
```python
import jax
import jax.numpy as jnp
from flax import linen as nn

class MLP(nn.Module):
    features: int
    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.features)(nn.relu(nn.Dense(self.features)(x)))

model = MLP(features=256)
params = model.init(jax.random.PRNGKey(0), jnp.ones((4, 128)))
apply_fn = lambda params, x: model.apply(params, x)

# Export to StableHLO
exported = jax.export.export(
    jax.jit(apply_fn)
)(params, jnp.ones((4, 128)))

print(exported.mlir_module())  # StableHLO MLIR text
```

**Reading the output.** The exported MLIR will have `stablehlo.*` ops. Identify: the `dot_general` ops (linear layers), the `reduce` op (for any softmax/layernorm), the `broadcast_in_dim` ops (bias broadcasting). Each one corresponds to a specific PyTorch/JAX operation.

**`stablehlo-opt` tool.** Like `mlir-opt` but StableHLO-aware. Lets you run StableHLO-specific passes: canonicalization, shape inference, constant folding. `pip install stablehlo` for the Python package.

### 03 — `xla-compilation-pipeline`

**The XLA lowering chain:**
```
StableHLO
    ↓ (stablehlo→hlo converter)
HLO (High Level Operations — XLA's internal IR)
    ↓ (XLA optimization passes: fusion, layout, buffer assignment)
LLO (Low Level Operations — scheduled + memory-assigned)
    ↓ (backend codegen)
    ├── GPU: PTX / cuBIN via NVPTX backend
    ├── TPU: TPU instructions via TPU backend
    └── CPU: LLVM IR via LLVM backend
```

**The XLA fusion heuristic.** XLA's fusion pass is different from Inductor's — it uses an HLO-level cost model based on instruction count and memory access patterns. It fuses elementwise ops aggressively but is more conservative around reductions. Understanding XLA's fusion decisions helps when you see unexpected performance on TPU.

**XLA's layout assignment.** XLA chooses between row-major (default) and column-major tensor layouts per op based on what the hardware and surrounding ops prefer. Layout conflicts (where one op wants row-major output but the next wants column-major input) cause `transpose` insertions. These show up as performance cliffs.

**What to read.** [OpenXLA XLA GPU architecture](https://openxla.org/xla/gpu_architecture) — the canonical overview of the XLA GPU compilation pipeline.

### 04 — `pjrt-interface`

**What PJRT is.** Pluggable XLA Runtime. A C++ API that hardware vendors implement to connect their hardware to JAX (and PyTorch/XLA). A PJRT device plugin provides: a list of devices, a way to compile StableHLO to the device, a way to run compiled programs, and a way to transfer buffers.

**Why it matters.** When JAX says "this runs on Tenstorrent Wormhole," that means Tenstorrent implements the PJRT interface. JAX calls `pjrt_plugin.Compile(stablehlo_module)` → Tenstorrent's TT-Forge compiles it → JAX calls `pjrt_plugin.Execute(compiled, inputs)` → Tenstorrent's runtime runs it. The framework never knows what hardware it's talking to.

**Connection to your work.** The PJRT interface is what makes the TT-Forge StableHLO ingestion path (Level 6, Topic 6) production-grade rather than a research prototype. JAX doesn't have special Tenstorrent code — it just uses the PJRT plugin.

**Resources.**
- [PJRT overview — Google OSS Blog](https://opensource.googleblog.com/2023/05/pjrt-simplifying-ml-hardware-and-framework-integration.html)
- [JAX PJRT plugin documentation](https://jax.readthedocs.io/en/latest/Custom_Operation_for_GPUs.html#pjrt)

### 05 — `iree-as-xla-alternative`

**When you'd use IREE instead of XLA.** XLA is optimized for Google's hardware (TPU) and NVIDIA GPUs. For AMD GPUs, Apple Metal, or Vulkan-based hardware, IREE's backends are often better. IREE's CPU backend (via LLVM + MLIR vectorization) is competitive with XLA's CPU backend for inference.

**IREE accepts StableHLO directly:**
```python
import iree.compiler as ireec

# Compile StableHLO for Metal (Apple GPU)
mlir_text = exported.mlir_module()  # from jax.export above
flatbuffer = ireec.compile_str(
    mlir_text,
    target_backends=["metal"],
    input_type="stablehlo"
)
```

**Build steps.**
1. Export your Flax/JAX MLP to StableHLO via `jax.export`.
2. Compile with IREE for CPU and Metal (both available on your Mac).
3. Measure throughput: IREE CPU vs JAX CPU (XLA CPU backend) vs JAX GPU (MPS).
4. Inspect the generated Metal shader (`--mlir-print-ir-after-all` during compile).

### 06 — `stablehlo-in-tt-forge`

**The TT-Forge ingestion path.** TT-Forge accepts PyTorch models via `torch.compile(backend="ttnn")` or via direct `torch.export` → StableHLO conversion. The PJRT path: JAX → PJRT plugin → `tt-mlir` compiler receives StableHLO → lowers through TTIR → TTNN → Metalium.

**The StableHLO → TTIR conversion.** In `tt-forge-fe` (the frontend), each StableHLO op is converted to a TTIR op via a `ConversionPattern`. For example, `stablehlo.dot_general` → `ttir.matmul`. Stablehlo's batching dimensions map to TTIR's explicit broadcast semantics. The conversion handles shape inference and type normalization.

**What to explore.** Clone `tt-forge`. Look at `lib/Conversion/StableHLOToTTIR/`. Find the `DotGeneralOpConversionPattern`. Trace how `stablehlo.dot_general` with `batching_dimension_numbers` maps to `ttir.matmul` with explicit batch dimensions. This is a concrete example of cross-dialect conversion.

### 07 — `pytorch-xla-path`

**PyTorch/XLA.** `torch_xla` is the PyTorch XLA integration. `torch.compile(backend="openxla")` lowers PyTorch to StableHLO via Dynamo + an XLA-specific backend. This is how PyTorch models run on TPUs.

```python
import torch_xla.core.xla_model as xm
device = xm.xla_device()  # TPU or CPU via XLA

# StableHLO export
from torch_xla.stablehlo import exported_program_to_stablehlo
ep = torch.export.export(model, (example_input,))
shlo = exported_program_to_stablehlo(ep)
print(shlo.get_stablehlo_text("forward"))
```

**Where this runs.** Google Colab has free TPU v2/v3 access. You can run PyTorch on TPU via `torch_xla`. The round-trip: `torch.compile(backend="openxla")` → Dynamo traces FX graph → FX → StableHLO → XLA compiles for TPU → TPU executes.

**Build steps.** Run the [PyTorch/XLA MNIST tutorial](https://pytorch.org/xla/master/) on Colab TPU. Export the model to StableHLO. Read the ops — compare to the IREE deployment from Topic 05. Same model, same StableHLO, two different backends.

## Project this week

```
compiler-and-kernels/
└── stablehlo/
    ├── flax_export.py             # JAX/Flax → StableHLO export
    ├── iree_deploy.py             # IREE CPU + Metal deployment
    ├── pytorch_xla_tpu.ipynb      # Colab: PyTorch → StableHLO on TPU
    └── reports/
        └── level7-stablehlo.md   # benchmark table + portability diagram
```

**Benchmark table:**

| Backend | Runtime | Throughput (tok/s) | Notes |
|---|---|---|---|
| XLA CPU | JAX/CPU | | |
| IREE CPU | IREE/LLVM | | |
| IREE Metal | IREE/MSL | your M5 Mac! | |
| XLA TPU | Google Colab | | |

## Definition of done

- [ ] You exported a JAX model to StableHLO and can read the op graph.
- [ ] You deployed via IREE to CPU and Metal with benchmark numbers.
- [ ] You can explain the PJRT interface and why it matters for custom hardware.
- [ ] You traced the `stablehlo.dot_general` → `ttir.matmul` conversion in TT-Forge.

## Resources

- **StableHLO spec** — [openxla.org/stablehlo/spec](https://openxla.org/stablehlo/spec).
- **StableHLO GitHub** — [github.com/openxla/stablehlo](https://github.com/openxla/stablehlo).
- **XLA GPU architecture** — [openxla.org/xla/gpu_architecture](https://openxla.org/xla/gpu_architecture).
- **PyTorch/XLA StableHLO export** — [docs.pytorch.org/xla/master/features/stablehlo.html](https://docs.pytorch.org/xla/master/features/stablehlo.html).
- **IREE StableHLO input** — [iree.dev/guides/ml-frameworks/jax](https://iree.dev/guides/ml-frameworks/jax/).
- **PJRT blog** — [opensource.googleblog.com/2023/05/pjrt-simplifying-ml-hardware](https://opensource.googleblog.com/2023/05/pjrt-simplifying-ml-hardware-and-framework-integration.html).

## What you'll be able to do after this week

> Export JAX models to StableHLO, read and modify the op graph, deploy through IREE to CPU and Metal, understand the XLA compilation pipeline, and explain PJRT as the hardware plugin interface. Understand why custom silicon companies chose StableHLO as their model ingestion format.
