# Setup

## On Google Colab (free, T4 GPU)

1. Open a new Colab notebook.
2. Runtime → Change runtime type → T4 GPU.
3. In the first cell:

```python
!pip install -q --upgrade torch triton
!nvidia-smi
```

The `nvidia-smi` output should show a T4 (or whatever GPU Colab gave you) with ~15 GB of memory.

4. Smoke test:

```python
import torch, triton, triton.language as tl
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
print("triton", triton.__version__)
print("device", torch.cuda.get_device_name(0))
```

You want `triton` version 3.4 or newer. If you have 3.7 or 3.6 you're current. If you have 3.2 or older, upgrade — several APIs changed.

5. To run the `.py` files from this folder, either paste them into cells or upload the folder and run them with `!python 01_vector_add.py`.

## On a cloud GPU (RunPod, Lambda, Vast.ai)

Pick a pre-built PyTorch 2.6+ image. They typically ship Triton matching the PyTorch nightly. SSH in and check the smoke test above.

## On an M-series Mac

You can't run Triton CUDA kernels on Mac. You can run CPU code that demonstrates the algorithms (e.g., the online-softmax derivation script) and read everything else. For runnable GPU exercises, use Colab. Level 6 (MLIR + IREE) is where your Mac becomes a first-class compute device via Metal.

## Pinning Triton if you need to reproduce numbers

```python
!pip install -q triton==3.7.0 torch==2.8.0
```

If a later Triton version breaks something in these scripts, the pin above is what we tested against.

## What to do if the smoke test fails

- "CUDA not available" on Colab → you forgot to switch the runtime to GPU.
- `triton.__version__` is 2.x → you have an old install. `pip install --upgrade triton`.
- `tl.make_tensor_descriptor` not found (later sub-modules) → Triton 3.4 or newer required.
