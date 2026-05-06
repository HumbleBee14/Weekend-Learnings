"""
Modern alternative to the standalone .cu files: compile inline from Python via PyTorch's
cpp_extension. No setup.py, no Makefile. Just embed the CUDA source as a Python string,
call load_inline(), get a Python module back. This is what most 2026 CUDA learning
content uses.

Run:
    pip install torch
    python run_in_pytorch.py

Needs a GPU. Free Colab T4 works.
"""

import time
import torch
from torch.utils.cpp_extension import load_inline

# ---- Source --------------------------------------------------------------

CPP_SRC = r"""
#include <torch/extension.h>

void launch_vector_add(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void launch_relu(torch::Tensor x, torch::Tensor y);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &launch_vector_add, "vector add (CUDA)");
    m.def("relu", &launch_relu, "ReLU (CUDA)");
}
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void relu_vec4_kernel(const float4* x, float4* y, int n_vec4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vec4) {
        float4 v = x[i];
        v.x = v.x > 0.0f ? v.x : 0.0f;
        v.y = v.y > 0.0f ? v.y : 0.0f;
        v.z = v.z > 0.0f ? v.z : 0.0f;
        v.w = v.w > 0.0f ? v.w : 0.0f;
        y[i] = v;
    }
}

void launch_vector_add(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int n = a.numel();
    int block = 256;
    int grid = (n + block - 1) / block;
    vector_add_kernel<<<grid, block>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
}

void launch_relu(torch::Tensor x, torch::Tensor y) {
    int n = x.numel();
    // Make sure n is divisible by 4 for vec4 path (caller's responsibility for now)
    int n_vec4 = n / 4;
    int block = 256;
    int grid = (n_vec4 + block - 1) / block;
    relu_vec4_kernel<<<grid, block>>>(
        reinterpret_cast<float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(y.data_ptr<float>()), n_vec4);
}
"""


def main():
    if not torch.cuda.is_available():
        raise SystemExit("Needs CUDA.")

    print("Compiling CUDA extension (first run takes 30-60 seconds)...")
    t0 = time.time()
    cu_mod = load_inline(
        name="my_cuda_kernels",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        functions=["vector_add", "relu"],
        verbose=False,
    )
    print(f"Compiled in {time.time() - t0:.1f}s\n")

    # ---- vector add ----
    n = 1 << 20
    a = torch.arange(n, dtype=torch.float32, device="cuda")
    b = torch.arange(n, dtype=torch.float32, device="cuda") * 2
    c = torch.empty_like(a)

    cu_mod.vector_add(a, b, c)
    torch.cuda.synchronize()

    expected = a + b
    print(f"vector_add  ok={torch.allclose(c, expected)}")

    # ---- relu ----
    x = torch.randn(n, device="cuda")
    y = torch.empty_like(x)
    cu_mod.relu(x, y)
    torch.cuda.synchronize()
    expected = torch.relu(x)
    print(f"relu        ok={torch.allclose(y, expected)}")


if __name__ == "__main__":
    main()
