// Vector add — the hello world of CUDA.
//
// Compile (T4 = sm_75; A100 = sm_80; H100 = sm_90; B200 = sm_100):
//     nvcc -O3 -arch=sm_80 vector_add.cu -o vector_add
// Run:
//     ./vector_add
//
// Or compile-and-run on Colab:
//     %%writefile vector_add.cu  ← in a cell with this content
//     !nvcc -O3 -arch=sm_75 vector_add.cu -o vector_add && ./vector_add

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do {                                            \
    cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(err));            \
        exit(1);                                                         \
    }                                                                    \
} while (0)

// __global__ = "kernel" — entry point callable from host, runs on device.
// Each thread runs one instance of this function.
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // Compute this thread's global index.
    // blockIdx: which block this thread is in (within the grid)
    // blockDim: how many threads per block
    // threadIdx: this thread's index within its block
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check: the last block typically has some threads with i >= n.
    // Without this check, those threads read past the end of the array.
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1 << 20;            // 1 million elements
    const size_t bytes = N * sizeof(float);

    // ---- 1. Allocate host (CPU) memory ----
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);

    // Initialize inputs with simple known values so we can verify
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // ---- 2. Allocate device (GPU) memory ----
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // ---- 3. Copy host → device ----
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // ---- 4. Launch the kernel ----
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;  // ceil(N/256)

    // Time it with CUDA events (the right way; clock() is wrong for async kernels)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);

    // Check for launch errors *and* synchronize so timing is correct
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, start, stop);

    // ---- 5. Copy device → host ----
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // ---- 6. Verify ----
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];  // = 3*i
        if (h_c[i] != expected) {
            printf("MISMATCH at i=%d: %f != %f\n", i, h_c[i], expected);
            ok = false;
            break;
        }
    }

    // ---- 7. Bandwidth math ----
    // Bytes moved = 2 reads (a, b) + 1 write (c) = 3 * N * sizeof(float)
    double gb_moved = 3.0 * N * sizeof(float) / 1e9;
    double bandwidth_gbps = gb_moved / (kernel_ms / 1000.0);

    printf("N=%d  block=%d  grid=%d\n", N, block_size, grid_size);
    printf("Result: %s\n", ok ? "OK" : "FAILED");
    printf("Kernel time: %.3f ms\n", kernel_ms);
    printf("Achieved bandwidth: %.1f GB/s\n", bandwidth_gbps);

    // ---- 8. Cleanup ----
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ok ? 0 : 1;
}
