// ReLU using vectorized loads (float4).
//
// Each thread now processes 4 floats per call. Fewer instructions, fewer memory
// requests, more bytes per cycle.
//
// Compile:
//     nvcc -O3 -arch=sm_80 relu_vec4.cu -o relu_vec4
// Run:
//     ./relu_vec4

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


// Plain ReLU — one float per thread. Baseline for comparison.
__global__ void relu_plain(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = v > 0.0f ? v : 0.0f;
    }
}


// Vectorized ReLU — one float4 (16 bytes) per thread.
// Caller must ensure n is a multiple of 4 and arrays are 16-byte aligned.
// cudaMalloc returns 256-byte aligned pointers, so this is fine.
__global__ void relu_vec4(const float4* x, float4* y, int n_vec4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vec4) {
        float4 v = x[i];                    // single 16-byte load
        v.x = v.x > 0.0f ? v.x : 0.0f;
        v.y = v.y > 0.0f ? v.y : 0.0f;
        v.z = v.z > 0.0f ? v.z : 0.0f;
        v.w = v.w > 0.0f ? v.w : 0.0f;
        y[i] = v;                           // single 16-byte store
    }
}


double benchmark(void (*launch)(const float*, float*, int), const char* name,
                 const float* d_x, float* d_y, int n) {
    // Warmup
    launch(d_x, d_y, n);
    cudaDeviceSynchronize();

    // Time over multiple runs
    const int RUNS = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < RUNS; i++) launch(d_x, d_y, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms;
    cudaEventElapsedTime(&total_ms, start, stop);
    double per_ms = total_ms / RUNS;

    // Bandwidth: read x + write y = 2 * n * 4 bytes
    double gb = 2.0 * n * sizeof(float) / 1e9;
    double bw = gb / (per_ms / 1000.0);

    printf("%-12s  %.3f ms  %6.1f GB/s\n", name, per_ms, bw);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return bw;
}


// Wrappers so we can pass to benchmark()
void launch_plain(const float* x, float* y, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    relu_plain<<<grid, block>>>(x, y, n);
}

void launch_vec4(const float* x, float* y, int n) {
    int n_vec4 = n / 4;
    int block = 256;
    int grid = (n_vec4 + block - 1) / block;
    relu_vec4<<<grid, block>>>(reinterpret_cast<const float4*>(x),
                                reinterpret_cast<float4*>(y), n_vec4);
}


int main() {
    const int N = 1 << 24;  // 16M elements
    const size_t bytes = N * sizeof(float);

    float* h_x = (float*)malloc(bytes);
    float* h_y = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_x[i] = (i % 7) - 3.0f;  // mix of pos/neg

    float *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    printf("N = %d (%.1f MB total)\n\n", N, 2.0 * bytes / 1e6);
    printf("%-12s  %-9s  %s\n", "version", "time", "bandwidth");
    printf("%-12s  %-9s  %s\n", "-------", "----", "---------");
    benchmark(launch_plain, "plain", d_x, d_y, N);
    benchmark(launch_vec4, "vec4", d_x, d_y, N);

    // Verify correctness of vec4 against the plain version
    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < N && ok; i++) {
        float expected = h_x[i] > 0 ? h_x[i] : 0;
        if (h_y[i] != expected) { printf("Mismatch at %d\n", i); ok = false; }
    }
    printf("\nCorrectness: %s\n", ok ? "OK" : "FAILED");

    cudaFree(d_x);
    cudaFree(d_y);
    free(h_x);
    free(h_y);
    return ok ? 0 : 1;
}
