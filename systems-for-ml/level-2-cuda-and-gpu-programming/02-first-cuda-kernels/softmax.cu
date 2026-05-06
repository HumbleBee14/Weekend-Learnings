// Numerically stable softmax — one block per row.
//
// Demonstrates: shared memory reduction, warp shuffles, the max-subtraction trick.
// The same `(running_max, running_sum)` recursion shows up in FlashAttention.
//
// Compile:
//     nvcc -O3 -arch=sm_80 softmax.cu -o softmax
// Verify with compute-sanitizer (catches shared memory races):
//     compute-sanitizer --tool racecheck ./softmax

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>

#define CUDA_CHECK(call) do {                                            \
    cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));    \
        exit(1);                                                         \
    }                                                                    \
} while (0)


// Warp-level reduction using __shfl_down_sync.
// Returns the max of all 32 lanes' `val` (only lane 0's return value is useful).
__device__ __forceinline__ float warp_reduce_max(float val) {
    unsigned mask = 0xffffffff;  // all 32 lanes active
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(mask, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}


// Block-level reduction: combine warp results via shared memory.
// BLOCK_SIZE must be a multiple of 32 and ≤ 1024.
template <int BLOCK_SIZE>
__device__ float block_reduce_max(float val, float* smem) {
    int lane = threadIdx.x % 32;       // 0..31 within warp
    int warp_id = threadIdx.x / 32;    // which warp am I

    val = warp_reduce_max(val);        // intra-warp reduce
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();

    // Final reduce across warps (one warp does it)
    if (warp_id == 0) {
        val = (threadIdx.x < BLOCK_SIZE / 32) ? smem[lane] : -FLT_MAX;
        val = warp_reduce_max(val);
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();
    return smem[0];
}

template <int BLOCK_SIZE>
__device__ float block_reduce_sum(float val, float* smem) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    val = warp_reduce_sum(val);
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();
    if (warp_id == 0) {
        val = (threadIdx.x < BLOCK_SIZE / 32) ? smem[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();
    return smem[0];
}


// One block per row. Each thread handles a stride of columns.
// Numerically stable: subtract max before exp.
template <int BLOCK_SIZE>
__global__ void softmax_kernel(const float* x, float* y, int N) {
    int row = blockIdx.x;
    const float* row_in = x + row * N;
    float* row_out = y + row * N;

    __shared__ float smem[BLOCK_SIZE / 32];

    // Phase 1: find row max
    float thread_max = -FLT_MAX;
    for (int j = threadIdx.x; j < N; j += BLOCK_SIZE) {
        thread_max = fmaxf(thread_max, row_in[j]);
    }
    float row_max = block_reduce_max<BLOCK_SIZE>(thread_max, smem);

    // Phase 2: compute exp(x - max), sum
    float thread_sum = 0.0f;
    for (int j = threadIdx.x; j < N; j += BLOCK_SIZE) {
        float e = __expf(row_in[j] - row_max);
        row_out[j] = e;        // store the unnormalized for now
        thread_sum += e;
    }
    float row_sum = block_reduce_sum<BLOCK_SIZE>(thread_sum, smem);

    // Phase 3: normalize
    float inv_sum = 1.0f / row_sum;
    for (int j = threadIdx.x; j < N; j += BLOCK_SIZE) {
        row_out[j] *= inv_sum;
    }
}


// CPU reference for verification
void softmax_cpu(const float* x, float* y, int B, int N) {
    for (int i = 0; i < B; i++) {
        float m = -FLT_MAX;
        for (int j = 0; j < N; j++) m = fmaxf(m, x[i * N + j]);
        float s = 0.0f;
        for (int j = 0; j < N; j++) {
            float e = expf(x[i * N + j] - m);
            y[i * N + j] = e;
            s += e;
        }
        for (int j = 0; j < N; j++) y[i * N + j] /= s;
    }
}


int main() {
    const int B = 64;       // batch size (rows)
    const int N = 4096;     // columns per row (vocab-size-shaped)
    const size_t bytes = B * N * sizeof(float);

    float* h_x = (float*)malloc(bytes);
    float* h_y = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);

    // Fill with values that will trigger overflow if we forget to subtract max
    for (int i = 0; i < B * N; i++) h_x[i] = (float)((i % 100) - 50) + 50.0f;  // up to ~100

    softmax_cpu(h_x, h_ref, B, N);

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    // Warmup
    softmax_kernel<256><<<B, 256>>>(d_x, d_y, N);
    cudaDeviceSynchronize();

    // Time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) softmax_kernel<256><<<B, 256>>>(d_x, d_y, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 100;

    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

    // Verify against CPU reference
    float max_err = 0.0f;
    for (int i = 0; i < B * N; i++) {
        max_err = fmaxf(max_err, fabsf(h_y[i] - h_ref[i]));
    }
    bool ok = max_err < 1e-5f;

    // Bandwidth: 2 reads + 1 write per element (max pass + sum pass + normalize pass)
    // Realistically more — closer to 4x for our 3-phase implementation.
    double gb = 4.0 * B * N * sizeof(float) / 1e9;
    double bw = gb / (ms / 1000.0);

    printf("Softmax (B=%d, N=%d):\n", B, N);
    printf("  time:           %.3f ms\n", ms);
    printf("  bandwidth:      %.1f GB/s (lower bound; counts ~4x traffic)\n", bw);
    printf("  max abs error:  %.2e\n", max_err);
    printf("  result:         %s\n", ok ? "OK" : "FAILED");

    free(h_x); free(h_y); free(h_ref);
    cudaFree(d_x); cudaFree(d_y);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ok ? 0 : 1;
}
