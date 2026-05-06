// Three matmul kernels: naive, coalesced, shared-memory tiled.
// Steps 1, 2, 3 of Boehm's progression. Read his blog for steps 4-7.
//
// Compile (T4 = sm_75, A100 = sm_80, H100 = sm_90):
//     nvcc -O3 -arch=sm_80 matmul.cu -lcublas -o matmul
// Run:
//     ./matmul

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do {                                                 \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(err)); exit(1);      \
    }                                                                         \
} while (0)


// ---- Step 1 — naive ----
// One thread per output element. Each thread does K multiply-adds reading from HBM.
__global__ void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}


// ---- Step 2 — global memory coalescing ----
// Subtle change: swap how (threadIdx.x, threadIdx.y) maps to (col, row).
// Now consecutive threads in a warp have consecutive `col` values, ensuring
// B reads are contiguous in memory (B[k*N + col] for col = 0,1,2,...).
__global__ void matmul_coalesced(const float* A, const float* B, float* C, int M, int N, int K) {
    // Use 1D thread layout, decompose by hand
    const int BLOCK_SIZE = 32;
    int x = blockIdx.x * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);
    int y = blockIdx.y * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
    if (x < N && y < M) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) sum += A[y * K + k] * B[k * N + x];
        C[y * N + x] = sum;
    }
}


// ---- Step 3 — shared memory tiling ----
// Each block computes a BM x BN tile of C, looping over BK chunks of K.
// All threads in the block cooperate to load A and B tiles into shared memory once,
// then reuse across the inner BK iterations.
template <int BM, int BN, int BK>
__global__ void matmul_smem(const float* A, const float* B, float* C, int M, int N, int K) {
    // Shared-memory tiles
    __shared__ float A_smem[BM][BK];
    __shared__ float B_smem[BK][BN];

    // Block-level coords (which BM × BN tile of C are we computing)
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    // Thread coords within block
    // Layout: BM rows × BN cols of threads (assumes BM*BN <= 1024 = max threads/block)
    int tx = threadIdx.x;          // 0..BN-1
    int ty = threadIdx.y;          // 0..BM-1

    // Output coords in C
    int row = block_row * BM + ty;
    int col = block_col * BN + tx;

    float acc = 0.0f;

    // Loop over K in BK-sized chunks
    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // Cooperatively load A and B tiles. Each thread loads one element of each.
        // (Assumes BM*BN >= BM*BK and BM*BN >= BK*BN. With BM=BN=32 and BK=32, that's 1024.)
        // For BM=BN=BK=32 each thread loads exactly one element of A and one of B.
        if (k_tile + tx < K && row < M)
            A_smem[ty][tx] = A[row * K + k_tile + tx];
        else
            A_smem[ty][tx] = 0.0f;

        if (k_tile + ty < K && col < N)
            B_smem[ty][tx] = B[(k_tile + ty) * N + col];
        else
            B_smem[ty][tx] = 0.0f;

        __syncthreads();   // make sure all loads complete before computing

        // Compute partial dot product on the in-SMEM tiles
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            acc += A_smem[ty][kk] * B_smem[kk][tx];
        }

        __syncthreads();   // make sure all compute completes before next load
    }

    if (row < M && col < N) C[row * N + col] = acc;
}


// ---- Reference: cuBLAS ----
void matmul_cublas(cublasHandle_t handle, const float* A, const float* B, float* C, int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    // cuBLAS is column-major; we have row-major data. Compute C^T = B^T · A^T which equals
    // (in column-major) C = A · B with the right strides. Standard trick.
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B, N,    // B is K×N row-major = N×K column-major
                A, K,
                &beta,
                C, N);
}


// ---- Bench helper ----
template <typename Launcher>
double bench(const char* name, Launcher launcher, const float* A, const float* B, float* C,
             int M, int N, int K, int RUNS = 50) {
    // Warmup
    launcher();
    cudaDeviceSynchronize();

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < RUNS; i++) launcher();
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float ms;
    cudaEventElapsedTime(&ms, s, e);
    ms /= RUNS;

    double tflops = (2.0 * M * N * K) / (ms / 1000.0) / 1e12;
    printf("%-12s  %.3f ms   %6.2f TFLOPS\n", name, ms, tflops);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return tflops;
}


int main() {
    const int M = 4096, N = 4096, K = 4096;
    const size_t bytes_A = M * K * sizeof(float);
    const size_t bytes_B = K * N * sizeof(float);
    const size_t bytes_C = M * N * sizeof(float);

    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C = (float*)malloc(bytes_C);
    float* h_C_ref = (float*)malloc(bytes_C);
    for (int i = 0; i < M*K; i++) h_A[i] = (float)(rand() % 7) / 7.0f - 0.5f;
    for (int i = 0; i < K*N; i++) h_B[i] = (float)(rand() % 7) / 7.0f - 0.5f;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    cublasHandle_t cublas;
    cublasCreate(&cublas);

    printf("M=N=K=%d  (FP32, %.1f GFLOPs of work per call)\n\n", M, 2.0 * M * N * K / 1e9);
    printf("%-12s  %-9s  %s\n", "kernel", "time", "TFLOPS");
    printf("%-12s  %-9s  %s\n", "------", "----", "------");

    // Naive
    bench("naive", [&]() {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }, d_A, d_B, d_C, M, N, K);

    // Coalesced
    bench("coalesced", [&]() {
        dim3 block(32 * 32);  // 1024 threads, decomposed inside the kernel
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        matmul_coalesced<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }, d_A, d_B, d_C, M, N, K);

    // SMEM tiled
    bench("smem_tiled", [&]() {
        constexpr int BM = 32, BN = 32, BK = 32;
        dim3 block(BN, BM);  // (tx, ty)
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        matmul_smem<BM, BN, BK><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }, d_A, d_B, d_C, M, N, K);

    // cuBLAS reference
    bench("cuBLAS", [&]() { matmul_cublas(cublas, d_A, d_B, d_C, M, N, K); },
          d_A, d_B, d_C, M, N, K);

    // Correctness: compare last (cuBLAS) result against smem_tiled run
    CUDA_CHECK(cudaMemcpy(h_C_ref, d_C, bytes_C, cudaMemcpyDeviceToHost));
    {
        constexpr int BM = 32, BN = 32, BK = 32;
        dim3 block(BN, BM);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        matmul_smem<BM, BN, BK><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < M * N; i++) max_err = fmaxf(max_err, fabsf(h_C[i] - h_C_ref[i]));
    printf("\nmax abs error vs cuBLAS: %.2e  %s\n", max_err, max_err < 1e-2f ? "OK" : "FAILED");

    cublasDestroy(cublas);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
