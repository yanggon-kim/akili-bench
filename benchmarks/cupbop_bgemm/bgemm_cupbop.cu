/**
 * CuPBoP-Vortex Batched GEMM kernel.
 *
 * Batched GEMM: C[b] = alpha * A[b] * B[b] + beta * C[b]  for b in [0, batch)
 *
 * Essential for multi-head attention where each head computes an
 * independent GEMM. Extends the cutlass-gemm to 3D grid dispatch
 * with batch dimension in blockIdx.z.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define TILE_M 8
#define TILE_N 8
#define TILE_K 8

// Batched tiled GEMM: blockIdx.z selects the batch
__global__
void batched_gemm_kernel(const float* A, const float* B, float* C,
                         int M, int N, int K, int batch,
                         float alpha, float beta) {
    __shared__ float tileA[TILE_M * TILE_K];
    __shared__ float tileB[TILE_K * TILE_N];

    int b = blockIdx.z;
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    // Per-batch offset
    int batch_offset_A = b * M * K;
    int batch_offset_B = b * K * N;
    int batch_offset_C = b * M * N;

    float sum = 0.0f;
    int numTiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_K + threadIdx.x;
        if (row < M && aCol < K)
            tileA[threadIdx.y * TILE_K + threadIdx.x] = A[batch_offset_A + row * K + aCol];
        else
            tileA[threadIdx.y * TILE_K + threadIdx.x] = 0.0f;

        int bRow = t * TILE_K + threadIdx.y;
        if (bRow < K && col < N)
            tileB[threadIdx.y * TILE_N + threadIdx.x] = B[batch_offset_B + bRow * N + col];
        else
            tileB[threadIdx.y * TILE_N + threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_K; k++)
            sum += tileA[threadIdx.y * TILE_K + k] * tileB[k * TILE_N + threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N) {
        int idx = batch_offset_C + row * N + col;
        C[idx] = alpha * sum + beta * C[idx];
    }
}

void bgemm_cpu(const float* A, const float* B, float* C,
               int M, int N, int K, int batch, float alpha, float beta) {
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                float sum = 0;
                for (int k = 0; k < K; k++)
                    sum += A[b*M*K + i*K + k] * B[b*K*N + k*N + j];
                C[b*M*N + i*N + j] = alpha * sum + beta * C[b*M*N + i*N + j];
            }
    }
}

int run_test(const char* name, int batch, int M, int N, int K,
             float alpha, float beta, int seed) {
    int sizeA = batch * M * K;
    int sizeB = batch * K * N;
    int sizeC = batch * M * N;

    printf("\n===== BATCHED GEMM TEST: %s =====\n", name);
    printf("batch=%d, M=%d, N=%d, K=%d, alpha=%.1f, beta=%.1f\n",
           batch, M, N, K, alpha, beta);

    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M, batch);
    printf("Grid=(%d,%d,%d), Block=(%d,%d)\n", grid.x, grid.y, grid.z, block.x, block.y);

    float *h_A = (float*)malloc(sizeA * sizeof(float));
    float *h_B = (float*)malloc(sizeB * sizeof(float));
    float *h_C = (float*)malloc(sizeC * sizeof(float));
    float *h_ref = (float*)malloc(sizeC * sizeof(float));

    srand(seed);
    for (int i = 0; i < sizeA; i++)
        h_A[i] = ((float)(rand() % 1000) / 1000.0f) * 4.0f - 2.0f;
    for (int i = 0; i < sizeB; i++)
        h_B[i] = ((float)(rand() % 1000) / 1000.0f) * 4.0f - 2.0f;
    for (int i = 0; i < sizeC; i++) {
        h_C[i] = ((float)(rand() % 1000) / 1000.0f) * 4.0f - 2.0f;
        h_ref[i] = h_C[i];
    }

    bgemm_cpu(h_A, h_B, h_ref, M, N, K, batch, alpha, beta);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA * sizeof(float));
    cudaMalloc(&d_B, sizeB * sizeof(float));
    cudaMalloc(&d_C, sizeC * sizeof(float));
    cudaMemcpy(d_A, h_A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sizeC * sizeof(float), cudaMemcpyHostToDevice);

    batched_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, batch, alpha, beta);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

    int passed = 1;
    float max_err = 0;
    for (int i = 0; i < sizeC; i++) {
        float err = fabsf(h_C[i] - h_ref[i]);
        if (err > max_err) max_err = err;
        float limit = 1e-2f + 1e-2f * fabsf(h_ref[i]);
        if (err > limit) {
            printf("FAILED at [%d]: expected=%f got=%f\n", i, h_ref[i], h_C[i]);
            passed = 0; break;
        }
    }
    if (passed) printf("PASSED (max error: %e)\n", max_err);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return passed;
}

int main() {
    cudaSetDevice(0);
    int ok = 1;

    // Test 1: 2 batches, 8x8 (tests blockIdx.z dispatch)
    ok &= run_test("batch2_8x8", 2, 8, 8, 8, 1.0f, 0.0f, 42);

    // Test 2: 4 batches, multi-tile
    ok &= run_test("batch4_8x8_k16", 4, 8, 8, 16, 1.0f, 0.0f, 123);

    // Test 3: Batched with alpha/beta
    ok &= run_test("batch2_alpha_beta", 2, 8, 8, 8, 2.0f, 0.5f, 456);

    printf("\n===== SUMMARY =====\n");
    printf("%s\n", ok ? "ALL BATCHED GEMM TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
