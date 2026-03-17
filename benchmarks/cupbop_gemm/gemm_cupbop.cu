/**
 * CuPBoP-Vortex compatible GEMM kernel.
 *
 * Implements the same tiled GEMM algorithm as the CUTLASS 3.0 version
 * (gemm.cu) but replaces SM90-specific features (TMA, GMMA, clusters)
 * with portable CUDA operations (shared memory tiling, FMA loops).
 *
 * Original: ColfaxResearch/cutlass-kernels/src/cutlass-gemm/gemm.cu
 * Algorithm: C = alpha * A * B + beta * C  (row-major A, col-major B, col-major C)
 *
 * For CuPBoP-Vortex, we use small tile/problem sizes to fit in simx.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

// Tile dimensions (matching cutlass-gemm concept but Vortex-sized)
#define TILE_M 8
#define TILE_N 8
#define TILE_K 8

// Tiled GEMM kernel: C = alpha * A * B + beta * C
// A is (M x K) row-major, B is (K x N) row-major, C is (M x N) row-major
__global__
void gemm_kernel(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 float alpha, float beta) {
    // Shared memory tiles (replaces TMA loads in SM90 version)
    __shared__ float tileA[TILE_M * TILE_K];
    __shared__ float tileB[TILE_K * TILE_N];

    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    float sum = 0.0f;

    int numTiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < numTiles; t++) {
        // Cooperative load of A tile (replaces TMA)
        int aCol = t * TILE_K + threadIdx.x;
        if (row < M && aCol < K)
            tileA[threadIdx.y * TILE_K + threadIdx.x] = A[row * K + aCol];
        else
            tileA[threadIdx.y * TILE_K + threadIdx.x] = 0.0f;

        // Cooperative load of B tile (replaces TMA)
        int bRow = t * TILE_K + threadIdx.y;
        if (bRow < K && col < N)
            tileB[threadIdx.y * TILE_N + threadIdx.x] = B[bRow * N + col];
        else
            tileB[threadIdx.y * TILE_N + threadIdx.x] = 0.0f;

        __syncthreads();

        // FMA accumulate (replaces GMMA tensor core ops)
        for (int k = 0; k < TILE_K; k++) {
            sum += tileA[threadIdx.y * TILE_K + k] * tileB[k * TILE_N + threadIdx.x];
        }

        __syncthreads();
    }

    // Epilogue: C = alpha * A*B + beta * C (same as cutlass epilogue)
    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// CPU reference GEMM (replaces cuBLAS reference in original)
void gemm_cpu(const float* A, const float* B, float* C,
              int M, int N, int K, float alpha, float beta) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

int run_gemm_test(const char* name, int M, int N, int K,
                  float alpha, float beta, int seed) {
    int sizeA = M * K;
    int sizeB = K * N;
    int sizeC = M * N;

    printf("\n===== GEMM TEST: %s =====\n", name);
    printf("M=%d, N=%d, K=%d, alpha=%.1f, beta=%.1f\n", M, N, K, alpha, beta);
    printf("Grid=(%d,%d), Block=(%d,%d)\n",
           (N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M,
           TILE_N, TILE_M);

    // Host allocations (matching original's initialize pattern)
    float *h_A = (float*)malloc(sizeA * sizeof(float));
    float *h_B = (float*)malloc(sizeB * sizeof(float));
    float *h_C = (float*)malloc(sizeC * sizeof(float));
    float *h_ref = (float*)malloc(sizeC * sizeof(float));

    // Random init (matching original's seed-based init, scope [-8, 8])
    srand(seed);
    for (int i = 0; i < sizeA; i++)
        h_A[i] = ((float)(rand() % 1000) / 1000.0f) * 16.0f - 8.0f;
    for (int i = 0; i < sizeB; i++)
        h_B[i] = ((float)(rand() % 1000) / 1000.0f) * 16.0f - 8.0f;
    for (int i = 0; i < sizeC; i++) {
        h_C[i] = ((float)(rand() % 1000) / 1000.0f) * 16.0f - 8.0f;
        h_ref[i] = h_C[i];  // copy for reference
    }

    // CPU reference
    gemm_cpu(h_A, h_B, h_ref, M, N, K, alpha, beta);

    // Device allocations
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA * sizeof(float));
    cudaMalloc(&d_B, sizeB * sizeof(float));
    cudaMalloc(&d_C, sizeC * sizeof(float));

    cudaMemcpy(d_A, h_A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sizeC * sizeof(float), cudaMemcpyHostToDevice);

    // Launch (2D grid/block like original, but Vortex-sized)
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(h_C, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify (matching original's verify pattern)
    int passed = 1;
    float max_err = 0.0f;
    for (int i = 0; i < sizeC; i++) {
        float err = fabsf(h_C[i] - h_ref[i]);
        float limit = 1e-2f + 1e-2f * fabsf(h_ref[i]);  // atol + rtol
        if (err > max_err) max_err = err;
        if (err > limit) {
            printf("FAILED at [%d]: expected=%f got=%f (err=%f)\n",
                   i, h_ref[i], h_C[i], err);
            passed = 0;
            break;
        }
    }

    if (passed) printf("PASSED (max error: %e)\n", max_err);
    else printf("FAILED\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return passed;
}

// Known-value test with explicit A, B, expected C
int run_known_gemm(const char* name, const float* A, const float* B,
                   const float* expected_C, int M, int N, int K,
                   float alpha, float beta) {
    int sizeA = M * K, sizeB = K * N, sizeC = M * N;

    printf("\n===== GEMM KNOWN TEST: %s =====\n", name);

    float *h_C = (float*)malloc(sizeC * sizeof(float));
    for (int i = 0; i < sizeC; i++) h_C[i] = 0.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA * sizeof(float));
    cudaMalloc(&d_B, sizeB * sizeof(float));
    cudaMalloc(&d_C, sizeC * sizeof(float));
    cudaMemcpy(d_A, A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sizeC * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

    int passed = 1;
    float max_err = 0;
    for (int i = 0; i < sizeC; i++) {
        float err = fabsf(h_C[i] - expected_C[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-3f) {
            printf("FAILED at [%d]: expected=%f got=%f\n", i, expected_C[i], h_C[i]);
            passed = 0; break;
        }
    }
    if (passed) printf("PASSED (max error: %e)\n", max_err);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_C);
    return passed;
}

int main() {
    cudaSetDevice(0);

    int all_passed = 1;

    // Test 1-3: Random
    all_passed &= run_gemm_test("random_8x8", 8, 8, 8, 1.0f, 0.0f, 42);
    all_passed &= run_gemm_test("random_16x16", 16, 16, 16, 1.0f, 0.0f, 123);
    all_passed &= run_gemm_test("random_alpha_beta", 8, 8, 16, 2.0f, 0.5f, 456);

    // Test 4: Identity matrix — A * I = A
    {
        float A[64], I[64], expected[64];
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++) {
                A[i*8+j] = (float)(i * 8 + j + 1);  // [1..64]
                I[i*8+j] = (i == j) ? 1.0f : 0.0f;
                expected[i*8+j] = A[i*8+j];          // A * I = A
            }
        all_passed &= run_known_gemm("identity_AI_eq_A", A, I, expected, 8, 8, 8, 1.0f, 0.0f);
    }

    // Test 5: All-ones — each element of C = K (sum of K ones)
    {
        float ones_A[64], ones_B[64], expected[64];
        for (int i = 0; i < 64; i++) { ones_A[i] = 1.0f; ones_B[i] = 1.0f; }
        for (int i = 0; i < 64; i++) expected[i] = 8.0f;  // 8x8, K=8: sum=8
        all_passed &= run_known_gemm("all_ones_sum_K", ones_A, ones_B, expected, 8, 8, 8, 1.0f, 0.0f);
    }

    // Test 6: Zero matrix — A * 0 = 0
    {
        float A[64], Z[64], expected[64];
        for (int i = 0; i < 64; i++) { A[i] = (float)(i+1); Z[i] = 0.0f; expected[i] = 0.0f; }
        all_passed &= run_known_gemm("zero_matrix", A, Z, expected, 8, 8, 8, 1.0f, 0.0f);
    }

    // Test 7: alpha=0, beta=1 → C stays unchanged (C = 0*AB + 1*C = C)
    all_passed &= run_gemm_test("alpha0_beta1_noop", 8, 8, 8, 0.0f, 1.0f, 777);

    printf("\n===== SUMMARY =====\n");
    printf("%s\n", all_passed ? "ALL GEMM TESTS PASSED" : "SOME GEMM TESTS FAILED");

    return all_passed ? 0 : 1;
}
