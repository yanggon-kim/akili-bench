#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

// Flash Attention forward kernel (from tspeterkim/flash-attention-minimal)
// Adapted for standalone CuPBoP-Vortex execution (no PyTorch dependency)
__global__
void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++)  {

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            float row_m = -1e30f;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = (row_m_prev > row_m) ? row_m_prev : row_m;
            float row_l_new = (expf(row_m_prev - row_m_new) * row_l_prev) + (expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1.0f / row_l_new) \
                    * ((row_l_prev * expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}

// CPU reference: standard (naive) attention — softmax(Q*K^T / sqrt(d)) * V
// This is the textbook formula, NOT flash attention. Used as ground truth.
void standard_attention_cpu(const float* Q, const float* K, const float* V,
                            float* O, int B, int nh, int N, int d) {
    float softmax_scale = 1.0f / sqrtf((float)d);

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < nh; h++) {
            int offset = (b * nh + h) * N * d;

            for (int i = 0; i < N; i++) {
                float scores[64];
                float max_score = -1e30f;
                for (int j = 0; j < N; j++) {
                    float sum = 0;
                    for (int k = 0; k < d; k++) {
                        sum += Q[offset + i * d + k] * K[offset + j * d + k];
                    }
                    scores[j] = sum * softmax_scale;
                    if (scores[j] > max_score) max_score = scores[j];
                }
                float sum_exp = 0;
                for (int j = 0; j < N; j++) {
                    scores[j] = expf(scores[j] - max_score);
                    sum_exp += scores[j];
                }
                for (int j = 0; j < N; j++) scores[j] /= sum_exp;
                for (int k = 0; k < d; k++) {
                    float val = 0;
                    for (int j = 0; j < N; j++) val += scores[j] * V[offset + j * d + k];
                    O[offset + i * d + k] = val;
                }
            }
        }
    }
}

// Run one test case: allocate, init, launch kernel, verify against CPU
int run_test(const char* name, int B, int nh, int N, int d, int Bc, int Br, int seed) {
    int Tc = (N + Bc - 1) / Bc;
    int Tr = (N + Br - 1) / Br;
    float softmax_scale = 1.0f / sqrtf((float)d);

    int total_qkv = B * nh * N * d;
    int total_lm = B * nh * N;

    printf("\n===== TEST: %s =====\n", name);
    printf("B=%d, nh=%d, N=%d, d=%d, Bc=%d, Br=%d, Tc=%d, Tr=%d\n",
           B, nh, N, d, Bc, Br, Tc, Tr);
    printf("Grid=(%d,%d), Block=(%d), Threads=%d\n", B, nh, Bc, B * nh * Bc);

    float *h_Q = (float*)malloc(total_qkv * sizeof(float));
    float *h_K = (float*)malloc(total_qkv * sizeof(float));
    float *h_V = (float*)malloc(total_qkv * sizeof(float));
    float *h_O = (float*)malloc(total_qkv * sizeof(float));
    float *h_l = (float*)malloc(total_lm * sizeof(float));
    float *h_m = (float*)malloc(total_lm * sizeof(float));
    float *h_ref = (float*)malloc(total_qkv * sizeof(float));

    srand(seed);
    for (int i = 0; i < total_qkv; i++) {
        h_Q[i] = (float)(rand() % 100) / 100.0f - 0.5f;
        h_K[i] = (float)(rand() % 100) / 100.0f - 0.5f;
        h_V[i] = (float)(rand() % 100) / 100.0f - 0.5f;
    }
    for (int i = 0; i < total_qkv; i++) h_O[i] = 0.0f;
    for (int i = 0; i < total_lm; i++) { h_l[i] = 0.0f; h_m[i] = -1e30f; }

    // CPU ground truth: standard (naive) attention
    standard_attention_cpu(h_Q, h_K, h_V, h_ref, B, nh, N, d);

    printf("CPU ref[0..3]: %f %f %f %f\n", h_ref[0], h_ref[1], h_ref[2], h_ref[3]);

    float *d_Q, *d_K, *d_V, *d_O, *d_l, *d_m;
    cudaMalloc(&d_Q, total_qkv * sizeof(float));
    cudaMalloc(&d_K, total_qkv * sizeof(float));
    cudaMalloc(&d_V, total_qkv * sizeof(float));
    cudaMalloc(&d_O, total_qkv * sizeof(float));
    cudaMalloc(&d_l, total_lm * sizeof(float));
    cudaMalloc(&d_m, total_lm * sizeof(float));

    cudaMemcpy(d_Q, h_Q, total_qkv * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, total_qkv * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, total_qkv * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_O, h_O, total_qkv * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, h_l, total_lm * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, total_lm * sizeof(float), cudaMemcpyHostToDevice);

    int sram_size = (3 * Bc * d + Bc * Br) * (int)sizeof(float);
    printf("Shared memory: %d bytes\n", sram_size);

    dim3 grid_dim(B, nh);
    dim3 block_dim(Bc);

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        d_Q, d_K, d_V, N, d, Tc, Tr, Bc, Br, softmax_scale,
        d_l, d_m, d_O
    );
    cudaDeviceSynchronize();

    cudaMemcpy(h_O, d_O, total_qkv * sizeof(float), cudaMemcpyDeviceToHost);

    int passed = 1;
    float max_err = 0.0f;
    for (int i = 0; i < total_qkv; i++) {
        float err = fabsf(h_O[i] - h_ref[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-2f) {
            printf("FAILED at [%d]: expected=%f got=%f (err=%f)\n",
                   i, h_ref[i], h_O[i], err);
            passed = 0;
            break;
        }
    }

    printf("GPU out[0..3]: %f %f %f %f\n", h_O[0], h_O[1], h_O[2], h_O[3]);

    if (passed) {
        printf("PASSED (max error: %e)\n", max_err);
    } else {
        printf("FAILED\n");
    }

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_l); cudaFree(d_m);
    free(h_Q); free(h_K); free(h_V);
    free(h_O); free(h_l); free(h_m); free(h_ref);

    return passed;
}

int main() {
    cudaSetDevice(0);

    int all_passed = 1;

    // Test 1: Single tile (Tc=Tr=1) — basic sanity check
    //   B=1, nh=1, N=8, d=8, Bc=Br=8 → 1 block, 8 threads
    all_passed &= run_test("single_tile", 1, 1, 8, 8, 8, 8, 42);

    // Test 2: Multi-tile (Tc=Tr=2) — tests the tiling/online-softmax loop
    //   B=1, nh=1, N=16, d=4, Bc=Br=8 → 1 block, 8 threads, 2 tiles each dim
    all_passed &= run_test("multi_tile", 1, 1, 16, 4, 8, 8, 123);

    // Test 3: Multi-head (nh=2) — tests 2D grid dispatch
    //   B=1, nh=2, N=8, d=4, Bc=Br=8 → 2 blocks (grid=(1,2)), 8 threads each
    all_passed &= run_test("multi_head", 1, 2, 8, 4, 8, 8, 456);

    printf("\n===== SUMMARY =====\n");
    if (all_passed) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("SOME TESTS FAILED\n");
    }

    return all_passed ? 0 : 1;
}
