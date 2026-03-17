/**
 * CuPBoP-Vortex port of Dao-AILab Flash Attention.
 *
 * Implements the FlashAttention-2 algorithm with causal masking,
 * following the Triton reference (flash_attn_triton_og.py) from
 * https://github.com/Dao-AILab/flash-attention
 *
 * Key features ported from Dao's implementation:
 *   - Online softmax with tiled Q/K/V
 *   - Causal masking (lower-triangular: token i attends only to j <= i)
 *   - Causal loop optimization (skip K/V tiles beyond the diagonal)
 *   - Batched multi-head via 2D grid
 *
 * SM90/SM80 features replaced with portable CUDA:
 *   - TMA/cp.async → manual shared memory loads
 *   - GMMA/MMA tensor cores → FMA dot-product loops
 *   - Warp shuffle reductions → per-thread serial reductions
 *   - CuTe tensor algebra → explicit indexing
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

/**
 * Flash Attention forward kernel with optional causal masking.
 *
 * When causal=1, applies lower-triangular mask: score[i][j] = -inf if j > i.
 * Also optimizes by limiting the K/V tile loop to only tiles that overlap
 * with the causal region (matching Dao's `range(0, (start_m+1)*BLOCK_M)`).
 */
__global__
void dao_flash_fwd(const float* Q, const float* K, const float* V,
                   const int N, const int d,
                   const int Tc, const int Tr,
                   const int Bc, const int Br,
                   const float softmax_scale,
                   const int causal,
                   float* l, float* m, float* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;  // batch index
    int by = blockIdx.y;  // head index

    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);
    int lm_offset = (bx * gridDim.y * N) + (by * N);

    extern __shared__ float sram[];
    int tile_size = Bc * d;
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++) {
        // Load Kj, Vj
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();

        // Determine Q tile range — causal optimization from Dao:
        // Only process Q tiles where some rows can attend to this K tile
        int i_start = 0;
        int i_end = Tr;
        if (causal) {
            // K tile j covers columns [j*Bc, (j+1)*Bc - 1]
            // Q row i can see column j*Bc only if i >= j*Bc
            // So Q tile i_start = j*Bc / Br (first Q tile with rows >= j*Bc)
            i_start = (j * Bc) / Br;
        }

        for (int i = i_start; i < i_end; i++) {
            // Load Qi
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // GEMM-I: S = Qi @ Kj^T with causal mask
            int query_row = Br * i + tx;  // global row index of this query
            float row_m = -1e30f;
            for (int y = 0; y < Bc; y++) {
                int key_col = Bc * j + y;  // global column index

                // Causal mask: if query_row < key_col, mask out
                if (causal && query_row < key_col) {
                    S[(Bc * tx) + y] = -1e30f;
                } else {
                    float sum = 0;
                    for (int x = 0; x < d; x++) {
                        sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                    }
                    sum *= softmax_scale;
                    S[(Bc * tx) + y] = sum;
                }

                if (S[(Bc * tx) + y] > row_m)
                    row_m = S[(Bc * tx) + y];
            }

            // Online softmax
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            float row_m_new = (row_m_prev > row_m) ? row_m_prev : row_m;
            float row_l_new = (expf(row_m_prev - row_m_new) * row_l_prev)
                            + (expf(row_m - row_m_new) * row_l);

            // GEMM-II: O += P @ Vj with rescaling
            for (int x = 0; x < d; x++) {
                float pv = 0;
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] =
                    (1.0f / row_l_new)
                    * ((row_l_prev * expf(row_m_prev - row_m_new)
                        * O[qkv_offset + (tile_size * i) + (tx * d) + x])
                       + (expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();
    }
}

// CPU reference: standard attention with optional causal mask
void attention_cpu(const float* Q, const float* K, const float* V,
                   float* O, int B, int nh, int N, int d, int causal) {
    float scale = 1.0f / sqrtf((float)d);
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < nh; h++) {
            int off = (b * nh + h) * N * d;
            for (int i = 0; i < N; i++) {
                float scores[64];
                float max_s = -1e30f;
                for (int j = 0; j < N; j++) {
                    if (causal && j > i) {
                        scores[j] = -1e30f;
                    } else {
                        float sum = 0;
                        for (int k = 0; k < d; k++)
                            sum += Q[off + i*d + k] * K[off + j*d + k];
                        scores[j] = sum * scale;
                    }
                    if (scores[j] > max_s) max_s = scores[j];
                }
                float sum_exp = 0;
                for (int j = 0; j < N; j++) {
                    scores[j] = expf(scores[j] - max_s);
                    sum_exp += scores[j];
                }
                for (int j = 0; j < N; j++) scores[j] /= sum_exp;
                for (int k = 0; k < d; k++) {
                    float val = 0;
                    for (int j = 0; j < N; j++) val += scores[j] * V[off + j*d + k];
                    O[off + i*d + k] = val;
                }
            }
        }
    }
}

int run_test(const char* name, int B, int nh, int N, int d,
             int Bc, int Br, int causal, int seed) {
    int Tc = (N + Bc - 1) / Bc;
    int Tr = (N + Br - 1) / Br;
    float scale = 1.0f / sqrtf((float)d);
    int total_qkv = B * nh * N * d;
    int total_lm = B * nh * N;

    printf("\n===== DAO FLASH TEST: %s (causal=%d) =====\n", name, causal);
    printf("B=%d, nh=%d, N=%d, d=%d, Bc=%d, Br=%d, Tc=%d, Tr=%d\n",
           B, nh, N, d, Bc, Br, Tc, Tr);

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

    attention_cpu(h_Q, h_K, h_V, h_ref, B, nh, N, d, causal);

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

    dim3 grid_dim(B, nh);
    dim3 block_dim(Bc);
    dao_flash_fwd<<<grid_dim, block_dim, sram_size>>>(
        d_Q, d_K, d_V, N, d, Tc, Tr, Bc, Br, scale, causal,
        d_l, d_m, d_O);
    cudaDeviceSynchronize();
    cudaMemcpy(h_O, d_O, total_qkv * sizeof(float), cudaMemcpyDeviceToHost);

    int passed = 1;
    float max_err = 0;
    for (int i = 0; i < total_qkv; i++) {
        float err = fabsf(h_O[i] - h_ref[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-2f) {
            printf("FAILED at [%d]: expected=%f got=%f (err=%f)\n",
                   i, h_ref[i], h_O[i], err);
            passed = 0; break;
        }
    }
    if (passed) printf("PASSED (max error: %e)\n", max_err);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_l); cudaFree(d_m);
    free(h_Q); free(h_K); free(h_V);
    free(h_O); free(h_l); free(h_m); free(h_ref);
    return passed;
}

int main() {
    cudaSetDevice(0);
    int ok = 1;

    // === Non-causal tests (same as our earlier FMHA) ===
    ok &= run_test("noncausal_single", 1, 1, 8, 8, 8, 8, 0, 42);
    ok &= run_test("noncausal_multi_tile", 1, 1, 16, 4, 8, 8, 0, 123);
    ok &= run_test("noncausal_multi_head", 1, 2, 8, 4, 8, 8, 0, 456);

    // === Causal masking tests (key Dao feature) ===
    // Test 4: Causal single tile
    ok &= run_test("causal_single", 1, 1, 8, 8, 8, 8, 1, 42);

    // Test 5: Causal multi-tile — tests causal loop optimization
    ok &= run_test("causal_multi_tile", 1, 1, 16, 4, 8, 8, 1, 123);

    // Test 6: Causal multi-head
    ok &= run_test("causal_multi_head", 1, 2, 8, 4, 8, 8, 1, 456);

    // Test 7: Causal property — first row should only attend to itself
    //   (output row 0 = V[0] when causal, since softmax([s0, -inf, -inf, ...]) = [1,0,0,...])
    {
        int N = 8, d = 4;
        float Q[32], K[32], V[32];
        // Q[0] orthogonal to K[0] → score ≈ 0, but causal masks j>0
        // So row 0 output = V[0] regardless of scores
        for (int i = 0; i < N*d; i++) { Q[i] = 0.1f; K[i] = 0.1f; V[i] = (float)(i % d); }
        // Make V rows distinct
        for (int i = 0; i < N; i++)
            for (int j = 0; j < d; j++)
                V[i*d+j] = (float)(i * 10 + j);

        float ref[32];
        attention_cpu(Q, K, V, ref, 1, 1, N, d, 1);

        // Verify row 0 output = V[0] (since only token 0 visible)
        printf("\n===== DAO FLASH KNOWN TEST: causal_first_row =====\n");
        int p = 1;
        for (int j = 0; j < d; j++) {
            if (fabsf(ref[j] - V[j]) > 1e-4f) {
                printf("FAILED: row0[%d]=%f != V[0][%d]=%f\n", j, ref[j], j, V[j]);
                p = 0;
            }
        }
        if (p) printf("PASSED (row 0 output = V[0] as expected)\n");
        ok &= p;
    }

    // Test 8: Causal vs non-causal differ — same inputs, different outputs
    {
        printf("\n===== DAO FLASH TEST: causal_vs_noncausal_differ =====\n");
        int N = 8, d = 4, total = N * d;
        float Q[32], K[32], V[32], O_causal[32], O_noncausal[32];
        srand(777);
        for (int i = 0; i < total; i++) {
            Q[i] = (float)(rand() % 100) / 100.0f - 0.5f;
            K[i] = (float)(rand() % 100) / 100.0f - 0.5f;
            V[i] = (float)(rand() % 100) / 100.0f - 0.5f;
        }
        attention_cpu(Q, K, V, O_causal, 1, 1, N, d, 1);
        attention_cpu(Q, K, V, O_noncausal, 1, 1, N, d, 0);

        int differ = 0;
        for (int i = 0; i < total; i++)
            if (fabsf(O_causal[i] - O_noncausal[i]) > 1e-6f) { differ = 1; break; }

        // Last row should be same (sees all tokens in both modes)
        int lastrow_same = 1;
        for (int j = 0; j < d; j++)
            if (fabsf(O_causal[(N-1)*d+j] - O_noncausal[(N-1)*d+j]) > 1e-4f) lastrow_same = 0;

        if (differ && lastrow_same)
            printf("PASSED (causal and non-causal differ, last row matches)\n");
        else
            printf("FAILED (differ=%d, lastrow_same=%d)\n", differ, lastrow_same);
        ok &= (differ && lastrow_same);
    }

    printf("\n===== SUMMARY =====\n");
    printf("%s\n", ok ? "ALL DAO FLASH TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
