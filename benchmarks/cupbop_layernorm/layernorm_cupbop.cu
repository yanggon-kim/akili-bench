/**
 * CuPBoP-Vortex Layer Normalization kernel.
 *
 * LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
 *
 * Essential transformer building block. Each block handles one row
 * (one token's feature vector), computing mean and variance in
 * shared memory, then normalizing with learnable gamma/beta.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

__global__
void layernorm_kernel(const float* input, const float* gamma, const float* beta,
                      float* output, int rows, int cols, float eps) {
    extern __shared__ float smem[];
    float* row_data = smem;
    float* reduce_buf = &smem[cols];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    // Load row
    for (int j = tid; j < cols; j += blockDim.x)
        row_data[j] = input[row * cols + j];
    __syncthreads();

    // Compute mean (cooperative reduction)
    float local_sum = 0;
    for (int j = tid; j < cols; j += blockDim.x)
        local_sum += row_data[j];
    reduce_buf[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
        __syncthreads();
    }
    float mean = reduce_buf[0] / (float)cols;

    // Compute variance
    float local_var = 0;
    for (int j = tid; j < cols; j += blockDim.x) {
        float diff = row_data[j] - mean;
        local_var += diff * diff;
    }
    reduce_buf[tid] = local_var;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
        __syncthreads();
    }
    float var = reduce_buf[0] / (float)cols;
    float inv_std = 1.0f / sqrtf(var + eps);

    // Normalize with gamma and beta
    for (int j = tid; j < cols; j += blockDim.x) {
        output[row * cols + j] = gamma[j] * (row_data[j] - mean) * inv_std + beta[j];
    }
}

void layernorm_cpu(const float* input, const float* gamma, const float* beta,
                   float* output, int rows, int cols, float eps) {
    for (int i = 0; i < rows; i++) {
        float mean = 0;
        for (int j = 0; j < cols; j++) mean += input[i * cols + j];
        mean /= cols;
        float var = 0;
        for (int j = 0; j < cols; j++) {
            float d = input[i * cols + j] - mean;
            var += d * d;
        }
        var /= cols;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int j = 0; j < cols; j++)
            output[i * cols + j] = gamma[j] * (input[i * cols + j] - mean) * inv_std + beta[j];
    }
}

int run_test(const char* name, int rows, int cols, int block_size, int seed) {
    int size = rows * cols;
    float eps = 1e-5f;

    printf("\n===== LAYERNORM TEST: %s =====\n", name);
    printf("rows=%d, cols=%d, block_size=%d\n", rows, cols, block_size);

    float *h_in = (float*)malloc(size * sizeof(float));
    float *h_out = (float*)malloc(size * sizeof(float));
    float *h_ref = (float*)malloc(size * sizeof(float));
    float *h_gamma = (float*)malloc(cols * sizeof(float));
    float *h_beta = (float*)malloc(cols * sizeof(float));

    srand(seed);
    for (int i = 0; i < size; i++)
        h_in[i] = (float)(rand() % 1000) / 100.0f - 5.0f;
    for (int j = 0; j < cols; j++) {
        h_gamma[j] = 0.5f + (float)(rand() % 100) / 100.0f;  // [0.5, 1.5]
        h_beta[j] = (float)(rand() % 100) / 100.0f - 0.5f;   // [-0.5, 0.5]
    }

    layernorm_cpu(h_in, h_gamma, h_beta, h_ref, rows, cols, eps);

    float *d_in, *d_out, *d_gamma, *d_beta;
    cudaMalloc(&d_in, size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));
    cudaMalloc(&d_gamma, cols * sizeof(float));
    cudaMalloc(&d_beta, cols * sizeof(float));

    cudaMemcpy(d_in, h_in, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, cols * sizeof(float), cudaMemcpyHostToDevice);

    int smem_size = (cols + block_size) * sizeof(float);
    layernorm_kernel<<<rows, block_size, smem_size>>>(
        d_in, d_gamma, d_beta, d_out, rows, cols, eps);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

    int passed = 1;
    float max_err = 0;
    for (int i = 0; i < size; i++) {
        float err = fabsf(h_out[i] - h_ref[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-3f) {
            printf("FAILED at [%d]: expected=%f got=%f (err=%f)\n", i, h_ref[i], h_out[i], err);
            passed = 0; break;
        }
    }
    if (passed) printf("PASSED (max error: %e)\n", max_err);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_gamma); cudaFree(d_beta);
    free(h_in); free(h_out); free(h_ref); free(h_gamma); free(h_beta);
    return passed;
}

// Known-value test with explicit inputs
int run_known_test(const char* name, const float* input,
                   const float* gamma, const float* beta,
                   int rows, int cols, int block_size) {
    int size = rows * cols;
    float eps = 1e-5f;

    printf("\n===== LAYERNORM KNOWN TEST: %s =====\n", name);

    float *h_ref = (float*)malloc(size * sizeof(float));
    float *h_out = (float*)malloc(size * sizeof(float));
    layernorm_cpu(input, gamma, beta, h_ref, rows, cols, eps);

    float *d_in, *d_out, *d_gamma, *d_beta;
    cudaMalloc(&d_in, size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));
    cudaMalloc(&d_gamma, cols * sizeof(float));
    cudaMalloc(&d_beta, cols * sizeof(float));
    cudaMemcpy(d_in, input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma, cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta, cols * sizeof(float), cudaMemcpyHostToDevice);

    int smem_size = (cols + block_size) * sizeof(float);
    layernorm_kernel<<<rows, block_size, smem_size>>>(
        d_in, d_gamma, d_beta, d_out, rows, cols, eps);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

    int passed = 1;
    float max_err = 0;
    for (int i = 0; i < size; i++) {
        float err = fabsf(h_out[i] - h_ref[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-3f) {
            printf("FAILED at [%d]: expected=%f got=%f (err=%f)\n",
                   i, h_ref[i], h_out[i], err);
            passed = 0; break;
        }
    }

    // Property: with gamma=1, beta=0 AND non-constant input,
    // output should have mean≈0, var≈1 per row
    int gamma_is_one = 1, beta_is_zero = 1;
    for (int j = 0; j < cols; j++) {
        if (fabsf(gamma[j] - 1.0f) > 1e-6f) gamma_is_one = 0;
        if (fabsf(beta[j]) > 1e-6f) beta_is_zero = 0;
    }
    if (gamma_is_one && beta_is_zero && passed) {
        for (int i = 0; i < rows; i++) {
            // Check if input row has non-zero variance
            float in_mean = 0;
            for (int j = 0; j < cols; j++) in_mean += input[i * cols + j];
            in_mean /= cols;
            float in_var = 0;
            for (int j = 0; j < cols; j++) {
                float dd = input[i * cols + j] - in_mean;
                in_var += dd * dd;
            }
            in_var /= cols;
            if (in_var < 1e-6f) continue;  // skip constant rows

            float mean = 0;
            for (int j = 0; j < cols; j++) mean += h_out[i * cols + j];
            mean /= cols;
            if (fabsf(mean) > 1e-3f) {
                printf("FAILED: row %d mean=%f (expected ~0)\n", i, mean);
                passed = 0; break;
            }
            float var = 0;
            for (int j = 0; j < cols; j++) {
                float d = h_out[i * cols + j] - mean;
                var += d * d;
            }
            var /= cols;
            if (fabsf(var - 1.0f) > 0.05f) {
                printf("FAILED: row %d variance=%f (expected ~1)\n", i, var);
                passed = 0; break;
            }
        }
    }

    if (passed) printf("PASSED (max error: %e)\n", max_err);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_gamma); cudaFree(d_beta);
    free(h_ref); free(h_out);
    return passed;
}

int main() {
    cudaSetDevice(0);
    int ok = 1;

    // Test 1-3: Random inputs
    ok &= run_test("random_8x8", 8, 8, 8, 42);
    ok &= run_test("random_4x16", 4, 16, 8, 123);
    ok &= run_test("random_16x8", 16, 8, 8, 456);

    // Test 4: Constant row → output = beta (since (x-mean)/std = 0)
    {
        float in4[16];  // 2 rows x 8 cols
        for (int i = 0; i < 8; i++) in4[i] = 5.0f;       // row 0: all 5
        for (int i = 0; i < 8; i++) in4[8 + i] = -3.0f;   // row 1: all -3
        float g4[8], b4[8];
        for (int j = 0; j < 8; j++) { g4[j] = 1.0f; b4[j] = 0.0f; }
        // Expected: all outputs ≈ 0 (since variance is 0 → clipped by eps)
        ok &= run_known_test("constant_row", in4, g4, b4, 2, 8, 8);
    }

    // Test 5: gamma=1, beta=0 → output has mean≈0, var≈1
    {
        float in5[32];  // 4 rows x 8
        srand(999);
        for (int i = 0; i < 32; i++) in5[i] = (float)(rand() % 1000) / 100.0f;
        float g5[8], b5[8];
        for (int j = 0; j < 8; j++) { g5[j] = 1.0f; b5[j] = 0.0f; }
        ok &= run_known_test("unit_gamma_zero_beta", in5, g5, b5, 4, 8, 8);
    }

    // Test 6: Sequential input [1,2,3,...,8], gamma=2, beta=1
    {
        float in6[8];
        for (int i = 0; i < 8; i++) in6[i] = (float)(i + 1);
        float g6[8], b6[8];
        for (int j = 0; j < 8; j++) { g6[j] = 2.0f; b6[j] = 1.0f; }
        ok &= run_known_test("sequential_gamma2_beta1", in6, g6, b6, 1, 8, 8);
    }

    // Test 7: Large values (stress numerical stability)
    {
        float in7[16];
        for (int i = 0; i < 16; i++) in7[i] = 1000.0f + (float)i;
        float g7[8], b7[8];
        for (int j = 0; j < 8; j++) { g7[j] = 1.0f; b7[j] = 0.0f; }
        ok &= run_known_test("large_values", in7, g7, b7, 2, 8, 8);
    }

    printf("\n===== SUMMARY =====\n");
    printf("%s\n", ok ? "ALL LAYERNORM TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
