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

int main() {
    cudaSetDevice(0);
    int ok = 1;

    // Test 1: Small (8 rows x 8 features)
    ok &= run_test("small_8x8", 8, 8, 8, 42);

    // Test 2: Wide features (4 rows x 16 features, threads loop)
    ok &= run_test("wide_4x16", 4, 16, 8, 123);

    // Test 3: Tall (16 rows x 8 features)
    ok &= run_test("tall_16x8", 16, 8, 8, 456);

    printf("\n===== SUMMARY =====\n");
    printf("%s\n", ok ? "ALL LAYERNORM TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
