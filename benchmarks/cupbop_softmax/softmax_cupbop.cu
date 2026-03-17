/**
 * CuPBoP-Vortex Softmax kernel.
 *
 * Row-wise softmax: out[i][j] = exp(x[i][j] - max_i) / sum_j(exp(x[i][j] - max_i))
 *
 * Related to the online_softmax.h in cutlass-kernels/src/fmha/ but
 * implemented as a standalone kernel for benchmarking.
 *
 * Uses shared memory for cooperative row-max and row-sum reductions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

// One block per row, threads cooperate on columns
// Shared memory: input row + partial max + partial sum
__global__
void softmax_kernel(const float* input, float* output, int rows, int cols) {
    extern __shared__ float smem[];
    float* row_data = smem;           // cols floats
    float* reduce_buf = &smem[cols];  // blockDim.x floats

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;

    // Load row into shared memory
    for (int j = tid; j < cols; j += blockDim.x) {
        row_data[j] = input[row * cols + j];
    }
    __syncthreads();

    // Step 1: Find row max (cooperative reduction)
    float local_max = -1e30f;
    for (int j = tid; j < cols; j += blockDim.x) {
        if (row_data[j] > local_max) local_max = row_data[j];
    }
    reduce_buf[tid] = local_max;
    __syncthreads();

    // Tree reduction for max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (reduce_buf[tid + s] > reduce_buf[tid])
                reduce_buf[tid] = reduce_buf[tid + s];
        }
        __syncthreads();
    }
    float row_max = reduce_buf[0];

    // Step 2: Compute exp(x - max) and row sum
    float local_sum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        row_data[j] = expf(row_data[j] - row_max);
        local_sum += row_data[j];
    }
    reduce_buf[tid] = local_sum;
    __syncthreads();

    // Tree reduction for sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            reduce_buf[tid] += reduce_buf[tid + s];
        }
        __syncthreads();
    }
    float row_sum = reduce_buf[0];

    // Step 3: Normalize
    float inv_sum = 1.0f / row_sum;
    for (int j = tid; j < cols; j += blockDim.x) {
        output[row * cols + j] = row_data[j] * inv_sum;
    }
}

void softmax_cpu(const float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float max_val = input[i * cols];
        for (int j = 1; j < cols; j++)
            if (input[i * cols + j] > max_val) max_val = input[i * cols + j];
        float sum = 0;
        for (int j = 0; j < cols; j++) {
            output[i * cols + j] = expf(input[i * cols + j] - max_val);
            sum += output[i * cols + j];
        }
        for (int j = 0; j < cols; j++)
            output[i * cols + j] /= sum;
    }
}

int run_test(const char* name, int rows, int cols, int block_size, int seed) {
    int size = rows * cols;

    printf("\n===== SOFTMAX TEST: %s =====\n", name);
    printf("rows=%d, cols=%d, block_size=%d\n", rows, cols, block_size);

    float *h_in = (float*)malloc(size * sizeof(float));
    float *h_out = (float*)malloc(size * sizeof(float));
    float *h_ref = (float*)malloc(size * sizeof(float));

    srand(seed);
    for (int i = 0; i < size; i++)
        h_in[i] = (float)(rand() % 1000) / 100.0f - 5.0f;

    softmax_cpu(h_in, h_ref, rows, cols);

    float *d_in, *d_out;
    cudaMalloc(&d_in, size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));
    cudaMemcpy(d_in, h_in, size * sizeof(float), cudaMemcpyHostToDevice);

    int smem_size = (cols + block_size) * sizeof(float);
    softmax_kernel<<<rows, block_size, smem_size>>>(d_in, d_out, rows, cols);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

    int passed = 1;
    float max_err = 0;
    for (int i = 0; i < size; i++) {
        float err = fabsf(h_out[i] - h_ref[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-4f) {
            printf("FAILED at [%d]: expected=%f got=%f\n", i, h_ref[i], h_out[i]);
            passed = 0; break;
        }
    }
    if (passed) printf("PASSED (max error: %e)\n", max_err);

    // Verify rows sum to 1.0
    for (int i = 0; i < rows && passed; i++) {
        float sum = 0;
        for (int j = 0; j < cols; j++) sum += h_out[i * cols + j];
        if (fabsf(sum - 1.0f) > 1e-3f) {
            printf("FAILED: row %d sums to %f (expected 1.0)\n", i, sum);
            passed = 0;
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out); free(h_ref);
    return passed;
}

int main() {
    cudaSetDevice(0);
    int ok = 1;

    // Test 1: Small (1 block, 8 threads, 8 cols)
    ok &= run_test("small_8x8", 8, 8, 8, 42);

    // Test 2: Wide rows (8 threads handle 16 cols via loop)
    ok &= run_test("wide_4x16", 4, 16, 8, 123);

    // Test 3: Many rows
    ok &= run_test("tall_16x8", 16, 8, 8, 456);

    printf("\n===== SUMMARY =====\n");
    printf("%s\n", ok ? "ALL SOFTMAX TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
