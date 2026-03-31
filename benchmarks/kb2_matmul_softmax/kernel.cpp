#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// Fused Matmul + Row-wise Softmax (attention score pattern)
// C[row] = softmax( A[row] @ B )
// One task per output row.

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto a_ptr = reinterpret_cast<float*>(arg->a_addr);
    auto b_ptr = reinterpret_cast<float*>(arg->b_addr);
    auto c_ptr = reinterpret_cast<float*>(arg->c_addr);

    uint32_t row = blockIdx.x;
    uint32_t N   = arg->N;
    uint32_t K   = arg->K;

    // Phase 1: Compute matmul row and find max
    float max_val = -1e30f;
    for (uint32_t j = 0; j < N; ++j) {
        float dot = 0.0f;
        for (uint32_t k = 0; k < K; ++k) {
            dot += a_ptr[row * K + k] * b_ptr[k * N + j];
        }
        c_ptr[row * N + j] = dot;  // store temporarily
        if (dot > max_val) max_val = dot;
    }

    // Phase 2: exp(x - max) and sum
    float sum_exp = 0.0f;
    for (uint32_t j = 0; j < N; ++j) {
        float e = expf(c_ptr[row * N + j] - max_val);
        c_ptr[row * N + j] = e;
        sum_exp += e;
    }

    // Phase 3: Normalize
    float inv_sum = 1.0f / sum_exp;
    for (uint32_t j = 0; j < N; ++j) {
        c_ptr[row * N + j] *= inv_sum;
    }
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->M, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
