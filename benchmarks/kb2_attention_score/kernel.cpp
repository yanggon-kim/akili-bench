#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// Attention score: softmax( Q @ K^T / sqrt(d) )
// Q is [M x D], K is [N x D], output is [M x N]
// One task per output row (one query vector).

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto q_ptr = reinterpret_cast<float*>(arg->q_addr);
    auto k_ptr = reinterpret_cast<float*>(arg->k_addr);
    auto c_ptr = reinterpret_cast<float*>(arg->c_addr);

    uint32_t row = blockIdx.x;
    uint32_t N   = arg->N;
    uint32_t D   = arg->D;
    float inv_sqrt_d = 1.0f / sqrtf((float)D);

    // Phase 1: Compute Q[row] @ K^T (= Q[row] dot K[j] for each j) / sqrt(d)
    //          and find max for numerical stability
    float max_val = -1e30f;
    for (uint32_t j = 0; j < N; ++j) {
        float dot = 0.0f;
        for (uint32_t d = 0; d < D; ++d) {
            dot += q_ptr[row * D + d] * k_ptr[j * D + d];
        }
        dot *= inv_sqrt_d;
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
