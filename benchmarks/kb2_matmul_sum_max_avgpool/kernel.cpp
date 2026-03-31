#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// Fused: Linear -> sum(dim=1) -> max(dim=1) -> mean(dim=1) -> logsumexp -> logsumexp
// After sum(dim=1, keepdim=True), result is [M,1]. Subsequent max/mean/logsumexp
// on dim=1 of a [M,1] tensor are identity ops (max of 1 element = itself, etc).
// So the whole thing reduces to: y[row] = sum_j( A[row] @ W[:,j] + bias[j] )
// One task per row.

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto a_ptr    = reinterpret_cast<float*>(arg->a_addr);
    auto w_ptr    = reinterpret_cast<float*>(arg->w_addr);
    auto bias_ptr = reinterpret_cast<float*>(arg->bias_addr);
    auto c_ptr    = reinterpret_cast<float*>(arg->c_addr);

    uint32_t row = blockIdx.x;
    uint32_t N   = arg->N;
    uint32_t K   = arg->K;

    // Compute matmul + bias for each col, then sum across cols
    float total = 0.0f;
    for (uint32_t j = 0; j < N; ++j) {
        float dot = bias_ptr[j];
        for (uint32_t k = 0; k < K; ++k) {
            dot += a_ptr[row * K + k] * w_ptr[k * N + j];
        }
        total += dot;
    }

    // After sum, shape is [M,1]. max/mean/logsumexp on dim=1 of size-1 are identity.
    // logsumexp of single element x = log(exp(x)) = x
    c_ptr[row] = total;
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->M, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
