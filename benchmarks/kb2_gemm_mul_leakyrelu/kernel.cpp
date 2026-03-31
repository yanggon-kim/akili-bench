#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// Fused Gemm + Multiply + LeakyReLU
// y = LeakyReLU( (x @ W + bias) * multiplier )
// One task per output row.

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto a_ptr    = reinterpret_cast<float*>(arg->a_addr);
    auto w_ptr    = reinterpret_cast<float*>(arg->w_addr);
    auto bias_ptr = reinterpret_cast<float*>(arg->bias_addr);
    auto c_ptr    = reinterpret_cast<float*>(arg->c_addr);

    uint32_t row = blockIdx.x;
    uint32_t N   = arg->N;
    uint32_t K   = arg->K;
    float multiplier     = arg->multiplier;
    float negative_slope = arg->negative_slope;

    for (uint32_t j = 0; j < N; ++j) {
        // Gemm: dot(A[row,:], W[:,j]) + bias[j]
        float sum = bias_ptr[j];
        for (uint32_t k = 0; k < K; ++k) {
            sum += a_ptr[row * K + k] * w_ptr[k * N + j];
        }
        // Multiply
        sum *= multiplier;
        // LeakyReLU
        c_ptr[row * N + j] = (sum >= 0.0f) ? sum : sum * negative_slope;
    }
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->M, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
