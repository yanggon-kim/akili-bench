#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// Fused: Linear -> Scale -> ResidualAdd(x+x=2x) -> Clamp -> LogSumExp -> Mish
// Output is [M x 1]: one scalar per row after logsumexp reduction.
// One task per output row.

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto a_ptr    = reinterpret_cast<float*>(arg->a_addr);
    auto w_ptr    = reinterpret_cast<float*>(arg->w_addr);
    auto bias_ptr = reinterpret_cast<float*>(arg->bias_addr);
    auto c_ptr    = reinterpret_cast<float*>(arg->c_addr);

    uint32_t row = blockIdx.x;
    uint32_t N   = arg->N;
    uint32_t K   = arg->K;
    float scale_factor = arg->scale_factor;
    float clamp_min    = arg->clamp_min;
    float clamp_max    = arg->clamp_max;

    // Phase 1: Compute matmul row, scale, residual add (x+x=2x), clamp
    // Phase 2: LogSumExp across cols
    // Find max for numerical stability
    float max_val = -1e30f;
    for (uint32_t j = 0; j < N; ++j) {
        float dot = bias_ptr[j];
        for (uint32_t k = 0; k < K; ++k) {
            dot += a_ptr[row * K + k] * w_ptr[k * N + j];
        }
        dot *= scale_factor;
        dot += dot;  // residual add: x + x = 2x
        // clamp
        if (dot < clamp_min) dot = clamp_min;
        if (dot > clamp_max) dot = clamp_max;
        if (dot > max_val) max_val = dot;
    }

    // Compute sum of exp(x - max)
    float sum_exp = 0.0f;
    for (uint32_t j = 0; j < N; ++j) {
        float dot = bias_ptr[j];
        for (uint32_t k = 0; k < K; ++k) {
            dot += a_ptr[row * K + k] * w_ptr[k * N + j];
        }
        dot *= scale_factor;
        dot += dot;
        if (dot < clamp_min) dot = clamp_min;
        if (dot > clamp_max) dot = clamp_max;
        sum_exp += expf(dot - max_val);
    }

    float lse = max_val + logf(sum_exp);

    // Final: x * mish(x) where mish(x) = x * tanh(softplus(x))
    // So result = lse * lse * tanh(log(1 + exp(lse)))
    float mish_lse = lse * tanhf(logf(1.0f + expf(lse)));
    c_ptr[row] = lse * mish_lse;
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->M, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
