#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// Fused Gemm + LayerNorm
// h = x @ W + bias        (matmul)
// y = LayerNorm(h, ln_w, ln_b)
// One task per output row.

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto a_ptr    = reinterpret_cast<float*>(arg->a_addr);
    auto w_ptr    = reinterpret_cast<float*>(arg->w_addr);
    auto bias_ptr = reinterpret_cast<float*>(arg->bias_addr);
    auto ln_w_ptr = reinterpret_cast<float*>(arg->ln_w_addr);
    auto ln_b_ptr = reinterpret_cast<float*>(arg->ln_b_addr);
    auto c_ptr    = reinterpret_cast<float*>(arg->c_addr);

    uint32_t row = blockIdx.x;
    uint32_t N   = arg->N;
    uint32_t K   = arg->K;
    float eps    = arg->eps;

    // Phase 1: Compute matmul row and store in output, also accumulate mean
    float sum = 0.0f;
    for (uint32_t j = 0; j < N; ++j) {
        float dot = bias_ptr[j];
        for (uint32_t k = 0; k < K; ++k) {
            dot += a_ptr[row * K + k] * w_ptr[k * N + j];
        }
        c_ptr[row * N + j] = dot;
        sum += dot;
    }
    float mean = sum / (float)N;

    // Phase 2: Compute variance
    float var = 0.0f;
    for (uint32_t j = 0; j < N; ++j) {
        float diff = c_ptr[row * N + j] - mean;
        var += diff * diff;
    }
    var = var / (float)N;

    // Phase 3: Normalize and apply affine transform
    float inv_std = 1.0f / sqrtf(var + eps);
    for (uint32_t j = 0; j < N; ++j) {
        float normalized = (c_ptr[row * N + j] - mean) * inv_std;
        c_ptr[row * N + j] = normalized * ln_w_ptr[j] + ln_b_ptr[j];
    }
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->M, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
