#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// Transformer MLP block: RMSNorm(x) -> SwiGLU(W_gate, W_up)
// h = RMSNorm(x, rms_w)           -- normalized input [K]
// gate = h @ W_gate                -- [N]
// up   = h @ W_up                  -- [N]
// y    = silu(gate) * up           -- [N]
// One task per output row.

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto x_ptr     = reinterpret_cast<float*>(arg->x_addr);
    auto rms_w_ptr = reinterpret_cast<float*>(arg->rms_w_addr);
    auto wg_ptr    = reinterpret_cast<float*>(arg->wg_addr);
    auto wu_ptr    = reinterpret_cast<float*>(arg->wu_addr);
    auto c_ptr     = reinterpret_cast<float*>(arg->c_addr);

    uint32_t row = blockIdx.x;
    uint32_t N   = arg->N;
    uint32_t K   = arg->K;
    float eps    = arg->eps;

    // Phase 1: RMSNorm -- compute RMS of input row
    float ss = 0.0f;
    for (uint32_t k = 0; k < K; ++k) {
        float v = x_ptr[row * K + k];
        ss += v * v;
    }
    float inv_rms = 1.0f / sqrtf(ss / (float)K + eps);

    // Phase 2: SwiGLU with normalized input
    // For each output col j:
    //   gate = sum_k( (x[k]*inv_rms*rms_w[k]) * Wg[k,j] )
    //   up   = sum_k( (x[k]*inv_rms*rms_w[k]) * Wu[k,j] )
    //   out  = silu(gate) * up
    for (uint32_t j = 0; j < N; ++j) {
        float gate = 0.0f;
        float up   = 0.0f;
        for (uint32_t k = 0; k < K; ++k) {
            float h = x_ptr[row * K + k] * inv_rms * rms_w_ptr[k];
            gate += h * wg_ptr[k * N + j];
            up   += h * wu_ptr[k * N + j];
        }
        float silu_gate = gate / (1.0f + expf(-gate));
        c_ptr[row * N + j] = silu_gate * up;
    }
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->M, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
