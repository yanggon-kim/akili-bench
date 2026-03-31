#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// Fused SwiGLU: y = silu(x @ Wa + ba) * (x @ Wb + bb)
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// One task per output row.

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto a_ptr  = reinterpret_cast<float*>(arg->a_addr);
    auto wa_ptr = reinterpret_cast<float*>(arg->wa_addr);
    auto ba_ptr = reinterpret_cast<float*>(arg->ba_addr);
    auto wb_ptr = reinterpret_cast<float*>(arg->wb_addr);
    auto bb_ptr = reinterpret_cast<float*>(arg->bb_addr);
    auto c_ptr  = reinterpret_cast<float*>(arg->c_addr);

    uint32_t row = blockIdx.x;
    uint32_t N   = arg->N;
    uint32_t K   = arg->K;

    for (uint32_t j = 0; j < N; ++j) {
        // Gate path: x @ Wa + ba
        float gate = ba_ptr[j];
        for (uint32_t k = 0; k < K; ++k) {
            gate += a_ptr[row * K + k] * wa_ptr[k * N + j];
        }
        // Up path: x @ Wb + bb
        float up = bb_ptr[j];
        for (uint32_t k = 0; k < K; ++k) {
            up += a_ptr[row * K + k] * wb_ptr[k * N + j];
        }
        // SiLU(gate) * up
        float silu_gate = gate / (1.0f + expf(-gate));
        c_ptr[row * N + j] = silu_gate * up;
    }
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->M, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
