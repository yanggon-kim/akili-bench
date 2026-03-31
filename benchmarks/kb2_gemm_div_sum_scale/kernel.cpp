#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// Fused Gemm + Divide + Sum + Scale
// y[row] = sum_j( (x[row] @ W[:,j]) / 2 ) * scaling_factor
// One task per output row. Output is a scalar per row.

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto a_ptr = reinterpret_cast<float*>(arg->a_addr);
    auto w_ptr = reinterpret_cast<float*>(arg->w_addr);
    auto c_ptr = reinterpret_cast<float*>(arg->c_addr);

    uint32_t row = blockIdx.x;
    uint32_t N   = arg->N;
    uint32_t K   = arg->K;
    float scaling_factor = arg->scaling_factor;

    // Compute matmul row, divide by 2, accumulate sum
    float total = 0.0f;
    for (uint32_t j = 0; j < N; ++j) {
        float dot = 0.0f;
        for (uint32_t k = 0; k < K; ++k) {
            dot += a_ptr[row * K + k] * w_ptr[k * N + j];
        }
        total += dot * 0.5f;  // divide by 2
    }

    c_ptr[row] = total * scaling_factor;
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->M, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
