#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// GELU: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// One task per row.

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto x_ptr = reinterpret_cast<float*>(arg->x_addr);
    auto y_ptr = reinterpret_cast<float*>(arg->y_addr);

    uint32_t row = blockIdx.x;
    uint32_t num_cols = arg->num_cols;
    uint32_t offset = row * num_cols;

    const float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/pi)
    const float coeff = 0.044715f;

    for (uint32_t c = 0; c < num_cols; ++c) {
        float val = x_ptr[offset + c];
        float x3 = val * val * val;
        float inner = sqrt_2_over_pi * (val + coeff * x3);
        y_ptr[offset + c] = 0.5f * val * (1.0f + tanhf(inner));
    }
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->num_rows, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
