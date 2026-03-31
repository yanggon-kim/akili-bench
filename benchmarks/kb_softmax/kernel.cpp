#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// Softmax: y[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))
// Per-row softmax. One task per row.

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto x_ptr = reinterpret_cast<float*>(arg->x_addr);
    auto y_ptr = reinterpret_cast<float*>(arg->y_addr);

    uint32_t row = blockIdx.x;
    uint32_t num_cols = arg->num_cols;
    uint32_t offset = row * num_cols;

    // 1. Find row max for numerical stability
    float max_val = x_ptr[offset];
    for (uint32_t c = 1; c < num_cols; ++c) {
        float v = x_ptr[offset + c];
        if (v > max_val) max_val = v;
    }

    // 2. Compute exp(x - max) and sum
    float sum_exp = 0.0f;
    for (uint32_t c = 0; c < num_cols; ++c) {
        float e = expf(x_ptr[offset + c] - max_val);
        y_ptr[offset + c] = e;
        sum_exp += e;
    }

    // 3. Normalize
    float inv_sum = 1.0f / sum_exp;
    for (uint32_t c = 0; c < num_cols; ++c) {
        y_ptr[offset + c] *= inv_sum;
    }
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->num_rows, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
