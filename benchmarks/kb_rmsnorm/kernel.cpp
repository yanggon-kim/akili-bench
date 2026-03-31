#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// RMSNorm: y = x * rsqrt(mean(x^2) + eps) * w
// Per-row normalization. One task per row.

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto x_ptr = reinterpret_cast<float*>(arg->x_addr);
    auto w_ptr = reinterpret_cast<float*>(arg->w_addr);
    auto y_ptr = reinterpret_cast<float*>(arg->y_addr);

    uint32_t row = blockIdx.x;
    uint32_t num_cols = arg->num_cols;
    uint32_t offset = row * num_cols;
    float eps = arg->eps;

    // 1. Compute sum of squares
    float sum_sq = 0.0f;
    for (uint32_t c = 0; c < num_cols; ++c) {
        float val = x_ptr[offset + c];
        sum_sq += val * val;
    }

    // 2. Compute inverse RMS
    float rstd = 1.0f / sqrtf(sum_sq / (float)num_cols + eps);

    // 3. Normalize and apply weight
    for (uint32_t c = 0; c < num_cols; ++c) {
        float val = x_ptr[offset + c];
        y_ptr[offset + c] = val * rstd * w_ptr[c];
    }
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->num_rows, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
