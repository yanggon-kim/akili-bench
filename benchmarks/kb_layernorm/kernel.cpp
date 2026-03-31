#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// LayerNorm: y = (x - mean) / sqrt(var + eps) * w + b
// Per-row normalization. One task per row.

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto x_ptr = reinterpret_cast<float*>(arg->x_addr);
    auto w_ptr = reinterpret_cast<float*>(arg->w_addr);
    auto b_ptr = reinterpret_cast<float*>(arg->b_addr);
    auto y_ptr = reinterpret_cast<float*>(arg->y_addr);

    uint32_t row = blockIdx.x;
    uint32_t num_cols = arg->num_cols;
    uint32_t offset = row * num_cols;
    float eps = arg->eps;

    // 1. Compute mean
    float sum = 0.0f;
    for (uint32_t c = 0; c < num_cols; ++c) {
        sum += x_ptr[offset + c];
    }
    float mean = sum / (float)num_cols;

    // 2. Compute variance
    float var = 0.0f;
    for (uint32_t c = 0; c < num_cols; ++c) {
        float diff = x_ptr[offset + c] - mean;
        var += diff * diff;
    }
    var = var / (float)num_cols;

    // 3. Normalize and apply affine transform
    float inv_std = 1.0f / sqrtf(var + eps);
    for (uint32_t c = 0; c < num_cols; ++c) {
        float normalized = (x_ptr[offset + c] - mean) * inv_std;
        y_ptr[offset + c] = normalized * w_ptr[c] + b_ptr[c];
    }
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->num_rows, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
