#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// RoPE (Rotary Position Embedding) Forward
// Matches Liger/Unsloth Triton RoPE kernels.
// For each (token, head), apply rotation to pairs of elements:
//   y[2i]   = x[2i] * cos(theta) - x[2i+1] * sin(theta)
//   y[2i+1] = x[2i] * sin(theta) + x[2i+1] * cos(theta)
// where theta = position * base^(-2i/d)
//
// One task per token. Each task processes all heads × head_dim.

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto x_ptr = reinterpret_cast<float*>(arg->x_addr);
    auto y_ptr = reinterpret_cast<float*>(arg->y_addr);

    uint32_t token = blockIdx.x;
    uint32_t head_dim = arg->head_dim;
    uint32_t num_heads = arg->num_heads;
    float theta_base = arg->theta_base;
    uint32_t half_dim = head_dim / 2;
    uint32_t stride = num_heads * head_dim;

    // position = token index (simplified; real impl uses position_ids)
    float position = (float)token;

    for (uint32_t h = 0; h < num_heads; ++h) {
        uint32_t head_offset = token * stride + h * head_dim;

        for (uint32_t i = 0; i < half_dim; ++i) {
            // theta = position * base^(-2i/head_dim)
            float freq = 1.0f / powf(theta_base, (2.0f * i) / head_dim);
            float theta = position * freq;
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);

            float x0 = x_ptr[head_offset + i];
            float x1 = x_ptr[head_offset + i + half_dim];

            y_ptr[head_offset + i]            = x0 * cos_t - x1 * sin_t;
            y_ptr[head_offset + i + half_dim] = x0 * sin_t + x1 * cos_t;
        }
    }
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->num_tokens, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
