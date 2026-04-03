#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// Volume rendering compositing forward pass.
// One task per ray. Fixed samples_per_ray layout.
// For each sample: alpha = 1 - exp(-sigma * delta), weight = alpha * T
// Accumulate color, depth, and weights_sum. Early terminate when T < T_thresh.

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto sigmas      = reinterpret_cast<float*>(arg->sigmas_addr);
    auto rgbs        = reinterpret_cast<float*>(arg->rgbs_addr);
    auto deltas      = reinterpret_cast<float*>(arg->deltas_addr);
    auto weights_sum = reinterpret_cast<float*>(arg->weights_sum_addr);
    auto depth       = reinterpret_cast<float*>(arg->depth_addr);
    auto image       = reinterpret_cast<float*>(arg->image_addr);

    uint32_t ray = blockIdx.x;
    uint32_t S = arg->samples_per_ray;
    float T_thresh = arg->T_thresh;

    uint32_t base = ray * S;
    uint32_t base3 = ray * S * 3;

    float T = 1.0f;
    float r = 0.0f, g = 0.0f, b = 0.0f;
    float ws = 0.0f;
    float t_accum = 0.0f;
    float dp = 0.0f;

    for (uint32_t step = 0; step < S; step++) {
        float sigma = sigmas[base + step];
        float delta = deltas[base + step];
        float alpha = 1.0f - expf(-sigma * delta);
        float weight = alpha * T;

        r += weight * rgbs[base3 + step * 3 + 0];
        g += weight * rgbs[base3 + step * 3 + 1];
        b += weight * rgbs[base3 + step * 3 + 2];

        t_accum += delta;
        dp += weight * t_accum;
        ws += weight;

        T *= (1.0f - alpha);
        if (T < T_thresh) break;
    }

    weights_sum[ray] = ws;
    depth[ray] = dp;
    image[ray * 3 + 0] = r;
    image[ray * 3 + 1] = g;
    image[ray * 3 + 2] = b;
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->n_rays, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
