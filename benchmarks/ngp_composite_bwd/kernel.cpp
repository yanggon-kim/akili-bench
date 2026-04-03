#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// Volume rendering compositing backward pass.
// One task per ray. Fixed samples_per_ray layout.
// Computes grad_sigmas and grad_rgbs given forward results and upstream gradients.

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto sigmas          = reinterpret_cast<float*>(arg->sigmas_addr);
    auto rgbs            = reinterpret_cast<float*>(arg->rgbs_addr);
    auto deltas          = reinterpret_cast<float*>(arg->deltas_addr);
    auto weights_sum     = reinterpret_cast<float*>(arg->weights_sum_addr);
    auto image           = reinterpret_cast<float*>(arg->image_addr);
    auto grad_weights_sum = reinterpret_cast<float*>(arg->grad_weights_sum_addr);
    auto grad_image      = reinterpret_cast<float*>(arg->grad_image_addr);
    auto grad_sigmas     = reinterpret_cast<float*>(arg->grad_sigmas_addr);
    auto grad_rgbs       = reinterpret_cast<float*>(arg->grad_rgbs_addr);

    uint32_t ray = blockIdx.x;
    uint32_t S = arg->samples_per_ray;
    float T_thresh = arg->T_thresh;

    uint32_t base = ray * S;
    uint32_t base3 = ray * S * 3;

    float gi0 = grad_image[ray * 3 + 0];
    float gi1 = grad_image[ray * 3 + 1];
    float gi2 = grad_image[ray * 3 + 2];
    float gws = grad_weights_sum[ray];

    float img_r = image[ray * 3 + 0];
    float img_g = image[ray * 3 + 1];
    float img_b = image[ray * 3 + 2];
    float ws_f  = weights_sum[ray];

    float T = 1.0f;
    float accum_r = 0.0f, accum_g = 0.0f, accum_b = 0.0f;
    float accum_ws = 0.0f;

    for (uint32_t step = 0; step < S; step++) {
        float sigma = sigmas[base + step];
        float delta = deltas[base + step];
        float alpha = 1.0f - expf(-sigma * delta);
        float weight = alpha * T;

        float c0 = rgbs[base3 + step * 3 + 0];
        float c1 = rgbs[base3 + step * 3 + 1];
        float c2 = rgbs[base3 + step * 3 + 2];

        accum_r += weight * c0;
        accum_g += weight * c1;
        accum_b += weight * c2;
        accum_ws += weight;

        // grad_rgb = grad_image * weight
        grad_rgbs[base3 + step * 3 + 0] = gi0 * weight;
        grad_rgbs[base3 + step * 3 + 1] = gi1 * weight;
        grad_rgbs[base3 + step * 3 + 2] = gi2 * weight;

        // grad_sigma = delta * (sum of gradient contributions)
        grad_sigmas[base + step] = delta * (
            gi0 * (T * c0 - (img_r - accum_r)) +
            gi1 * (T * c1 - (img_g - accum_g)) +
            gi2 * (T * c2 - (img_b - accum_b)) +
            gws * (1.0f - ws_f)
        );

        T *= (1.0f - alpha);
        if (T < T_thresh) break;
    }
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->n_rays, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
