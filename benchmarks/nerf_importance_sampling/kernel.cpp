#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// Importance sampling via CDF inversion: one task per ray.
// For each output sample s: u = (s+0.5)/n_new, binary search CDF for u,
// output the found index as a float position.

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto cdfs = reinterpret_cast<float*>(arg->cdfs_addr);
    auto out  = reinterpret_cast<float*>(arg->out_addr);

    uint32_t ray = blockIdx.x;
    uint32_t n_cdf = arg->n_cdf;
    uint32_t n_new = arg->n_new;

    float* cdf = cdfs + ray * n_cdf;
    float* dst = out  + ray * n_new;

    for (uint32_t s = 0; s < n_new; ++s) {
        float u = ((float)s + 0.5f) / (float)n_new;

        // Binary search: find smallest index where cdf[index] >= u
        uint32_t lo = 0, hi = n_cdf;
        while (lo < hi) {
            uint32_t mid = (lo + hi) / 2;
            if (cdf[mid] < u)
                lo = mid + 1;
            else
                hi = mid;
        }
        if (lo >= n_cdf)
            lo = n_cdf - 1;

        dst[s] = (float)lo;
    }
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->num_tasks, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
