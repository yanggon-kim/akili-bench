#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// Per-ray inclusive prefix sum (weighted accumulation).
// output[i] = input[0] + input[1] + ... + input[i]
// One task per ray.

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto input  = reinterpret_cast<float*>(arg->input_addr);
    auto output = reinterpret_cast<float*>(arg->output_addr);

    uint32_t ray = blockIdx.x;
    uint32_t samples = arg->samples_per_ray;
    uint32_t offset = ray * samples;

    float accum = 0.0f;
    for (uint32_t i = 0; i < samples; ++i) {
        accum += input[offset + i];
        output[offset + i] = accum;
    }
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->num_tasks, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
