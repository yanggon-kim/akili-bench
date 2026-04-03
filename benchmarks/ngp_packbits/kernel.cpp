#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// Packbits: one task per group of 8 density values.
// Packs 8 float density values into 1 uint8_t bitfield.
// Bit i = (density[i] > threshold) ? 1 : 0

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto density  = reinterpret_cast<float*>(arg->density_addr);
    auto bitfield = reinterpret_cast<uint8_t*>(arg->bitfield_addr);

    uint32_t gid = blockIdx.x;
    float threshold = arg->threshold;

    float* g = density + gid * 8;
    uint8_t bits = 0;
    for (uint32_t i = 0; i < 8; ++i) {
        if (g[i] > threshold)
            bits |= (uint8_t)(1 << i);
    }
    bitfield[gid] = bits;
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->num_tasks, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
