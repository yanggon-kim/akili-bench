#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// Morton encoding helpers
static inline uint32_t expand_bits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

static inline uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z) {
    return expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2);
}

static inline float my_clamp(float x, float mn, float mx) {
    return fminf(mx, fmaxf(mn, x));
}

// Simplified ray marching: single cascade (C=1), dt_gamma=0 (constant step),
// no atomics — each ray gets a pre-allocated output slot of max_steps entries.
// One task per ray.

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto rays_o  = reinterpret_cast<float*>(arg->rays_o_addr);
    auto rays_d  = reinterpret_cast<float*>(arg->rays_d_addr);
    auto nears   = reinterpret_cast<float*>(arg->nears_addr);
    auto fars    = reinterpret_cast<float*>(arg->fars_addr);
    auto grid    = reinterpret_cast<uint8_t*>(arg->grid_addr);
    auto xyzs    = reinterpret_cast<float*>(arg->xyzs_addr);
    auto deltas  = reinterpret_cast<float*>(arg->deltas_addr);
    auto counts  = reinterpret_cast<uint32_t*>(arg->counts_addr);

    uint32_t ray = blockIdx.x;
    uint32_t H = arg->grid_res;
    uint32_t max_steps = arg->max_steps;
    float bound = arg->bound;
    float dt = arg->dt_min;  // constant step size
    float rH = 1.0f / (float)H;

    float ox = rays_o[ray * 3 + 0];
    float oy = rays_o[ray * 3 + 1];
    float oz = rays_o[ray * 3 + 2];
    float dx = rays_d[ray * 3 + 0];
    float dy = rays_d[ray * 3 + 1];
    float dz = rays_d[ray * 3 + 2];
    float near = nears[ray];
    float far  = fars[ray];

    // Output slot base for this ray
    float* out_xyzs   = xyzs   + ray * max_steps * 3;
    float* out_deltas = deltas + ray * max_steps;

    float t = near;
    uint32_t num_steps = 0;

    while (t < far && num_steps < max_steps) {
        float x = my_clamp(ox + t * dx, -bound, bound);
        float y = my_clamp(oy + t * dy, -bound, bound);
        float z = my_clamp(oz + t * dz, -bound, bound);

        // Map to grid coordinates [0, H-1]
        int nx = (int)my_clamp(0.5f * (x / bound + 1.0f) * H, 0.0f, (float)(H - 1));
        int ny = (int)my_clamp(0.5f * (y / bound + 1.0f) * H, 0.0f, (float)(H - 1));
        int nz = (int)my_clamp(0.5f * (z / bound + 1.0f) * H, 0.0f, (float)(H - 1));

        uint32_t idx = morton3D((uint32_t)nx, (uint32_t)ny, (uint32_t)nz);
        uint8_t occ = grid[idx / 8] & (1 << (idx % 8));

        if (occ) {
            out_xyzs[num_steps * 3 + 0] = x;
            out_xyzs[num_steps * 3 + 1] = y;
            out_xyzs[num_steps * 3 + 2] = z;
            out_deltas[num_steps] = dt;
            num_steps++;
        }

        t += dt;
    }

    counts[ray] = num_steps;
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->n_rays, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
