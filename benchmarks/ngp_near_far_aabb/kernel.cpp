#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// Ray-AABB near/far intersection (torch-ngp style with min_near clamping).
// One task per ray. Computes slab intersection for x/y/z axes.

static inline float fmaxf_v(float a, float b) { return a > b ? a : b; }
static inline float fminf_v(float a, float b) { return a < b ? a : b; }

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto rays_o = reinterpret_cast<float*>(arg->rays_o_addr);
    auto rays_d = reinterpret_cast<float*>(arg->rays_d_addr);
    auto nears  = reinterpret_cast<float*>(arg->nears_addr);
    auto fars   = reinterpret_cast<float*>(arg->fars_addr);

    uint32_t n = blockIdx.x;
    float min_near = arg->min_near;

    float ox = rays_o[n * 3 + 0];
    float oy = rays_o[n * 3 + 1];
    float oz = rays_o[n * 3 + 2];
    float dx = rays_d[n * 3 + 0];
    float dy = rays_d[n * 3 + 1];
    float dz = rays_d[n * 3 + 2];

    float rdx = 1.0f / dx;
    float rdy = 1.0f / dy;
    float rdz = 1.0f / dz;

    // X slab
    float near = ((dx > 0.0f ? arg->aabb0 : arg->aabb3) - ox) * rdx;
    float far  = ((dx > 0.0f ? arg->aabb3 : arg->aabb0) - ox) * rdx;

    // Y slab
    near = fmaxf_v(near, ((dy > 0.0f ? arg->aabb1 : arg->aabb4) - oy) * rdy);
    far  = fminf_v(far,  ((dy > 0.0f ? arg->aabb4 : arg->aabb1) - oy) * rdy);

    // Z slab
    near = fmaxf_v(near, ((dz > 0.0f ? arg->aabb2 : arg->aabb5) - oz) * rdz);
    far  = fminf_v(far,  ((dz > 0.0f ? arg->aabb5 : arg->aabb2) - oz) * rdz);

    // torch-ngp style: clamp near to min_near, set to min_near if miss
    nears[n] = (near < far) ? fmaxf_v(near, min_near) : min_near;
    fars[n]  = (near < far) ? far : min_near;
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->num_tasks, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
