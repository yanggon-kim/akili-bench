#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// Ray-AABB intersection: one task per ray.
// Computes tmin, tmax, and whether the ray hits the AABB.

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto rays_o = reinterpret_cast<float*>(arg->rays_o_addr);
    auto rays_d = reinterpret_cast<float*>(arg->rays_d_addr);
    auto aabb   = reinterpret_cast<float*>(arg->aabb_addr);
    auto t_mins = reinterpret_cast<float*>(arg->t_mins_addr);
    auto t_maxs = reinterpret_cast<float*>(arg->t_maxs_addr);
    auto hits   = reinterpret_cast<uint32_t*>(arg->hits_addr);

    uint32_t tid = blockIdx.x;
    float min_near = arg->min_near;

    float ox = rays_o[tid * 3 + 0];
    float oy = rays_o[tid * 3 + 1];
    float oz = rays_o[tid * 3 + 2];
    float dx = rays_d[tid * 3 + 0];
    float dy = rays_d[tid * 3 + 1];
    float dz = rays_d[tid * 3 + 2];

    float idx = 1.0f / dx;
    float idy = 1.0f / dy;
    float idz = 1.0f / dz;

    // X slab
    float tmin = (aabb[idx > 0.0f ? 0 : 3] - ox) * idx;
    float tmax = (aabb[idx > 0.0f ? 3 : 0] - ox) * idx;

    // Y slab
    float tymin = (aabb[idy > 0.0f ? 1 : 4] - oy) * idy;
    float tymax = (aabb[idy > 0.0f ? 4 : 1] - oy) * idy;

    if (tmin > tymax || tymin > tmax) {
        t_mins[tid] = 0.0f;
        t_maxs[tid] = 0.0f;
        hits[tid] = 0;
        return;
    }
    if (tymin > tmin) tmin = tymin;
    if (tymax < tmax) tmax = tymax;

    // Z slab
    float tzmin = (aabb[idz > 0.0f ? 2 : 5] - oz) * idz;
    float tzmax = (aabb[idz > 0.0f ? 5 : 2] - oz) * idz;

    if (tmin > tzmax || tzmin > tmax) {
        t_mins[tid] = 0.0f;
        t_maxs[tid] = 0.0f;
        hits[tid] = 0;
        return;
    }
    if (tzmin > tmin) tmin = tzmin;
    if (min_near > tmin) tmin = min_near;
    if (tzmax < tmax) tmax = tzmax;

    t_mins[tid] = tmin;
    t_maxs[tid] = tmax;
    hits[tid] = (tmin <= tmax) ? 1 : 0;
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->num_tasks, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
