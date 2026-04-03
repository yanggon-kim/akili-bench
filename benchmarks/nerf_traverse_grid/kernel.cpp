#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// DDA-style ray marching through a 3D boolean occupancy grid.
// One task per ray. March from near to far in step_size increments,
// check occupancy grid, store hits.

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto rays_o   = reinterpret_cast<float*>(arg->rays_o_addr);
    auto rays_d   = reinterpret_cast<float*>(arg->rays_d_addr);
    auto nears    = reinterpret_cast<float*>(arg->nears_addr);
    auto fars     = reinterpret_cast<float*>(arg->fars_addr);
    auto grid     = reinterpret_cast<uint8_t*>(arg->grid_addr);
    auto t_starts = reinterpret_cast<float*>(arg->t_starts_addr);
    auto t_ends   = reinterpret_cast<float*>(arg->t_ends_addr);
    auto counts   = reinterpret_cast<uint32_t*>(arg->counts_addr);

    uint32_t tid = blockIdx.x;
    uint32_t res = arg->grid_res;
    uint32_t max_samples = arg->max_samples;
    float step_size = arg->step_size;

    float ox = rays_o[tid * 3 + 0];
    float oy = rays_o[tid * 3 + 1];
    float oz = rays_o[tid * 3 + 2];
    float dx = rays_d[tid * 3 + 0];
    float dy = rays_d[tid * 3 + 1];
    float dz = rays_d[tid * 3 + 2];
    float near = nears[tid];
    float far  = fars[tid];

    float aabb_min_x = arg->aabb0;
    float aabb_min_y = arg->aabb1;
    float aabb_min_z = arg->aabb2;
    float aabb_size_x = arg->aabb3 - arg->aabb0;
    float aabb_size_y = arg->aabb4 - arg->aabb1;
    float aabb_size_z = arg->aabb5 - arg->aabb2;

    uint32_t offset = tid * max_samples;
    float t = near;
    uint32_t count = 0;

    while (t < far && count < max_samples) {
        float x = ox + t * dx;
        float y = oy + t * dy;
        float z = oz + t * dz;

        // Grid coordinates
        int gx = (int)((x - aabb_min_x) / aabb_size_x * (float)res);
        int gy = (int)((y - aabb_min_y) / aabb_size_y * (float)res);
        int gz = (int)((z - aabb_min_z) / aabb_size_z * (float)res);

        // Clamp to [0, res-1]
        if (gx < 0) gx = 0;
        if (gx >= (int)res) gx = (int)res - 1;
        if (gy < 0) gy = 0;
        if (gy >= (int)res) gy = (int)res - 1;
        if (gz < 0) gz = 0;
        if (gz >= (int)res) gz = (int)res - 1;

        int idx = gx * (int)(res * res) + gy * (int)res + gz;

        if (grid[idx]) {
            t_starts[offset + count] = t;
            t_ends[offset + count]   = t + step_size;
            count++;
        }
        t += step_size;
    }

    counts[tid] = count;
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->num_tasks, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
