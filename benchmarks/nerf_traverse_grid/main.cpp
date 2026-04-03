#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <cmath>
#include <vortex.h>
#include "common.h"

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

const char* kernel_file = "kernel.vxbin";
uint32_t n_rays = 16;
uint32_t grid_res = 8;
uint32_t max_samples = 16;

vx_device_h device = nullptr;
vx_buffer_h rays_o_buffer = nullptr;
vx_buffer_h rays_d_buffer = nullptr;
vx_buffer_h nears_buffer = nullptr;
vx_buffer_h fars_buffer = nullptr;
vx_buffer_h grid_buffer = nullptr;
vx_buffer_h t_starts_buffer = nullptr;
vx_buffer_h t_ends_buffer = nullptr;
vx_buffer_h counts_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex NeRF Traverse Grid Benchmark." << std::endl;
   std::cout << "Usage: [-k: kernel] [-r rays] [-g grid_res] [-s max_samples] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "r:g:s:k:h")) != -1) {
    switch (c) {
    case 'r': n_rays = atoi(optarg); break;
    case 'g': grid_res = atoi(optarg); break;
    case 's': max_samples = atoi(optarg); break;
    case 'k': kernel_file = optarg; break;
    case 'h': show_usage(); exit(0); break;
    default: show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(rays_o_buffer);
    vx_mem_free(rays_d_buffer);
    vx_mem_free(nears_buffer);
    vx_mem_free(fars_buffer);
    vx_mem_free(grid_buffer);
    vx_mem_free(t_starts_buffer);
    vx_mem_free(t_ends_buffer);
    vx_mem_free(counts_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

// CPU reference: same grid traversal
static void cpu_traverse_grid(
    const float* rays_o, const float* rays_d,
    const float* nears, const float* fars,
    const uint8_t* grid, uint32_t res,
    float step_size,
    float aabb_min_x, float aabb_min_y, float aabb_min_z,
    float aabb_size_x, float aabb_size_y, float aabb_size_z,
    uint32_t n, uint32_t max_samp,
    float* t_starts, float* t_ends, uint32_t* counts) {

  for (uint32_t tid = 0; tid < n; ++tid) {
    float ox = rays_o[tid*3+0], oy = rays_o[tid*3+1], oz = rays_o[tid*3+2];
    float dx = rays_d[tid*3+0], dy = rays_d[tid*3+1], dz = rays_d[tid*3+2];
    float near = nears[tid], far = fars[tid];

    uint32_t offset = tid * max_samp;
    float t = near;
    uint32_t count = 0;

    while (t < far && count < max_samp) {
      float x = ox + t * dx;
      float y = oy + t * dy;
      float z = oz + t * dz;

      int gx = (int)((x - aabb_min_x) / aabb_size_x * (float)res);
      int gy = (int)((y - aabb_min_y) / aabb_size_y * (float)res);
      int gz = (int)((z - aabb_min_z) / aabb_size_z * (float)res);

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
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  std::srand(42);

  float step_size = 0.25f;
  float h_aabb[6] = {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};
  float aabb_size_x = h_aabb[3] - h_aabb[0];
  float aabb_size_y = h_aabb[4] - h_aabb[1];
  float aabb_size_z = h_aabb[5] - h_aabb[2];

  uint32_t grid_total = grid_res * grid_res * grid_res;
  uint32_t rays_o_size   = n_rays * 3 * sizeof(float);
  uint32_t rays_d_size   = n_rays * 3 * sizeof(float);
  uint32_t nears_size    = n_rays * sizeof(float);
  uint32_t fars_size     = n_rays * sizeof(float);
  uint32_t grid_size     = grid_total * sizeof(uint8_t);
  uint32_t t_starts_size = n_rays * max_samples * sizeof(float);
  uint32_t t_ends_size   = n_rays * max_samples * sizeof(float);
  uint32_t counts_size   = n_rays * sizeof(uint32_t);

  std::cout << "Traverse Grid: n_rays=" << n_rays
            << " grid_res=" << grid_res
            << " max_samples=" << max_samples << std::endl;

  // Generate rays
  std::vector<float> h_rays_o(n_rays * 3), h_rays_d(n_rays * 3);
  std::vector<float> h_nears(n_rays), h_fars(n_rays);
  for (uint32_t i = 0; i < n_rays * 3; ++i)
    h_rays_o[i] = 0.2f * static_cast<float>(rand()) / RAND_MAX - 0.1f;
  for (uint32_t i = 0; i < n_rays; ++i) {
    float dx = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    float dy = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    float dz = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    float len = sqrtf(dx*dx + dy*dy + dz*dz);
    if (len < 1e-6f) len = 1.0f;
    h_rays_d[i*3+0] = dx / len;
    h_rays_d[i*3+1] = dy / len;
    h_rays_d[i*3+2] = dz / len;
    h_nears[i] = 0.01f;
    h_fars[i]  = 2.0f;
  }

  // Generate random occupancy grid
  std::vector<uint8_t> h_grid(grid_total);
  for (uint32_t i = 0; i < grid_total; ++i)
    h_grid[i] = (rand() % 2) ? 1 : 0;

  RT_CHECK(vx_dev_open(&device));

  kernel_arg.num_tasks   = n_rays;
  kernel_arg.grid_res    = grid_res;
  kernel_arg.max_samples = max_samples;
  kernel_arg.step_size   = step_size;
  kernel_arg.aabb0 = h_aabb[0];
  kernel_arg.aabb1 = h_aabb[1];
  kernel_arg.aabb2 = h_aabb[2];
  kernel_arg.aabb3 = h_aabb[3];
  kernel_arg.aabb4 = h_aabb[4];
  kernel_arg.aabb5 = h_aabb[5];

  RT_CHECK(vx_mem_alloc(device, rays_o_size, VX_MEM_READ, &rays_o_buffer));
  RT_CHECK(vx_mem_address(rays_o_buffer, &kernel_arg.rays_o_addr));
  RT_CHECK(vx_mem_alloc(device, rays_d_size, VX_MEM_READ, &rays_d_buffer));
  RT_CHECK(vx_mem_address(rays_d_buffer, &kernel_arg.rays_d_addr));
  RT_CHECK(vx_mem_alloc(device, nears_size, VX_MEM_READ, &nears_buffer));
  RT_CHECK(vx_mem_address(nears_buffer, &kernel_arg.nears_addr));
  RT_CHECK(vx_mem_alloc(device, fars_size, VX_MEM_READ, &fars_buffer));
  RT_CHECK(vx_mem_address(fars_buffer, &kernel_arg.fars_addr));
  RT_CHECK(vx_mem_alloc(device, grid_size, VX_MEM_READ, &grid_buffer));
  RT_CHECK(vx_mem_address(grid_buffer, &kernel_arg.grid_addr));
  RT_CHECK(vx_mem_alloc(device, t_starts_size, VX_MEM_WRITE, &t_starts_buffer));
  RT_CHECK(vx_mem_address(t_starts_buffer, &kernel_arg.t_starts_addr));
  RT_CHECK(vx_mem_alloc(device, t_ends_size, VX_MEM_WRITE, &t_ends_buffer));
  RT_CHECK(vx_mem_address(t_ends_buffer, &kernel_arg.t_ends_addr));
  RT_CHECK(vx_mem_alloc(device, counts_size, VX_MEM_WRITE, &counts_buffer));
  RT_CHECK(vx_mem_address(counts_buffer, &kernel_arg.counts_addr));

  RT_CHECK(vx_copy_to_dev(rays_o_buffer, h_rays_o.data(), 0, rays_o_size));
  RT_CHECK(vx_copy_to_dev(rays_d_buffer, h_rays_d.data(), 0, rays_d_size));
  RT_CHECK(vx_copy_to_dev(nears_buffer, h_nears.data(), 0, nears_size));
  RT_CHECK(vx_copy_to_dev(fars_buffer, h_fars.data(), 0, fars_size));
  RT_CHECK(vx_copy_to_dev(grid_buffer, h_grid.data(), 0, grid_size));

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // Read back results
  std::vector<float> h_t_starts(n_rays * max_samples, 0);
  std::vector<float> h_t_ends(n_rays * max_samples, 0);
  std::vector<uint32_t> h_counts(n_rays, 0);
  RT_CHECK(vx_copy_from_dev(h_t_starts.data(), t_starts_buffer, 0, t_starts_size));
  RT_CHECK(vx_copy_from_dev(h_t_ends.data(), t_ends_buffer, 0, t_ends_size));
  RT_CHECK(vx_copy_from_dev(h_counts.data(), counts_buffer, 0, counts_size));

  // CPU reference
  std::vector<float> ref_t_starts(n_rays * max_samples, 0);
  std::vector<float> ref_t_ends(n_rays * max_samples, 0);
  std::vector<uint32_t> ref_counts(n_rays, 0);
  cpu_traverse_grid(h_rays_o.data(), h_rays_d.data(),
                    h_nears.data(), h_fars.data(),
                    h_grid.data(), grid_res,
                    step_size,
                    h_aabb[0], h_aabb[1], h_aabb[2],
                    aabb_size_x, aabb_size_y, aabb_size_z,
                    n_rays, max_samples,
                    ref_t_starts.data(), ref_t_ends.data(), ref_counts.data());

  // Verify: compare counts and first few t_starts values
  int errors = 0;
  for (uint32_t i = 0; i < n_rays; ++i) {
    if (h_counts[i] != ref_counts[i]) {
      if (errors < 10)
        printf("*** error: ray[%d] count expected=%u, actual=%u\n", i, ref_counts[i], h_counts[i]);
      ++errors;
      continue;
    }
    // Check first min(count, 4) t_starts values
    uint32_t check = h_counts[i] < 4 ? h_counts[i] : 4;
    uint32_t offset = i * max_samples;
    for (uint32_t j = 0; j < check; ++j) {
      float diff = h_t_starts[offset + j] - ref_t_starts[offset + j];
      if (diff < 0) diff = -diff;
      float mag = ref_t_starts[offset + j];
      if (mag < 0) mag = -mag;
      if (mag < 1e-6f) mag = 1e-6f;
      if (diff / mag > 1e-5f) {
        if (errors < 10)
          printf("*** error: ray[%d] t_starts[%d] expected=%f, actual=%f\n",
                 i, j, ref_t_starts[offset + j], h_t_starts[offset + j]);
        ++errors;
      }
    }
  }

  cleanup();
  if (errors) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
