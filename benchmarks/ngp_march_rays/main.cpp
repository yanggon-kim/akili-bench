#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <cmath>
#include <cstdlib>
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
uint32_t n_rays    = 8;
uint32_t grid_res  = 16;
uint32_t max_steps = 32;

vx_device_h device       = nullptr;
vx_buffer_h rays_o_buf   = nullptr;
vx_buffer_h rays_d_buf   = nullptr;
vx_buffer_h nears_buf    = nullptr;
vx_buffer_h fars_buf     = nullptr;
vx_buffer_h grid_buf     = nullptr;
vx_buffer_h xyzs_buf     = nullptr;
vx_buffer_h deltas_buf   = nullptr;
vx_buffer_h counts_buf   = nullptr;
vx_buffer_h krnl_buffer  = nullptr;
vx_buffer_h args_buffer  = nullptr;
kernel_arg_t kernel_arg  = {};

static void show_usage() {
   std::cout << "Vortex NGP March Rays Benchmark." << std::endl;
   std::cout << "Usage: [-k: kernel] [-r n_rays] [-g grid_res] [-s max_steps] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "r:g:s:k:h")) != -1) {
    switch (c) {
    case 'r': n_rays    = atoi(optarg); break;
    case 'g': grid_res  = atoi(optarg); break;
    case 's': max_steps = atoi(optarg); break;
    case 'k': kernel_file = optarg; break;
    case 'h': show_usage(); exit(0); break;
    default: show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(rays_o_buf);
    vx_mem_free(rays_d_buf);
    vx_mem_free(nears_buf);
    vx_mem_free(fars_buf);
    vx_mem_free(grid_buf);
    vx_mem_free(xyzs_buf);
    vx_mem_free(deltas_buf);
    vx_mem_free(counts_buf);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

// Morton helpers (same as kernel)
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
    return std::fmin(mx, std::fmax(mn, x));
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  std::srand(42);

  float bound = 1.0f;
  float SQRT3 = 1.7320508075688772f;
  float dt_min = 2.0f * SQRT3 * bound / (float)max_steps;

  uint32_t H = grid_res;
  uint32_t grid_bytes = H * H * H / 8;
  if (grid_bytes == 0) grid_bytes = 1;  // minimum

  std::cout << "NGP March Rays: n_rays=" << n_rays << " grid=" << H
            << " max_steps=" << max_steps << " bound=" << bound
            << " dt_min=" << dt_min << std::endl;

  // Generate input data
  std::vector<float> h_rays_o(n_rays * 3), h_rays_d(n_rays * 3);
  std::vector<float> h_nears(n_rays), h_fars(n_rays);
  std::vector<uint8_t> h_grid(grid_bytes);

  for (uint32_t i = 0; i < n_rays * 3; i++) {
    h_rays_o[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
    h_rays_d[i] = (float)rand() / RAND_MAX - 0.5f;
  }
  // Normalize ray directions
  for (uint32_t i = 0; i < n_rays; i++) {
    float *d = &h_rays_d[i * 3];
    float len = sqrtf(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
    if (len > 0) { d[0] /= len; d[1] /= len; d[2] /= len; }
  }
  // Compute near/far via AABB [-bound, bound]^3
  for (uint32_t i = 0; i < n_rays; i++) {
    float ox = h_rays_o[i*3], oy = h_rays_o[i*3+1], oz = h_rays_o[i*3+2];
    float dx = h_rays_d[i*3], dy = h_rays_d[i*3+1], dz = h_rays_d[i*3+2];
    float rdx = 1.0f/dx, rdy = 1.0f/dy, rdz = 1.0f/dz;
    float near = ((dx > 0 ? -bound : bound) - ox) * rdx;
    float far  = ((dx > 0 ? bound : -bound) - ox) * rdx;
    near = std::fmax(near, ((dy > 0 ? -bound : bound) - oy) * rdy);
    far  = std::fmin(far,  ((dy > 0 ? bound : -bound) - oy) * rdy);
    near = std::fmax(near, ((dz > 0 ? -bound : bound) - oz) * rdz);
    far  = std::fmin(far,  ((dz > 0 ? bound : -bound) - oz) * rdz);
    h_nears[i] = (near < far) ? std::fmax(near, 0.01f) : 0.01f;
    h_fars[i]  = (near < far) ? far : 0.01f;
  }
  // Random occupancy grid (~50% occupied)
  for (uint32_t i = 0; i < grid_bytes; i++)
    h_grid[i] = (rand() % 2) ? 0xFF : 0x00;

  // Allocate device buffers
  RT_CHECK(vx_dev_open(&device));

  kernel_arg.n_rays    = n_rays;
  kernel_arg.grid_res  = H;
  kernel_arg.max_steps = max_steps;
  kernel_arg.bound     = bound;
  kernel_arg.dt_min    = dt_min;

  uint32_t rays_o_size  = n_rays * 3 * sizeof(float);
  uint32_t rays_d_size  = n_rays * 3 * sizeof(float);
  uint32_t nears_size   = n_rays * sizeof(float);
  uint32_t fars_size    = n_rays * sizeof(float);
  uint32_t xyzs_size    = n_rays * max_steps * 3 * sizeof(float);
  uint32_t deltas_size  = n_rays * max_steps * sizeof(float);
  uint32_t counts_size  = n_rays * sizeof(uint32_t);

  RT_CHECK(vx_mem_alloc(device, rays_o_size, VX_MEM_READ, &rays_o_buf));
  RT_CHECK(vx_mem_address(rays_o_buf, &kernel_arg.rays_o_addr));
  RT_CHECK(vx_mem_alloc(device, rays_d_size, VX_MEM_READ, &rays_d_buf));
  RT_CHECK(vx_mem_address(rays_d_buf, &kernel_arg.rays_d_addr));
  RT_CHECK(vx_mem_alloc(device, nears_size, VX_MEM_READ, &nears_buf));
  RT_CHECK(vx_mem_address(nears_buf, &kernel_arg.nears_addr));
  RT_CHECK(vx_mem_alloc(device, fars_size, VX_MEM_READ, &fars_buf));
  RT_CHECK(vx_mem_address(fars_buf, &kernel_arg.fars_addr));
  RT_CHECK(vx_mem_alloc(device, grid_bytes, VX_MEM_READ, &grid_buf));
  RT_CHECK(vx_mem_address(grid_buf, &kernel_arg.grid_addr));
  RT_CHECK(vx_mem_alloc(device, xyzs_size, VX_MEM_WRITE, &xyzs_buf));
  RT_CHECK(vx_mem_address(xyzs_buf, &kernel_arg.xyzs_addr));
  RT_CHECK(vx_mem_alloc(device, deltas_size, VX_MEM_WRITE, &deltas_buf));
  RT_CHECK(vx_mem_address(deltas_buf, &kernel_arg.deltas_addr));
  RT_CHECK(vx_mem_alloc(device, counts_size, VX_MEM_WRITE, &counts_buf));
  RT_CHECK(vx_mem_address(counts_buf, &kernel_arg.counts_addr));

  RT_CHECK(vx_copy_to_dev(rays_o_buf, h_rays_o.data(), 0, rays_o_size));
  RT_CHECK(vx_copy_to_dev(rays_d_buf, h_rays_d.data(), 0, rays_d_size));
  RT_CHECK(vx_copy_to_dev(nears_buf, h_nears.data(), 0, nears_size));
  RT_CHECK(vx_copy_to_dev(fars_buf, h_fars.data(), 0, fars_size));
  RT_CHECK(vx_copy_to_dev(grid_buf, h_grid.data(), 0, grid_bytes));

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // Read back results
  std::vector<float> h_xyzs(n_rays * max_steps * 3, 0);
  std::vector<float> h_deltas(n_rays * max_steps, 0);
  std::vector<uint32_t> h_counts(n_rays, 0);

  RT_CHECK(vx_copy_from_dev(h_xyzs.data(), xyzs_buf, 0, xyzs_size));
  RT_CHECK(vx_copy_from_dev(h_deltas.data(), deltas_buf, 0, deltas_size));
  RT_CHECK(vx_copy_from_dev(h_counts.data(), counts_buf, 0, counts_size));

  // CPU reference
  int errors = 0;
  uint32_t total_steps = 0;
  for (uint32_t ray = 0; ray < n_rays; ray++) {
    float ox = h_rays_o[ray*3], oy = h_rays_o[ray*3+1], oz = h_rays_o[ray*3+2];
    float dx = h_rays_d[ray*3], dy = h_rays_d[ray*3+1], dz = h_rays_d[ray*3+2];
    float near = h_nears[ray], far = h_fars[ray];

    float t = near;
    uint32_t ref_count = 0;
    std::vector<float> ref_xyz;
    std::vector<float> ref_dt;

    while (t < far && ref_count < max_steps) {
      float x = my_clamp(ox + t * dx, -bound, bound);
      float y = my_clamp(oy + t * dy, -bound, bound);
      float z = my_clamp(oz + t * dz, -bound, bound);

      int nx = (int)my_clamp(0.5f * (x / bound + 1.0f) * H, 0.0f, (float)(H - 1));
      int ny = (int)my_clamp(0.5f * (y / bound + 1.0f) * H, 0.0f, (float)(H - 1));
      int nz = (int)my_clamp(0.5f * (z / bound + 1.0f) * H, 0.0f, (float)(H - 1));

      uint32_t idx = morton3D((uint32_t)nx, (uint32_t)ny, (uint32_t)nz);
      uint8_t occ = h_grid[idx / 8] & (1 << (idx % 8));

      if (occ) {
        ref_xyz.push_back(x);
        ref_xyz.push_back(y);
        ref_xyz.push_back(z);
        ref_dt.push_back(dt_min);
        ref_count++;
      }
      t += dt_min;
    }

    // Check count
    if (h_counts[ray] != ref_count) {
      if (errors < 10)
        printf("*** error: ray %d count: expected=%u, actual=%u\n", ray, ref_count, h_counts[ray]);
      ++errors;
      continue;
    }

    // Check first few xyz positions
    uint32_t check_n = (ref_count < 4) ? ref_count : 4;
    for (uint32_t s = 0; s < check_n; s++) {
      for (int c = 0; c < 3; c++) {
        float expected = ref_xyz[s * 3 + c];
        float actual   = h_xyzs[ray * max_steps * 3 + s * 3 + c];
        if (std::fabs(expected - actual) > 1e-4f) {
          if (errors < 10)
            printf("*** error: ray %d step %d dim %d: expected=%f, actual=%f\n",
                   ray, s, c, expected, actual);
          ++errors;
        }
      }
    }

    total_steps += ref_count;
  }

  std::cout << "Total marched steps: " << total_steps << std::endl;

  cleanup();
  if (errors) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
