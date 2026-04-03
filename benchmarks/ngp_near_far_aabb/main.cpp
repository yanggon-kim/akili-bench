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
uint32_t n_rays = 64;

vx_device_h device = nullptr;
vx_buffer_h rays_o_buffer = nullptr;
vx_buffer_h rays_d_buffer = nullptr;
vx_buffer_h nears_buffer = nullptr;
vx_buffer_h fars_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex NGP Near-Far AABB Benchmark." << std::endl;
   std::cout << "Usage: [-k: kernel] [-r rays] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "r:k:h")) != -1) {
    switch (c) {
    case 'r': n_rays = atoi(optarg); break;
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
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

// CPU reference: torch-ngp style ray-AABB intersection
static void cpu_near_far_aabb(
    const float* rays_o, const float* rays_d,
    float aabb0, float aabb1, float aabb2,
    float aabb3, float aabb4, float aabb5,
    float min_near, uint32_t n,
    float* nears, float* fars) {
  for (uint32_t i = 0; i < n; ++i) {
    float ox = rays_o[i*3+0], oy = rays_o[i*3+1], oz = rays_o[i*3+2];
    float dx = rays_d[i*3+0], dy = rays_d[i*3+1], dz = rays_d[i*3+2];
    float rdx = 1.0f / dx, rdy = 1.0f / dy, rdz = 1.0f / dz;

    float near = ((dx > 0.0f ? aabb0 : aabb3) - ox) * rdx;
    float far  = ((dx > 0.0f ? aabb3 : aabb0) - ox) * rdx;
    near = fmaxf(near, ((dy > 0.0f ? aabb1 : aabb4) - oy) * rdy);
    far  = fminf(far,  ((dy > 0.0f ? aabb4 : aabb1) - oy) * rdy);
    near = fmaxf(near, ((dz > 0.0f ? aabb2 : aabb5) - oz) * rdz);
    far  = fminf(far,  ((dz > 0.0f ? aabb5 : aabb2) - oz) * rdz);

    nears[i] = (near < far) ? fmaxf(near, min_near) : min_near;
    fars[i]  = (near < far) ? far : min_near;
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  std::srand(42);

  float min_near = 0.01f;
  float aabb[6] = {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};

  uint32_t rays_o_size = n_rays * 3 * sizeof(float);
  uint32_t rays_d_size = n_rays * 3 * sizeof(float);
  uint32_t out_size    = n_rays * sizeof(float);

  std::cout << "NGP Near-Far AABB: n_rays=" << n_rays << std::endl;

  // Generate rays: origins near center, random normalized directions
  std::vector<float> h_rays_o(n_rays * 3), h_rays_d(n_rays * 3);
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
  }

  RT_CHECK(vx_dev_open(&device));

  kernel_arg.num_tasks = n_rays;
  kernel_arg.min_near  = min_near;
  kernel_arg.aabb0 = aabb[0];
  kernel_arg.aabb1 = aabb[1];
  kernel_arg.aabb2 = aabb[2];
  kernel_arg.aabb3 = aabb[3];
  kernel_arg.aabb4 = aabb[4];
  kernel_arg.aabb5 = aabb[5];

  RT_CHECK(vx_mem_alloc(device, rays_o_size, VX_MEM_READ, &rays_o_buffer));
  RT_CHECK(vx_mem_address(rays_o_buffer, &kernel_arg.rays_o_addr));
  RT_CHECK(vx_mem_alloc(device, rays_d_size, VX_MEM_READ, &rays_d_buffer));
  RT_CHECK(vx_mem_address(rays_d_buffer, &kernel_arg.rays_d_addr));
  RT_CHECK(vx_mem_alloc(device, out_size, VX_MEM_WRITE, &nears_buffer));
  RT_CHECK(vx_mem_address(nears_buffer, &kernel_arg.nears_addr));
  RT_CHECK(vx_mem_alloc(device, out_size, VX_MEM_WRITE, &fars_buffer));
  RT_CHECK(vx_mem_address(fars_buffer, &kernel_arg.fars_addr));

  RT_CHECK(vx_copy_to_dev(rays_o_buffer, h_rays_o.data(), 0, rays_o_size));
  RT_CHECK(vx_copy_to_dev(rays_d_buffer, h_rays_d.data(), 0, rays_d_size));

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // Read back results
  std::vector<float> h_nears(n_rays, 0), h_fars(n_rays, 0);
  RT_CHECK(vx_copy_from_dev(h_nears.data(), nears_buffer, 0, out_size));
  RT_CHECK(vx_copy_from_dev(h_fars.data(), fars_buffer, 0, out_size));

  // CPU reference
  std::vector<float> ref_nears(n_rays), ref_fars(n_rays);
  cpu_near_far_aabb(h_rays_o.data(), h_rays_d.data(),
                    aabb[0], aabb[1], aabb[2], aabb[3], aabb[4], aabb[5],
                    min_near, n_rays,
                    ref_nears.data(), ref_fars.data());

  // Verify with relative tolerance 1e-5
  int errors = 0;
  for (uint32_t i = 0; i < n_rays; ++i) {
    // Check nears
    float diff = h_nears[i] - ref_nears[i];
    if (diff < 0) diff = -diff;
    float mag = ref_nears[i];
    if (mag < 0) mag = -mag;
    if (mag < 1e-6f) mag = 1e-6f;
    if (diff / mag > 1e-5f && diff > 1e-7f) {
      if (errors < 10)
        printf("*** error: ray[%d] near expected=%f, actual=%f\n", i, ref_nears[i], h_nears[i]);
      ++errors;
    }
    // Check fars
    diff = h_fars[i] - ref_fars[i];
    if (diff < 0) diff = -diff;
    mag = ref_fars[i];
    if (mag < 0) mag = -mag;
    if (mag < 1e-6f) mag = 1e-6f;
    if (diff / mag > 1e-5f && diff > 1e-7f) {
      if (errors < 10)
        printf("*** error: ray[%d] far expected=%f, actual=%f\n", i, ref_fars[i], h_fars[i]);
      ++errors;
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
