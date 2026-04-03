#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <cmath>
#include <vortex.h>
#include "common.h"

#define FLOAT_ULP 6

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
vx_buffer_h aabb_buffer = nullptr;
vx_buffer_h t_mins_buffer = nullptr;
vx_buffer_h t_maxs_buffer = nullptr;
vx_buffer_h hits_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex NeRF Ray-AABB Intersect Benchmark." << std::endl;
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
    vx_mem_free(aabb_buffer);
    vx_mem_free(t_mins_buffer);
    vx_mem_free(t_maxs_buffer);
    vx_mem_free(hits_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

// CPU reference for ray-AABB intersection
static void cpu_ray_aabb_intersect(
    const float* rays_o, const float* rays_d, const float* aabb,
    float min_near, uint32_t n,
    float* t_mins, float* t_maxs, uint32_t* hits) {
  for (uint32_t i = 0; i < n; ++i) {
    float ox = rays_o[i*3+0], oy = rays_o[i*3+1], oz = rays_o[i*3+2];
    float dx = rays_d[i*3+0], dy = rays_d[i*3+1], dz = rays_d[i*3+2];
    float idx = 1.0f / dx, idy = 1.0f / dy, idz = 1.0f / dz;

    float tmin = (aabb[idx > 0.0f ? 0 : 3] - ox) * idx;
    float tmax = (aabb[idx > 0.0f ? 3 : 0] - ox) * idx;
    float tymin = (aabb[idy > 0.0f ? 1 : 4] - oy) * idy;
    float tymax = (aabb[idy > 0.0f ? 4 : 1] - oy) * idy;

    if (tmin > tymax || tymin > tmax) {
      t_mins[i] = 0.0f; t_maxs[i] = 0.0f; hits[i] = 0;
      continue;
    }
    if (tymin > tmin) tmin = tymin;
    if (tymax < tmax) tmax = tymax;

    float tzmin = (aabb[idz > 0.0f ? 2 : 5] - oz) * idz;
    float tzmax = (aabb[idz > 0.0f ? 5 : 2] - oz) * idz;

    if (tmin > tzmax || tzmin > tmax) {
      t_mins[i] = 0.0f; t_maxs[i] = 0.0f; hits[i] = 0;
      continue;
    }
    if (tzmin > tmin) tmin = tzmin;
    if (min_near > tmin) tmin = min_near;
    if (tzmax < tmax) tmax = tzmax;

    t_mins[i] = tmin;
    t_maxs[i] = tmax;
    hits[i] = (tmin <= tmax) ? 1 : 0;
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  std::srand(42);

  float min_near = 0.01f;
  float h_aabb[6] = {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};

  uint32_t rays_o_size = n_rays * 3 * sizeof(float);
  uint32_t rays_d_size = n_rays * 3 * sizeof(float);
  uint32_t aabb_size   = 6 * sizeof(float);
  uint32_t t_size      = n_rays * sizeof(float);
  uint32_t hits_size   = n_rays * sizeof(uint32_t);

  std::cout << "Ray-AABB Intersect: n_rays=" << n_rays << std::endl;

  // Generate rays
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

  RT_CHECK(vx_mem_alloc(device, rays_o_size, VX_MEM_READ, &rays_o_buffer));
  RT_CHECK(vx_mem_address(rays_o_buffer, &kernel_arg.rays_o_addr));
  RT_CHECK(vx_mem_alloc(device, rays_d_size, VX_MEM_READ, &rays_d_buffer));
  RT_CHECK(vx_mem_address(rays_d_buffer, &kernel_arg.rays_d_addr));
  RT_CHECK(vx_mem_alloc(device, aabb_size, VX_MEM_READ, &aabb_buffer));
  RT_CHECK(vx_mem_address(aabb_buffer, &kernel_arg.aabb_addr));
  RT_CHECK(vx_mem_alloc(device, t_size, VX_MEM_WRITE, &t_mins_buffer));
  RT_CHECK(vx_mem_address(t_mins_buffer, &kernel_arg.t_mins_addr));
  RT_CHECK(vx_mem_alloc(device, t_size, VX_MEM_WRITE, &t_maxs_buffer));
  RT_CHECK(vx_mem_address(t_maxs_buffer, &kernel_arg.t_maxs_addr));
  RT_CHECK(vx_mem_alloc(device, hits_size, VX_MEM_WRITE, &hits_buffer));
  RT_CHECK(vx_mem_address(hits_buffer, &kernel_arg.hits_addr));

  RT_CHECK(vx_copy_to_dev(rays_o_buffer, h_rays_o.data(), 0, rays_o_size));
  RT_CHECK(vx_copy_to_dev(rays_d_buffer, h_rays_d.data(), 0, rays_d_size));
  RT_CHECK(vx_copy_to_dev(aabb_buffer, h_aabb, 0, aabb_size));

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // Read back results
  std::vector<float> h_t_mins(n_rays, 0), h_t_maxs(n_rays, 0);
  std::vector<uint32_t> h_hits(n_rays, 0);
  RT_CHECK(vx_copy_from_dev(h_t_mins.data(), t_mins_buffer, 0, t_size));
  RT_CHECK(vx_copy_from_dev(h_t_maxs.data(), t_maxs_buffer, 0, t_size));
  RT_CHECK(vx_copy_from_dev(h_hits.data(), hits_buffer, 0, hits_size));

  // CPU reference
  std::vector<float> ref_t_mins(n_rays), ref_t_maxs(n_rays);
  std::vector<uint32_t> ref_hits(n_rays);
  cpu_ray_aabb_intersect(h_rays_o.data(), h_rays_d.data(), h_aabb,
                         min_near, n_rays,
                         ref_t_mins.data(), ref_t_maxs.data(), ref_hits.data());

  // Verify
  int errors = 0;
  for (uint32_t i = 0; i < n_rays; ++i) {
    if (h_hits[i] != ref_hits[i]) {
      if (errors < 10)
        printf("*** error: ray[%d] hit expected=%u, actual=%u\n", i, ref_hits[i], h_hits[i]);
      ++errors;
      continue;
    }
    if (ref_hits[i]) {
      union { float f; int32_t i; } fa, fb;
      fa.f = h_t_mins[i]; fb.f = ref_t_mins[i];
      if (std::abs(fa.i - fb.i) > FLOAT_ULP) {
        if (errors < 10)
          printf("*** error: ray[%d] tmin expected=%f, actual=%f\n", i, ref_t_mins[i], h_t_mins[i]);
        ++errors;
      }
      fa.f = h_t_maxs[i]; fb.f = ref_t_maxs[i];
      if (std::abs(fa.i - fb.i) > FLOAT_ULP) {
        if (errors < 10)
          printf("*** error: ray[%d] tmax expected=%f, actual=%f\n", i, ref_t_maxs[i], h_t_maxs[i]);
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
