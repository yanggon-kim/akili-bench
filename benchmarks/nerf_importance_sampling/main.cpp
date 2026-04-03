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
uint32_t n_rays = 32;
uint32_t n_cdf  = 16;
uint32_t n_new  = 8;

vx_device_h device = nullptr;
vx_buffer_h cdfs_buffer = nullptr;
vx_buffer_h out_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex NeRF Importance Sampling Benchmark." << std::endl;
   std::cout << "Usage: [-k: kernel] [-r rays] [-c cdf_size] [-n new_samples] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "r:c:n:k:h")) != -1) {
    switch (c) {
    case 'r': n_rays = atoi(optarg); break;
    case 'c': n_cdf  = atoi(optarg); break;
    case 'n': n_new  = atoi(optarg); break;
    case 'k': kernel_file = optarg; break;
    case 'h': show_usage(); exit(0); break;
    default: show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(cdfs_buffer);
    vx_mem_free(out_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

// CPU reference: same binary search on CDF
static void cpu_importance_sampling(
    const float* cdfs, uint32_t n_rays, uint32_t n_cdf, uint32_t n_new,
    float* out) {
  for (uint32_t r = 0; r < n_rays; ++r) {
    const float* cdf = cdfs + r * n_cdf;
    float* dst = out + r * n_new;
    for (uint32_t s = 0; s < n_new; ++s) {
      float u = ((float)s + 0.5f) / (float)n_new;
      uint32_t lo = 0, hi = n_cdf;
      while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (cdf[mid] < u)
          lo = mid + 1;
        else
          hi = mid;
      }
      if (lo >= n_cdf)
        lo = n_cdf - 1;
      dst[s] = (float)lo;
    }
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  std::srand(42);

  uint32_t cdfs_size = n_rays * n_cdf * sizeof(float);
  uint32_t out_size  = n_rays * n_new * sizeof(float);

  std::cout << "Importance Sampling: n_rays=" << n_rays
            << " n_cdf=" << n_cdf << " n_new=" << n_new << std::endl;

  // Generate monotonically increasing CDF per ray (0 -> 1)
  std::vector<float> h_cdfs(n_rays * n_cdf);
  for (uint32_t r = 0; r < n_rays; ++r) {
    for (uint32_t c = 0; c < n_cdf; ++c) {
      h_cdfs[r * n_cdf + c] = (float)(c + 1) / (float)n_cdf;
    }
  }

  RT_CHECK(vx_dev_open(&device));

  kernel_arg.num_tasks = n_rays;
  kernel_arg.n_cdf = n_cdf;
  kernel_arg.n_new = n_new;

  RT_CHECK(vx_mem_alloc(device, cdfs_size, VX_MEM_READ, &cdfs_buffer));
  RT_CHECK(vx_mem_address(cdfs_buffer, &kernel_arg.cdfs_addr));
  RT_CHECK(vx_mem_alloc(device, out_size, VX_MEM_WRITE, &out_buffer));
  RT_CHECK(vx_mem_address(out_buffer, &kernel_arg.out_addr));

  RT_CHECK(vx_copy_to_dev(cdfs_buffer, h_cdfs.data(), 0, cdfs_size));

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // Read back results
  std::vector<float> h_out(n_rays * n_new, 0);
  RT_CHECK(vx_copy_from_dev(h_out.data(), out_buffer, 0, out_size));

  // CPU reference
  std::vector<float> ref_out(n_rays * n_new);
  cpu_importance_sampling(h_cdfs.data(), n_rays, n_cdf, n_new, ref_out.data());

  // Verify: exact match (integer indices stored as floats)
  int errors = 0;
  for (uint32_t i = 0; i < n_rays * n_new; ++i) {
    if (h_out[i] != ref_out[i]) {
      if (errors < 10)
        printf("*** error: [%d] expected=%f, actual=%f\n", i, ref_out[i], h_out[i]);
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
