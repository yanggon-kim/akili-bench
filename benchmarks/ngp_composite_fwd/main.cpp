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
uint32_t n_rays = 32;
uint32_t samples_per_ray = 16;

vx_device_h device           = nullptr;
vx_buffer_h sigmas_buf       = nullptr;
vx_buffer_h rgbs_buf         = nullptr;
vx_buffer_h deltas_buf       = nullptr;
vx_buffer_h weights_sum_buf  = nullptr;
vx_buffer_h depth_buf        = nullptr;
vx_buffer_h image_buf        = nullptr;
vx_buffer_h krnl_buffer      = nullptr;
vx_buffer_h args_buffer      = nullptr;
kernel_arg_t kernel_arg      = {};

static void show_usage() {
   std::cout << "Vortex NGP Composite Forward Benchmark." << std::endl;
   std::cout << "Usage: [-k: kernel] [-r n_rays] [-s samples_per_ray] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "r:s:k:h")) != -1) {
    switch (c) {
    case 'r': n_rays = atoi(optarg); break;
    case 's': samples_per_ray = atoi(optarg); break;
    case 'k': kernel_file = optarg; break;
    case 'h': show_usage(); exit(0); break;
    default: show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(sigmas_buf);
    vx_mem_free(rgbs_buf);
    vx_mem_free(deltas_buf);
    vx_mem_free(weights_sum_buf);
    vx_mem_free(depth_buf);
    vx_mem_free(image_buf);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  std::srand(42);

  float T_thresh = 1e-4f;
  uint32_t total_samples = n_rays * samples_per_ray;

  std::cout << "NGP Composite Forward: n_rays=" << n_rays
            << " samples_per_ray=" << samples_per_ray
            << " T_thresh=" << T_thresh << std::endl;

  // Generate input data
  std::vector<float> h_sigmas(total_samples);
  std::vector<float> h_rgbs(total_samples * 3);
  std::vector<float> h_deltas(total_samples);

  for (uint32_t i = 0; i < total_samples; i++) {
    h_sigmas[i] = 20.0f * (float)rand() / RAND_MAX;  // [0, 20]
    h_deltas[i] = 0.001f + 0.009f * (float)rand() / RAND_MAX;  // [0.001, 0.01]
  }
  for (uint32_t i = 0; i < total_samples * 3; i++) {
    h_rgbs[i] = (float)rand() / RAND_MAX;  // [0, 1]
  }

  // Allocate device buffers
  RT_CHECK(vx_dev_open(&device));

  kernel_arg.n_rays = n_rays;
  kernel_arg.samples_per_ray = samples_per_ray;
  kernel_arg.T_thresh = T_thresh;

  uint32_t sigmas_size = total_samples * sizeof(float);
  uint32_t rgbs_size   = total_samples * 3 * sizeof(float);
  uint32_t deltas_size = total_samples * sizeof(float);
  uint32_t ws_size     = n_rays * sizeof(float);
  uint32_t depth_size  = n_rays * sizeof(float);
  uint32_t image_size  = n_rays * 3 * sizeof(float);

  RT_CHECK(vx_mem_alloc(device, sigmas_size, VX_MEM_READ, &sigmas_buf));
  RT_CHECK(vx_mem_address(sigmas_buf, &kernel_arg.sigmas_addr));
  RT_CHECK(vx_mem_alloc(device, rgbs_size, VX_MEM_READ, &rgbs_buf));
  RT_CHECK(vx_mem_address(rgbs_buf, &kernel_arg.rgbs_addr));
  RT_CHECK(vx_mem_alloc(device, deltas_size, VX_MEM_READ, &deltas_buf));
  RT_CHECK(vx_mem_address(deltas_buf, &kernel_arg.deltas_addr));
  RT_CHECK(vx_mem_alloc(device, ws_size, VX_MEM_WRITE, &weights_sum_buf));
  RT_CHECK(vx_mem_address(weights_sum_buf, &kernel_arg.weights_sum_addr));
  RT_CHECK(vx_mem_alloc(device, depth_size, VX_MEM_WRITE, &depth_buf));
  RT_CHECK(vx_mem_address(depth_buf, &kernel_arg.depth_addr));
  RT_CHECK(vx_mem_alloc(device, image_size, VX_MEM_WRITE, &image_buf));
  RT_CHECK(vx_mem_address(image_buf, &kernel_arg.image_addr));

  RT_CHECK(vx_copy_to_dev(sigmas_buf, h_sigmas.data(), 0, sigmas_size));
  RT_CHECK(vx_copy_to_dev(rgbs_buf, h_rgbs.data(), 0, rgbs_size));
  RT_CHECK(vx_copy_to_dev(deltas_buf, h_deltas.data(), 0, deltas_size));

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // Read back results
  std::vector<float> h_ws(n_rays, 0);
  std::vector<float> h_depth(n_rays, 0);
  std::vector<float> h_image(n_rays * 3, 0);

  RT_CHECK(vx_copy_from_dev(h_ws.data(), weights_sum_buf, 0, ws_size));
  RT_CHECK(vx_copy_from_dev(h_depth.data(), depth_buf, 0, depth_size));
  RT_CHECK(vx_copy_from_dev(h_image.data(), image_buf, 0, image_size));

  // CPU reference
  int errors = 0;
  for (uint32_t ray = 0; ray < n_rays; ray++) {
    uint32_t base = ray * samples_per_ray;
    uint32_t base3 = ray * samples_per_ray * 3;

    float T = 1.0f;
    float ref_r = 0, ref_g = 0, ref_b = 0, ref_ws = 0;
    float t_accum = 0, ref_dp = 0;

    for (uint32_t step = 0; step < samples_per_ray; step++) {
      float sigma = h_sigmas[base + step];
      float delta = h_deltas[base + step];
      float alpha = 1.0f - std::exp(-sigma * delta);
      float weight = alpha * T;

      ref_r += weight * h_rgbs[base3 + step * 3 + 0];
      ref_g += weight * h_rgbs[base3 + step * 3 + 1];
      ref_b += weight * h_rgbs[base3 + step * 3 + 2];

      t_accum += delta;
      ref_dp += weight * t_accum;
      ref_ws += weight;

      T *= (1.0f - alpha);
      if (T < T_thresh) break;
    }

    // Compare weights_sum
    if (std::fabs(h_ws[ray] - ref_ws) > 1e-4f) {
      if (errors < 10)
        printf("*** error: ray %d weights_sum: expected=%f, actual=%f\n", ray, ref_ws, h_ws[ray]);
      ++errors;
    }
    // Compare depth
    if (std::fabs(h_depth[ray] - ref_dp) > 1e-4f) {
      if (errors < 10)
        printf("*** error: ray %d depth: expected=%f, actual=%f\n", ray, ref_dp, h_depth[ray]);
      ++errors;
    }
    // Compare image RGB
    float ref_img[3] = {ref_r, ref_g, ref_b};
    for (int c = 0; c < 3; c++) {
      if (std::fabs(h_image[ray * 3 + c] - ref_img[c]) > 1e-4f) {
        if (errors < 10)
          printf("*** error: ray %d image[%d]: expected=%f, actual=%f\n",
                 ray, c, ref_img[c], h_image[ray * 3 + c]);
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
