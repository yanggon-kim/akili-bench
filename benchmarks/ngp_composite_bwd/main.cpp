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

vx_device_h device              = nullptr;
vx_buffer_h sigmas_buf          = nullptr;
vx_buffer_h rgbs_buf            = nullptr;
vx_buffer_h deltas_buf          = nullptr;
vx_buffer_h weights_sum_buf     = nullptr;
vx_buffer_h image_buf           = nullptr;
vx_buffer_h grad_weights_sum_buf = nullptr;
vx_buffer_h grad_image_buf      = nullptr;
vx_buffer_h grad_sigmas_buf     = nullptr;
vx_buffer_h grad_rgbs_buf       = nullptr;
vx_buffer_h krnl_buffer         = nullptr;
vx_buffer_h args_buffer         = nullptr;
kernel_arg_t kernel_arg         = {};

static void show_usage() {
   std::cout << "Vortex NGP Composite Backward Benchmark." << std::endl;
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
    vx_mem_free(image_buf);
    vx_mem_free(grad_weights_sum_buf);
    vx_mem_free(grad_image_buf);
    vx_mem_free(grad_sigmas_buf);
    vx_mem_free(grad_rgbs_buf);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

// CPU forward pass to compute weights_sum and image
static void cpu_forward(
    const float* sigmas, const float* rgbs, const float* deltas,
    uint32_t n_rays, uint32_t S, float T_thresh,
    float* weights_sum, float* image)
{
  for (uint32_t ray = 0; ray < n_rays; ray++) {
    uint32_t base = ray * S;
    uint32_t base3 = ray * S * 3;
    float T = 1.0f;
    float r = 0, g = 0, b = 0, ws = 0;
    for (uint32_t step = 0; step < S; step++) {
      float alpha = 1.0f - std::exp(-sigmas[base + step] * deltas[base + step]);
      float weight = alpha * T;
      r += weight * rgbs[base3 + step * 3 + 0];
      g += weight * rgbs[base3 + step * 3 + 1];
      b += weight * rgbs[base3 + step * 3 + 2];
      ws += weight;
      T *= (1.0f - alpha);
      if (T < T_thresh) break;
    }
    weights_sum[ray] = ws;
    image[ray * 3 + 0] = r;
    image[ray * 3 + 1] = g;
    image[ray * 3 + 2] = b;
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  std::srand(42);

  float T_thresh = 1e-4f;
  uint32_t total_samples = n_rays * samples_per_ray;

  std::cout << "NGP Composite Backward: n_rays=" << n_rays
            << " samples_per_ray=" << samples_per_ray
            << " T_thresh=" << T_thresh << std::endl;

  // Generate input data
  std::vector<float> h_sigmas(total_samples);
  std::vector<float> h_rgbs(total_samples * 3);
  std::vector<float> h_deltas(total_samples);

  for (uint32_t i = 0; i < total_samples; i++) {
    h_sigmas[i] = 20.0f * (float)rand() / RAND_MAX;
    h_deltas[i] = 0.001f + 0.009f * (float)rand() / RAND_MAX;
  }
  for (uint32_t i = 0; i < total_samples * 3; i++) {
    h_rgbs[i] = (float)rand() / RAND_MAX;
  }

  // Run CPU forward to get weights_sum and image
  std::vector<float> h_weights_sum(n_rays);
  std::vector<float> h_image(n_rays * 3);
  cpu_forward(h_sigmas.data(), h_rgbs.data(), h_deltas.data(),
              n_rays, samples_per_ray, T_thresh,
              h_weights_sum.data(), h_image.data());

  // Set gradients: grad_weights_sum=1.0, grad_image=[1,1,1]
  std::vector<float> h_grad_ws(n_rays, 1.0f);
  std::vector<float> h_grad_image(n_rays * 3, 1.0f);

  // Allocate device buffers
  RT_CHECK(vx_dev_open(&device));

  kernel_arg.n_rays = n_rays;
  kernel_arg.samples_per_ray = samples_per_ray;
  kernel_arg.T_thresh = T_thresh;

  uint32_t sigmas_size = total_samples * sizeof(float);
  uint32_t rgbs_size   = total_samples * 3 * sizeof(float);
  uint32_t deltas_size = total_samples * sizeof(float);
  uint32_t ws_size     = n_rays * sizeof(float);
  uint32_t image_size  = n_rays * 3 * sizeof(float);
  uint32_t grad_sigmas_size = total_samples * sizeof(float);
  uint32_t grad_rgbs_size   = total_samples * 3 * sizeof(float);

  RT_CHECK(vx_mem_alloc(device, sigmas_size, VX_MEM_READ, &sigmas_buf));
  RT_CHECK(vx_mem_address(sigmas_buf, &kernel_arg.sigmas_addr));
  RT_CHECK(vx_mem_alloc(device, rgbs_size, VX_MEM_READ, &rgbs_buf));
  RT_CHECK(vx_mem_address(rgbs_buf, &kernel_arg.rgbs_addr));
  RT_CHECK(vx_mem_alloc(device, deltas_size, VX_MEM_READ, &deltas_buf));
  RT_CHECK(vx_mem_address(deltas_buf, &kernel_arg.deltas_addr));
  RT_CHECK(vx_mem_alloc(device, ws_size, VX_MEM_READ, &weights_sum_buf));
  RT_CHECK(vx_mem_address(weights_sum_buf, &kernel_arg.weights_sum_addr));
  RT_CHECK(vx_mem_alloc(device, image_size, VX_MEM_READ, &image_buf));
  RT_CHECK(vx_mem_address(image_buf, &kernel_arg.image_addr));
  RT_CHECK(vx_mem_alloc(device, ws_size, VX_MEM_READ, &grad_weights_sum_buf));
  RT_CHECK(vx_mem_address(grad_weights_sum_buf, &kernel_arg.grad_weights_sum_addr));
  RT_CHECK(vx_mem_alloc(device, image_size, VX_MEM_READ, &grad_image_buf));
  RT_CHECK(vx_mem_address(grad_image_buf, &kernel_arg.grad_image_addr));
  RT_CHECK(vx_mem_alloc(device, grad_sigmas_size, VX_MEM_WRITE, &grad_sigmas_buf));
  RT_CHECK(vx_mem_address(grad_sigmas_buf, &kernel_arg.grad_sigmas_addr));
  RT_CHECK(vx_mem_alloc(device, grad_rgbs_size, VX_MEM_WRITE, &grad_rgbs_buf));
  RT_CHECK(vx_mem_address(grad_rgbs_buf, &kernel_arg.grad_rgbs_addr));

  RT_CHECK(vx_copy_to_dev(sigmas_buf, h_sigmas.data(), 0, sigmas_size));
  RT_CHECK(vx_copy_to_dev(rgbs_buf, h_rgbs.data(), 0, rgbs_size));
  RT_CHECK(vx_copy_to_dev(deltas_buf, h_deltas.data(), 0, deltas_size));
  RT_CHECK(vx_copy_to_dev(weights_sum_buf, h_weights_sum.data(), 0, ws_size));
  RT_CHECK(vx_copy_to_dev(image_buf, h_image.data(), 0, image_size));
  RT_CHECK(vx_copy_to_dev(grad_weights_sum_buf, h_grad_ws.data(), 0, ws_size));
  RT_CHECK(vx_copy_to_dev(grad_image_buf, h_grad_image.data(), 0, image_size));

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // Read back results
  std::vector<float> h_grad_sigmas(total_samples, 0);
  std::vector<float> h_grad_rgbs(total_samples * 3, 0);

  RT_CHECK(vx_copy_from_dev(h_grad_sigmas.data(), grad_sigmas_buf, 0, grad_sigmas_size));
  RT_CHECK(vx_copy_from_dev(h_grad_rgbs.data(), grad_rgbs_buf, 0, grad_rgbs_size));

  // CPU reference backward
  int errors = 0;
  for (uint32_t ray = 0; ray < n_rays; ray++) {
    uint32_t base = ray * samples_per_ray;
    uint32_t base3 = ray * samples_per_ray * 3;

    float gi0 = h_grad_image[ray * 3 + 0];
    float gi1 = h_grad_image[ray * 3 + 1];
    float gi2 = h_grad_image[ray * 3 + 2];
    float gws = h_grad_ws[ray];

    float img_r = h_image[ray * 3 + 0];
    float img_g = h_image[ray * 3 + 1];
    float img_b = h_image[ray * 3 + 2];
    float ws_f  = h_weights_sum[ray];

    float T = 1.0f;
    float accum_r = 0, accum_g = 0, accum_b = 0;

    for (uint32_t step = 0; step < samples_per_ray; step++) {
      float sigma = h_sigmas[base + step];
      float delta = h_deltas[base + step];
      float alpha = 1.0f - std::exp(-sigma * delta);
      float weight = alpha * T;

      float c0 = h_rgbs[base3 + step * 3 + 0];
      float c1 = h_rgbs[base3 + step * 3 + 1];
      float c2 = h_rgbs[base3 + step * 3 + 2];

      accum_r += weight * c0;
      accum_g += weight * c1;
      accum_b += weight * c2;

      // Expected grad_rgb
      float exp_gc0 = gi0 * weight;
      float exp_gc1 = gi1 * weight;
      float exp_gc2 = gi2 * weight;

      // Expected grad_sigma
      float exp_gs = delta * (
          gi0 * (T * c0 - (img_r - accum_r)) +
          gi1 * (T * c1 - (img_g - accum_g)) +
          gi2 * (T * c2 - (img_b - accum_b)) +
          gws * (1.0f - ws_f)
      );

      // Check grad_sigmas
      if (std::fabs(h_grad_sigmas[base + step] - exp_gs) > 1e-3f) {
        if (errors < 10)
          printf("*** error: ray %d step %d grad_sigma: expected=%f, actual=%f\n",
                 ray, step, exp_gs, h_grad_sigmas[base + step]);
        ++errors;
      }

      // Check grad_rgbs
      float exp_gc[3] = {exp_gc0, exp_gc1, exp_gc2};
      for (int ch = 0; ch < 3; ch++) {
        if (std::fabs(h_grad_rgbs[base3 + step * 3 + ch] - exp_gc[ch]) > 1e-3f) {
          if (errors < 10)
            printf("*** error: ray %d step %d grad_rgb[%d]: expected=%f, actual=%f\n",
                   ray, step, ch, exp_gc[ch], h_grad_rgbs[base3 + step * 3 + ch]);
          ++errors;
        }
      }

      T *= (1.0f - alpha);
      if (T < T_thresh) break;
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
