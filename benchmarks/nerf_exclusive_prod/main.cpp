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
uint32_t samples_per_ray = 16;

vx_device_h device = nullptr;
vx_buffer_h input_buffer = nullptr;
vx_buffer_h output_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex NeRF Exclusive Product Benchmark." << std::endl;
   std::cout << "Usage: [-k: kernel] [-r rays] [-s samples] [-h: help]" << std::endl;
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
    vx_mem_free(input_buffer);
    vx_mem_free(output_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  std::srand(42);

  uint32_t total = n_rays * samples_per_ray;
  uint32_t buf_size = total * sizeof(float);

  std::cout << "Exclusive Product: n_rays=" << n_rays
            << " samples_per_ray=" << samples_per_ray << std::endl;

  // Generate transmittance-like input values in [0.9, 1.0]
  std::vector<float> h_input(total);
  for (uint32_t i = 0; i < total; ++i)
    h_input[i] = 0.9f + 0.1f * static_cast<float>(rand()) / RAND_MAX;

  RT_CHECK(vx_dev_open(&device));

  kernel_arg.num_tasks = n_rays;
  kernel_arg.samples_per_ray = samples_per_ray;

  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &input_buffer));
  RT_CHECK(vx_mem_address(input_buffer, &kernel_arg.input_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_WRITE, &output_buffer));
  RT_CHECK(vx_mem_address(output_buffer, &kernel_arg.output_addr));

  RT_CHECK(vx_copy_to_dev(input_buffer, h_input.data(), 0, buf_size));
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  std::vector<float> h_output(total, 0);
  RT_CHECK(vx_copy_from_dev(h_output.data(), output_buffer, 0, buf_size));

  // CPU reference: exclusive prefix product per ray
  int errors = 0;
  for (uint32_t r = 0; r < n_rays; ++r) {
    uint32_t offset = r * samples_per_ray;
    float accum = 1.0f;
    for (uint32_t s = 0; s < samples_per_ray; ++s) {
      float expected = accum;
      float actual = h_output[offset + s];
      // Use relative error for accumulated products
      float diff = std::fabs(expected - actual);
      float mag = std::fmax(std::fabs(expected), std::fabs(actual));
      if (mag > 0.0f && diff / mag > 1e-4f) {
        if (errors < 10)
          printf("*** error: ray[%u][%u] expected=%f, actual=%f\n", r, s, expected, actual);
        ++errors;
      }
      accum *= h_input[offset + s];
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
