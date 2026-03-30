#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <cmath>
#include <vortex.h>
#include "common.h"

#define FLOAT_ULP 30  // RoPE has sin/cos/pow chain — needs more tolerance

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
uint32_t num_tokens = 32;
uint32_t head_dim = 16;
uint32_t num_heads = 4;
float theta_base = 10000.0f;

vx_device_h device = nullptr;
vx_buffer_h x_buffer = nullptr;
vx_buffer_h y_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex RoPE Benchmark (Triton-style)." << std::endl;
   std::cout << "Usage: [-k: kernel] [-t tokens] [-d head_dim] [-n heads] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "t:d:n:k:h")) != -1) {
    switch (c) {
    case 't': num_tokens = atoi(optarg); break;
    case 'd': head_dim = atoi(optarg); break;
    case 'n': num_heads = atoi(optarg); break;
    case 'k': kernel_file = optarg; break;
    case 'h': show_usage(); exit(0); break;
    default: show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(x_buffer);
    vx_mem_free(y_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  std::srand(42);

  uint32_t total_elems = num_tokens * num_heads * head_dim;
  uint32_t buf_size = total_elems * sizeof(float);

  std::cout << "RoPE: tokens=" << num_tokens << " heads=" << num_heads
            << " head_dim=" << head_dim << " theta=" << theta_base << std::endl;

  RT_CHECK(vx_dev_open(&device));

  kernel_arg.num_tokens = num_tokens;
  kernel_arg.head_dim = head_dim;
  kernel_arg.num_heads = num_heads;
  kernel_arg.theta_base = theta_base;

  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &x_buffer));
  RT_CHECK(vx_mem_address(x_buffer, &kernel_arg.x_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_WRITE, &y_buffer));
  RT_CHECK(vx_mem_address(y_buffer, &kernel_arg.y_addr));

  std::vector<float> h_x(total_elems), h_y(total_elems, 0);
  for (uint32_t i = 0; i < total_elems; ++i)
    h_x[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;

  RT_CHECK(vx_copy_to_dev(x_buffer, h_x.data(), 0, buf_size));
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  RT_CHECK(vx_copy_from_dev(h_y.data(), y_buffer, 0, buf_size));

  // Verify
  int errors = 0;
  uint32_t half_dim = head_dim / 2;
  uint32_t stride = num_heads * head_dim;
  for (uint32_t t = 0; t < num_tokens; ++t) {
    float position = (float)t;
    for (uint32_t h = 0; h < num_heads; ++h) {
      uint32_t off = t * stride + h * head_dim;
      for (uint32_t i = 0; i < half_dim; ++i) {
        float freq = 1.0f / powf(theta_base, (2.0f * i) / head_dim);
        float theta = position * freq;
        float cos_t = cosf(theta), sin_t = sinf(theta);
        float x0 = h_x[off + i], x1 = h_x[off + i + half_dim];
        float ey0 = x0 * cos_t - x1 * sin_t;
        float ey1 = x0 * sin_t + x1 * cos_t;

        float ay0 = h_y[off + i];
        float ay1 = h_y[off + i + half_dim];
        float rel0 = (fabsf(ey0) > 1e-6f) ? fabsf((ay0 - ey0) / ey0) : fabsf(ay0 - ey0);
        float rel1 = (fabsf(ey1) > 1e-6f) ? fabsf((ay1 - ey1) / ey1) : fabsf(ay1 - ey1);
        if (rel0 > 1e-3f) {
          if (errors < 10)
            printf("*** error: [t%d,h%d,%d] expected=%f, actual=%f, rel=%e\n", t, h, i, ey0, ay0, rel0);
          ++errors;
        }
        if (rel1 > 1e-3f) {
          if (errors < 10)
            printf("*** error: [t%d,h%d,%d+half] expected=%f, actual=%f, rel=%e\n", t, h, i, ey1, ay1, rel1);
          ++errors;
        }
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
