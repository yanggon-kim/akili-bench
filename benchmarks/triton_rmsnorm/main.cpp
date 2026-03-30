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
uint32_t num_rows = 64;
uint32_t num_cols = 64;

vx_device_h device = nullptr;
vx_buffer_h x_buffer = nullptr;
vx_buffer_h w_buffer = nullptr;
vx_buffer_h y_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex RMSNorm Benchmark (Triton-style)." << std::endl;
   std::cout << "Usage: [-k: kernel] [-r rows] [-c cols] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "r:c:k:h")) != -1) {
    switch (c) {
    case 'r': num_rows = atoi(optarg); break;
    case 'c': num_cols = atoi(optarg); break;
    case 'k': kernel_file = optarg; break;
    case 'h': show_usage(); exit(0); break;
    default: show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(x_buffer);
    vx_mem_free(w_buffer);
    vx_mem_free(y_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  std::srand(42);

  uint32_t num_points = num_rows * num_cols;
  uint32_t mat_size = num_points * sizeof(float);
  uint32_t vec_size = num_cols * sizeof(float);
  float eps = 1e-6f;

  std::cout << "RMSNorm: rows=" << num_rows << " cols=" << num_cols << " eps=" << eps << std::endl;

  RT_CHECK(vx_dev_open(&device));

  kernel_arg.num_rows = num_rows;
  kernel_arg.num_cols = num_cols;
  kernel_arg.eps = eps;

  RT_CHECK(vx_mem_alloc(device, mat_size, VX_MEM_READ, &x_buffer));
  RT_CHECK(vx_mem_address(x_buffer, &kernel_arg.x_addr));
  RT_CHECK(vx_mem_alloc(device, vec_size, VX_MEM_READ, &w_buffer));
  RT_CHECK(vx_mem_address(w_buffer, &kernel_arg.w_addr));
  RT_CHECK(vx_mem_alloc(device, mat_size, VX_MEM_WRITE, &y_buffer));
  RT_CHECK(vx_mem_address(y_buffer, &kernel_arg.y_addr));

  std::vector<float> h_x(num_points), h_w(num_cols), h_y(num_points, 0);
  for (uint32_t i = 0; i < num_points; ++i)
    h_x[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;
  for (uint32_t i = 0; i < num_cols; ++i)
    h_w[i] = 0.5f + static_cast<float>(rand()) / RAND_MAX;

  RT_CHECK(vx_copy_to_dev(x_buffer, h_x.data(), 0, mat_size));
  RT_CHECK(vx_copy_to_dev(w_buffer, h_w.data(), 0, vec_size));
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  RT_CHECK(vx_copy_from_dev(h_y.data(), y_buffer, 0, mat_size));

  // Verify: y = x * rsqrt(mean(x^2) + eps) * w
  int errors = 0;
  for (uint32_t r = 0; r < num_rows; ++r) {
    float sum_sq = 0;
    for (uint32_t c = 0; c < num_cols; ++c) {
      float v = h_x[r * num_cols + c];
      sum_sq += v * v;
    }
    float rstd = 1.0f / sqrtf(sum_sq / num_cols + eps);
    for (uint32_t c = 0; c < num_cols; ++c) {
      float expected = h_x[r * num_cols + c] * rstd * h_w[c];
      float actual = h_y[r * num_cols + c];
      union { float f; int32_t i; } fa, fb;
      fa.f = actual; fb.f = expected;
      if (std::abs(fa.i - fb.i) > FLOAT_ULP) {
        if (errors < 10)
          printf("*** error: [%d,%d] expected=%f, actual=%f\n", r, c, expected, actual);
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
