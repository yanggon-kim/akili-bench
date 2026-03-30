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
vx_buffer_h a_buffer = nullptr;
vx_buffer_h b_buffer = nullptr;
vx_buffer_h c_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex SwiGLU Benchmark (Triton-style)." << std::endl;
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
    vx_mem_free(a_buffer);
    vx_mem_free(b_buffer);
    vx_mem_free(c_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  std::srand(42);

  uint32_t num_points = num_rows * num_cols;
  uint32_t buf_size = num_points * sizeof(float);

  std::cout << "SwiGLU: rows=" << num_rows << " cols=" << num_cols << std::endl;

  RT_CHECK(vx_dev_open(&device));

  kernel_arg.num_rows = num_rows;
  kernel_arg.num_cols = num_cols;

  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &a_buffer));
  RT_CHECK(vx_mem_address(a_buffer, &kernel_arg.a_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &b_buffer));
  RT_CHECK(vx_mem_address(b_buffer, &kernel_arg.b_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_WRITE, &c_buffer));
  RT_CHECK(vx_mem_address(c_buffer, &kernel_arg.c_addr));

  // Generate inputs
  std::vector<float> h_a(num_points), h_b(num_points), h_c(num_points, 0);
  for (uint32_t i = 0; i < num_points; ++i) {
    h_a[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;  // [-1, 1]
    h_b[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;
  }

  RT_CHECK(vx_copy_to_dev(a_buffer, h_a.data(), 0, buf_size));
  RT_CHECK(vx_copy_to_dev(b_buffer, h_b.data(), 0, buf_size));
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  RT_CHECK(vx_copy_from_dev(h_c.data(), c_buffer, 0, buf_size));

  // Verify: c = silu(a) * b
  int errors = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    float sig = 1.0f / (1.0f + expf(-h_a[i]));
    float silu = h_a[i] * sig;
    float expected = silu * h_b[i];
    union { float f; int32_t i; } fa, fb;
    fa.f = h_c[i]; fb.f = expected;
    if (std::abs(fa.i - fb.i) > FLOAT_ULP) {
      if (errors < 10)
        printf("*** error: [%d] expected=%f, actual=%f\n", i, expected, h_c[i]);
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
