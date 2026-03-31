#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <cmath>
#include <vortex.h>
#include "common.h"

#define FLOAT_ULP 40  // dot product accumulates K products — needs wide tolerance

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
uint32_t M = 32;
uint32_t K = 32;

vx_device_h device = nullptr;
vx_buffer_h a_buffer = nullptr;
vx_buffer_h x_buffer = nullptr;
vx_buffer_h y_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex KB MatVec Benchmark." << std::endl;
   std::cout << "Usage: [-k: kernel] [-m M] [-p K] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "m:p:k:h")) != -1) {
    switch (c) {
    case 'm': M = atoi(optarg); break;
    case 'p': K = atoi(optarg); break;
    case 'k': kernel_file = optarg; break;
    case 'h': show_usage(); exit(0); break;
    default: show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(a_buffer);
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

  uint32_t a_size = M * K * sizeof(float);
  uint32_t x_size = K * sizeof(float);
  uint32_t y_size = M * sizeof(float);

  std::cout << "MatVec: M=" << M << " K=" << K << std::endl;

  RT_CHECK(vx_dev_open(&device));

  kernel_arg.M = M;
  kernel_arg.K = K;

  RT_CHECK(vx_mem_alloc(device, a_size, VX_MEM_READ, &a_buffer));
  RT_CHECK(vx_mem_address(a_buffer, &kernel_arg.a_addr));
  RT_CHECK(vx_mem_alloc(device, x_size, VX_MEM_READ, &x_buffer));
  RT_CHECK(vx_mem_address(x_buffer, &kernel_arg.x_addr));
  RT_CHECK(vx_mem_alloc(device, y_size, VX_MEM_WRITE, &y_buffer));
  RT_CHECK(vx_mem_address(y_buffer, &kernel_arg.y_addr));

  std::vector<float> h_a(M * K), h_x(K), h_y(M, 0);
  for (uint32_t i = 0; i < M * K; ++i)
    h_a[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;
  for (uint32_t i = 0; i < K; ++i)
    h_x[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;

  RT_CHECK(vx_copy_to_dev(a_buffer, h_a.data(), 0, a_size));
  RT_CHECK(vx_copy_to_dev(x_buffer, h_x.data(), 0, x_size));
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  RT_CHECK(vx_copy_from_dev(h_y.data(), y_buffer, 0, y_size));

  // Verify
  int errors = 0;
  for (uint32_t i = 0; i < M; ++i) {
    float expected = 0;
    for (uint32_t k = 0; k < K; ++k)
      expected += h_a[i * K + k] * h_x[k];
    float actual = h_y[i];
    union { float f; int32_t i; } fa, fb;
    fa.f = actual; fb.f = expected;
    if (std::abs(fa.i - fb.i) > FLOAT_ULP) {
      if (errors < 10)
        printf("*** error: [%d] expected=%f, actual=%f\n", i, expected, actual);
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
