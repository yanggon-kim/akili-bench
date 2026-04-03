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
uint32_t M = 16;
uint32_t N = 16;
uint32_t K = 16;

vx_device_h device = nullptr;
vx_buffer_h a_buffer = nullptr;
vx_buffer_h w_buffer = nullptr;
vx_buffer_h bias_buffer = nullptr;
vx_buffer_h c_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex KBT Gemm+Mul+LeakyReLU (TCU) Benchmark." << std::endl;
   std::cout << "Usage: [-k: kernel] [-m M] [-n N] [-p K] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "m:n:p:k:h")) != -1) {
    switch (c) {
    case 'm': M = atoi(optarg); break;
    case 'n': N = atoi(optarg); break;
    case 'p': K = atoi(optarg); break;
    case 'k': kernel_file = optarg; break;
    case 'h': show_usage(); exit(0); break;
    default: show_usage(); exit(-1);
    }
  }
  M = ((M + 7) / 8) * 8;
  N = ((N + 7) / 8) * 8;
  K = ((K + 7) / 8) * 8;
}

void cleanup() {
  if (device) {
    vx_mem_free(a_buffer);
    vx_mem_free(w_buffer);
    vx_mem_free(bias_buffer);
    vx_mem_free(c_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  std::srand(42);

  float multiplier     = 1.5f;
  float negative_slope = 0.01f;

  uint32_t a_size    = M * K * sizeof(float);
  uint32_t w_size    = K * N * sizeof(float);
  uint32_t bias_size = N * sizeof(float);
  uint32_t c_size    = M * N * sizeof(float);

  std::cout << "TCU Gemm+Mul+LeakyReLU: M=" << M << " N=" << N << " K=" << K
            << " multiplier=" << multiplier << " negative_slope=" << negative_slope << std::endl;

  RT_CHECK(vx_dev_open(&device));

  uint64_t NT;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &NT));

  kernel_arg.grid_dim[0] = M / 8;
  kernel_arg.grid_dim[1] = 1;
  kernel_arg.grid_dim[2] = 1;
  kernel_arg.block_dim[0] = NT;
  kernel_arg.block_dim[1] = 1;
  kernel_arg.block_dim[2] = 1;
  kernel_arg.M = M;
  kernel_arg.N = N;
  kernel_arg.K = K;
  kernel_arg.multiplier     = multiplier;
  kernel_arg.negative_slope = negative_slope;

  RT_CHECK(vx_mem_alloc(device, a_size, VX_MEM_READ, &a_buffer));
  RT_CHECK(vx_mem_address(a_buffer, &kernel_arg.a_addr));
  RT_CHECK(vx_mem_alloc(device, w_size, VX_MEM_READ, &w_buffer));
  RT_CHECK(vx_mem_address(w_buffer, &kernel_arg.w_addr));
  RT_CHECK(vx_mem_alloc(device, bias_size, VX_MEM_READ, &bias_buffer));
  RT_CHECK(vx_mem_address(bias_buffer, &kernel_arg.bias_addr));
  RT_CHECK(vx_mem_alloc(device, c_size, VX_MEM_WRITE, &c_buffer));
  RT_CHECK(vx_mem_address(c_buffer, &kernel_arg.c_addr));

  std::vector<float> h_a(M * K), h_w(K * N), h_bias(N), h_c(M * N, 0);
  for (uint32_t i = 0; i < M * K; ++i)
    h_a[i] = 0.5f * static_cast<float>(rand()) / RAND_MAX - 0.25f;
  for (uint32_t i = 0; i < K * N; ++i)
    h_w[i] = 0.5f * static_cast<float>(rand()) / RAND_MAX - 0.25f;
  for (uint32_t i = 0; i < N; ++i)
    h_bias[i] = 0.1f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);

  RT_CHECK(vx_copy_to_dev(a_buffer, h_a.data(), 0, a_size));
  RT_CHECK(vx_copy_to_dev(w_buffer, h_w.data(), 0, w_size));
  RT_CHECK(vx_copy_to_dev(bias_buffer, h_bias.data(), 0, bias_size));
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  RT_CHECK(vx_copy_from_dev(h_c.data(), c_buffer, 0, c_size));

  // Verify
  int errors = 0;
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      float sum = h_bias[j];
      for (uint32_t k = 0; k < K; ++k)
        sum += h_a[i * K + k] * h_w[k * N + j];
      float val = sum * multiplier;
      float expected = (val >= 0.0f) ? val : val * negative_slope;
      float actual = h_c[i * N + j];
      float diff = fabsf(actual - expected);
      float rel = (fabsf(expected) > 1e-6f) ? diff / fabsf(expected) : diff;
      if (rel > 5e-2f) {
        if (errors < 10)
          printf("*** error: [%d,%d] expected=%f, actual=%f, rel=%e\n", i, j, expected, actual, rel);
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
