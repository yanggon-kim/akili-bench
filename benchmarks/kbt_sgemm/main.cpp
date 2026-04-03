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
vx_buffer_h b_buffer = nullptr;
vx_buffer_h c_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex KBT SGEMM (TCU) Benchmark." << std::endl;
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
  // Ensure multiples of 8
  M = ((M + 7) / 8) * 8;
  N = ((N + 7) / 8) * 8;
  K = ((K + 7) / 8) * 8;
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

  uint32_t a_size = M * K * sizeof(float);
  uint32_t b_size = K * N * sizeof(float);
  uint32_t c_size = M * N * sizeof(float);

  std::cout << "TCU SGEMM: M=" << M << " N=" << N << " K=" << K << std::endl;

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

  RT_CHECK(vx_mem_alloc(device, a_size, VX_MEM_READ, &a_buffer));
  RT_CHECK(vx_mem_address(a_buffer, &kernel_arg.a_addr));
  RT_CHECK(vx_mem_alloc(device, b_size, VX_MEM_READ, &b_buffer));
  RT_CHECK(vx_mem_address(b_buffer, &kernel_arg.b_addr));
  RT_CHECK(vx_mem_alloc(device, c_size, VX_MEM_WRITE, &c_buffer));
  RT_CHECK(vx_mem_address(c_buffer, &kernel_arg.c_addr));

  std::vector<float> h_a(M * K), h_b(K * N), h_c(M * N, 0);
  for (uint32_t i = 0; i < M * K; ++i)
    h_a[i] = 0.5f * static_cast<float>(rand()) / RAND_MAX - 0.25f;
  for (uint32_t i = 0; i < K * N; ++i)
    h_b[i] = 0.5f * static_cast<float>(rand()) / RAND_MAX - 0.25f;

  RT_CHECK(vx_copy_to_dev(a_buffer, h_a.data(), 0, a_size));
  RT_CHECK(vx_copy_to_dev(b_buffer, h_b.data(), 0, b_size));
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
      float expected = 0;
      for (uint32_t k = 0; k < K; ++k)
        expected += h_a[i * K + k] * h_b[k * N + j];
      float actual = h_c[i * N + j];
      float diff = fabsf(actual - expected);
      float rel = (fabsf(expected) > 1e-3f) ? diff / fabsf(expected) : diff;
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
