#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <cmath>
#include <vortex.h>
#include "common.h"

#define FLOAT_ULP 20

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
vx_buffer_h wa_buffer = nullptr;
vx_buffer_h ba_buffer = nullptr;
vx_buffer_h wb_buffer = nullptr;
vx_buffer_h bb_buffer = nullptr;
vx_buffer_h c_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex KB2 SwiGLU (Gemm+SiLU+Mul) Benchmark." << std::endl;
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
}

void cleanup() {
  if (device) {
    vx_mem_free(a_buffer);
    vx_mem_free(wa_buffer);
    vx_mem_free(ba_buffer);
    vx_mem_free(wb_buffer);
    vx_mem_free(bb_buffer);
    vx_mem_free(c_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  std::srand(42);

  uint32_t a_size    = M * K * sizeof(float);
  uint32_t w_size    = K * N * sizeof(float);
  uint32_t bias_size = N * sizeof(float);
  uint32_t c_size    = M * N * sizeof(float);

  std::cout << "SwiGLU: M=" << M << " N=" << N << " K=" << K << std::endl;

  RT_CHECK(vx_dev_open(&device));

  kernel_arg.M = M;
  kernel_arg.N = N;
  kernel_arg.K = K;

  RT_CHECK(vx_mem_alloc(device, a_size, VX_MEM_READ, &a_buffer));
  RT_CHECK(vx_mem_address(a_buffer, &kernel_arg.a_addr));
  RT_CHECK(vx_mem_alloc(device, w_size, VX_MEM_READ, &wa_buffer));
  RT_CHECK(vx_mem_address(wa_buffer, &kernel_arg.wa_addr));
  RT_CHECK(vx_mem_alloc(device, bias_size, VX_MEM_READ, &ba_buffer));
  RT_CHECK(vx_mem_address(ba_buffer, &kernel_arg.ba_addr));
  RT_CHECK(vx_mem_alloc(device, w_size, VX_MEM_READ, &wb_buffer));
  RT_CHECK(vx_mem_address(wb_buffer, &kernel_arg.wb_addr));
  RT_CHECK(vx_mem_alloc(device, bias_size, VX_MEM_READ, &bb_buffer));
  RT_CHECK(vx_mem_address(bb_buffer, &kernel_arg.bb_addr));
  RT_CHECK(vx_mem_alloc(device, c_size, VX_MEM_WRITE, &c_buffer));
  RT_CHECK(vx_mem_address(c_buffer, &kernel_arg.c_addr));

  std::vector<float> h_a(M * K), h_wa(K * N), h_ba(N), h_wb(K * N), h_bb(N), h_c(M * N, 0);
  for (uint32_t i = 0; i < M * K; ++i)
    h_a[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;
  for (uint32_t i = 0; i < K * N; ++i)
    h_wa[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;
  for (uint32_t i = 0; i < N; ++i)
    h_ba[i] = 0.1f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
  for (uint32_t i = 0; i < K * N; ++i)
    h_wb[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;
  for (uint32_t i = 0; i < N; ++i)
    h_bb[i] = 0.1f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);

  RT_CHECK(vx_copy_to_dev(a_buffer, h_a.data(), 0, a_size));
  RT_CHECK(vx_copy_to_dev(wa_buffer, h_wa.data(), 0, w_size));
  RT_CHECK(vx_copy_to_dev(ba_buffer, h_ba.data(), 0, bias_size));
  RT_CHECK(vx_copy_to_dev(wb_buffer, h_wb.data(), 0, w_size));
  RT_CHECK(vx_copy_to_dev(bb_buffer, h_bb.data(), 0, bias_size));
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
      // Gate path
      float gate = h_ba[j];
      for (uint32_t k = 0; k < K; ++k)
        gate += h_a[i * K + k] * h_wa[k * N + j];
      // Up path
      float up = h_bb[j];
      for (uint32_t k = 0; k < K; ++k)
        up += h_a[i * K + k] * h_wb[k * N + j];
      // SiLU(gate) * up
      float silu_gate = gate / (1.0f + expf(-gate));
      float expected = silu_gate * up;
      float actual = h_c[i * N + j];
      float rel = (fabsf(expected) > 1e-6f) ? fabsf((actual - expected) / expected) : fabsf(actual - expected);
      
      if (rel > 1e-3f) {
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
