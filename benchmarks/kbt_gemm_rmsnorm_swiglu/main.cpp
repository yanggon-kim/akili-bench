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
vx_buffer_h x_buffer = nullptr;
vx_buffer_h rms_w_buffer = nullptr;
vx_buffer_h wg_buffer = nullptr;
vx_buffer_h wu_buffer = nullptr;
vx_buffer_h norm_buffer = nullptr;
vx_buffer_h gate_buffer = nullptr;
vx_buffer_h c_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex KBT GEMM+RMSNorm+SwiGLU (TCU) Benchmark." << std::endl;
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
    vx_mem_free(x_buffer);
    vx_mem_free(rms_w_buffer);
    vx_mem_free(wg_buffer);
    vx_mem_free(wu_buffer);
    vx_mem_free(norm_buffer);
    vx_mem_free(gate_buffer);
    vx_mem_free(c_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  std::srand(42);

  float eps = 1e-5f;

  uint32_t x_size    = M * K * sizeof(float);
  uint32_t rms_w_size = K * sizeof(float);
  uint32_t w_size    = K * N * sizeof(float);
  uint32_t norm_size = M * K * sizeof(float);
  uint32_t gate_size = M * N * sizeof(float);
  uint32_t c_size    = M * N * sizeof(float);

  std::cout << "TCU GEMM+RMSNorm+SwiGLU: M=" << M << " N=" << N << " K=" << K << std::endl;

  RT_CHECK(vx_dev_open(&device));

  uint64_t NT;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &NT));

  kernel_arg.grid_dim[0]  = M / 8;
  kernel_arg.grid_dim[1]  = 1;
  kernel_arg.grid_dim[2]  = 1;
  kernel_arg.block_dim[0] = NT;
  kernel_arg.block_dim[1] = 1;
  kernel_arg.block_dim[2] = 1;
  kernel_arg.M   = M;
  kernel_arg.N   = N;
  kernel_arg.K   = K;
  kernel_arg.eps = eps;

  RT_CHECK(vx_mem_alloc(device, x_size, VX_MEM_READ, &x_buffer));
  RT_CHECK(vx_mem_address(x_buffer, &kernel_arg.x_addr));
  RT_CHECK(vx_mem_alloc(device, rms_w_size, VX_MEM_READ, &rms_w_buffer));
  RT_CHECK(vx_mem_address(rms_w_buffer, &kernel_arg.rms_w_addr));
  RT_CHECK(vx_mem_alloc(device, w_size, VX_MEM_READ, &wg_buffer));
  RT_CHECK(vx_mem_address(wg_buffer, &kernel_arg.wg_addr));
  RT_CHECK(vx_mem_alloc(device, w_size, VX_MEM_READ, &wu_buffer));
  RT_CHECK(vx_mem_address(wu_buffer, &kernel_arg.wu_addr));
  RT_CHECK(vx_mem_alloc(device, norm_size, VX_MEM_READ_WRITE, &norm_buffer));
  RT_CHECK(vx_mem_address(norm_buffer, &kernel_arg.norm_addr));
  RT_CHECK(vx_mem_alloc(device, gate_size, VX_MEM_READ_WRITE, &gate_buffer));
  RT_CHECK(vx_mem_address(gate_buffer, &kernel_arg.gate_addr));
  RT_CHECK(vx_mem_alloc(device, c_size, VX_MEM_READ_WRITE, &c_buffer));
  RT_CHECK(vx_mem_address(c_buffer, &kernel_arg.c_addr));

  std::vector<float> h_x(M * K), h_rms_w(K);
  std::vector<float> h_wg(K * N), h_wu(K * N);
  std::vector<float> h_c(M * N, 0);

  for (uint32_t i = 0; i < M * K; ++i)
    h_x[i] = 0.5f * static_cast<float>(rand()) / RAND_MAX - 0.25f;
  for (uint32_t i = 0; i < K; ++i)
    h_rms_w[i] = 0.5f * static_cast<float>(rand()) / RAND_MAX + 0.5f;
  for (uint32_t i = 0; i < K * N; ++i)
    h_wg[i] = 0.5f * static_cast<float>(rand()) / RAND_MAX - 0.25f;
  for (uint32_t i = 0; i < K * N; ++i)
    h_wu[i] = 0.5f * static_cast<float>(rand()) / RAND_MAX - 0.25f;

  RT_CHECK(vx_copy_to_dev(x_buffer, h_x.data(), 0, x_size));
  RT_CHECK(vx_copy_to_dev(rms_w_buffer, h_rms_w.data(), 0, rms_w_size));
  RT_CHECK(vx_copy_to_dev(wg_buffer, h_wg.data(), 0, w_size));
  RT_CHECK(vx_copy_to_dev(wu_buffer, h_wu.data(), 0, w_size));
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  RT_CHECK(vx_copy_from_dev(h_c.data(), c_buffer, 0, c_size));

  // Verify: RMSNorm, gate = normalized@Wg, up = normalized@Wu, silu(gate)*up
  std::vector<float> ref_norm(M * K, 0);
  for (uint32_t i = 0; i < M; ++i) {
    float ss = 0;
    for (uint32_t k = 0; k < K; ++k) {
      float v = h_x[i * K + k];
      ss += v * v;
    }
    float inv_rms = 1.0f / sqrtf(ss / (float)K + eps);
    for (uint32_t k = 0; k < K; ++k)
      ref_norm[i * K + k] = h_x[i * K + k] * inv_rms * h_rms_w[k];
  }

  std::vector<float> ref_gate(M * N, 0), ref_up(M * N, 0);
  for (uint32_t i = 0; i < M; ++i)
    for (uint32_t j = 0; j < N; ++j) {
      float sum_g = 0, sum_u = 0;
      for (uint32_t k = 0; k < K; ++k) {
        sum_g += ref_norm[i * K + k] * h_wg[k * N + j];
        sum_u += ref_norm[i * K + k] * h_wu[k * N + j];
      }
      ref_gate[i * N + j] = sum_g;
      ref_up[i * N + j]   = sum_u;
    }

  int errors = 0;
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      float g = ref_gate[i * N + j];
      float u = ref_up[i * N + j];
      float silu_g = g / (1.0f + expf(-g));
      float expected = silu_g * u;
      float actual   = h_c[i * N + j];
      float diff = fabsf(actual - expected);
      if (diff > 1e-1f) {
        if (errors < 10)
          printf("*** error: [%d,%d] expected=%f, actual=%f, diff=%e\n", i, j, expected, actual, diff);
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
