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
vx_buffer_h w_buffer = nullptr;
vx_buffer_h bias_buffer = nullptr;
vx_buffer_h c_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex KB2 Matmul+Scale+ResidAdd+Clamp+LogSumExp+Mish." << std::endl;
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

  float scale_factor = 2.0f;
  float clamp_min = -10.0f;
  float clamp_max = 10.0f;

  uint32_t a_size    = M * K * sizeof(float);
  uint32_t w_size    = K * N * sizeof(float);
  uint32_t bias_size = N * sizeof(float);
  uint32_t c_size    = M * sizeof(float);  // one scalar per row

  std::cout << "Matmul+Scale+ResidAdd+Clamp+LSE+Mish: M=" << M << " N=" << N << " K=" << K
            << " scale=" << scale_factor << " clamp=[" << clamp_min << "," << clamp_max << "]" << std::endl;

  RT_CHECK(vx_dev_open(&device));

  kernel_arg.M = M;
  kernel_arg.N = N;
  kernel_arg.K = K;
  kernel_arg.scale_factor = scale_factor;
  kernel_arg.clamp_min = clamp_min;
  kernel_arg.clamp_max = clamp_max;

  RT_CHECK(vx_mem_alloc(device, a_size, VX_MEM_READ, &a_buffer));
  RT_CHECK(vx_mem_address(a_buffer, &kernel_arg.a_addr));
  RT_CHECK(vx_mem_alloc(device, w_size, VX_MEM_READ, &w_buffer));
  RT_CHECK(vx_mem_address(w_buffer, &kernel_arg.w_addr));
  RT_CHECK(vx_mem_alloc(device, bias_size, VX_MEM_READ, &bias_buffer));
  RT_CHECK(vx_mem_address(bias_buffer, &kernel_arg.bias_addr));
  RT_CHECK(vx_mem_alloc(device, c_size, VX_MEM_WRITE, &c_buffer));
  RT_CHECK(vx_mem_address(c_buffer, &kernel_arg.c_addr));

  std::vector<float> h_a(M * K), h_w(K * N), h_bias(N), h_c(M, 0);
  for (uint32_t i = 0; i < M * K; ++i)
    h_a[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;
  for (uint32_t i = 0; i < K * N; ++i)
    h_w[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;
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
    // Compute matmul row -> scale -> resid add -> clamp
    std::vector<float> row_vals(N);
    for (uint32_t j = 0; j < N; ++j) {
      float dot = h_bias[j];
      for (uint32_t k = 0; k < K; ++k)
        dot += h_a[i * K + k] * h_w[k * N + j];
      dot *= scale_factor;
      dot += dot;  // x + x
      if (dot < clamp_min) dot = clamp_min;
      if (dot > clamp_max) dot = clamp_max;
      row_vals[j] = dot;
    }
    // LogSumExp
    float max_val = row_vals[0];
    for (uint32_t j = 1; j < N; ++j)
      if (row_vals[j] > max_val) max_val = row_vals[j];
    float sum_exp = 0;
    for (uint32_t j = 0; j < N; ++j)
      sum_exp += expf(row_vals[j] - max_val);
    float lse = max_val + logf(sum_exp);
    // x * mish(x) where mish(x) = x * tanh(softplus(x))
    float mish_lse = lse * tanhf(logf(1.0f + expf(lse)));
    float expected = lse * mish_lse;

    float actual = h_c[i];
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
