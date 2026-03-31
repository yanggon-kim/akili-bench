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
uint32_t M = 16;  // seq_len (queries)
uint32_t N = 16;  // seq_len (keys)
uint32_t D = 16;  // head_dim

vx_device_h device = nullptr;
vx_buffer_h q_buffer = nullptr;
vx_buffer_h k_buffer = nullptr;
vx_buffer_h c_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex KB2 Attention Score (Q@K^T/sqrt(d)+Softmax)." << std::endl;
   std::cout << "Usage: [-k: kernel] [-m M(seq)] [-n N(seq)] [-d D(head_dim)] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "m:n:d:k:h")) != -1) {
    switch (c) {
    case 'm': M = atoi(optarg); break;
    case 'n': N = atoi(optarg); break;
    case 'd': D = atoi(optarg); break;
    case 'k': kernel_file = optarg; break;
    case 'h': show_usage(); exit(0); break;
    default: show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(q_buffer);
    vx_mem_free(k_buffer);
    vx_mem_free(c_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  std::srand(42);

  uint32_t q_size = M * D * sizeof(float);
  uint32_t k_size = N * D * sizeof(float);
  uint32_t c_size = M * N * sizeof(float);

  std::cout << "Attention Score: M=" << M << " N=" << N << " D=" << D << std::endl;

  RT_CHECK(vx_dev_open(&device));

  kernel_arg.M = M;
  kernel_arg.N = N;
  kernel_arg.D = D;

  RT_CHECK(vx_mem_alloc(device, q_size, VX_MEM_READ, &q_buffer));
  RT_CHECK(vx_mem_address(q_buffer, &kernel_arg.q_addr));
  RT_CHECK(vx_mem_alloc(device, k_size, VX_MEM_READ, &k_buffer));
  RT_CHECK(vx_mem_address(k_buffer, &kernel_arg.k_addr));
  RT_CHECK(vx_mem_alloc(device, c_size, VX_MEM_READ_WRITE, &c_buffer));
  RT_CHECK(vx_mem_address(c_buffer, &kernel_arg.c_addr));

  std::vector<float> h_q(M * D), h_k(N * D), h_c(M * N, 0);
  for (uint32_t i = 0; i < M * D; ++i)
    h_q[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;
  for (uint32_t i = 0; i < N * D; ++i)
    h_k[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;

  RT_CHECK(vx_copy_to_dev(q_buffer, h_q.data(), 0, q_size));
  RT_CHECK(vx_copy_to_dev(k_buffer, h_k.data(), 0, k_size));
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  RT_CHECK(vx_copy_from_dev(h_c.data(), c_buffer, 0, c_size));

  // Verify
  float inv_sqrt_d = 1.0f / sqrtf((float)D);
  int errors = 0;
  for (uint32_t i = 0; i < M; ++i) {
    // Compute Q[i] @ K^T / sqrt(d)
    std::vector<float> scores(N);
    for (uint32_t j = 0; j < N; ++j) {
      float dot = 0;
      for (uint32_t d = 0; d < D; ++d)
        dot += h_q[i * D + d] * h_k[j * D + d];
      scores[j] = dot * inv_sqrt_d;
    }
    // Softmax
    float max_val = scores[0];
    for (uint32_t j = 1; j < N; ++j)
      if (scores[j] > max_val) max_val = scores[j];
    float sum_exp = 0;
    for (uint32_t j = 0; j < N; ++j)
      sum_exp += expf(scores[j] - max_val);
    for (uint32_t j = 0; j < N; ++j) {
      float expected = expf(scores[j] - max_val) / sum_exp;
      float actual = h_c[i * N + j];
      union { float f; int32_t i; } fa, fb;
      fa.f = actual; fb.f = expected;
      if (std::abs(fa.i - fb.i) > FLOAT_ULP) {
        if (errors < 10)
          printf("*** error: [%d,%d] expected=%f, actual=%f\n", i, j, expected, actual);
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
