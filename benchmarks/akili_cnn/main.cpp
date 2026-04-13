// akili_cnn — SIMT 2D convolution benchmark.
//
// Shape-configurable via CLI:
//   -c C_in  -o C_out  -h H  -w W  -s K_size
// stride=1, padding=0 (valid). Output: (C_out, H-K+1, W-K+1).
// Prints KCYC[CONV,nt=<NT>]: <cycles> and PASSED!/FAILED!.

#include <iostream>
#include <unistd.h>
#include <string.h>
#include <cstring>
#include <vector>
#include <vortex.h>
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include "common.h"

#ifndef NUM_THREADS
#define NUM_THREADS 8
#endif

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret) break;                                      \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

static float frand() { return 2.0f * (float(rand()) / RAND_MAX) - 1.0f; }

static void conv2d_cpu(std::vector<float>& O,
                       const std::vector<float>& I,
                       const std::vector<float>& W,
                       uint32_t C_in, uint32_t C_out,
                       uint32_t H, uint32_t Wd, uint32_t K,
                       uint32_t H_out, uint32_t W_out) {
  O.assign((size_t)C_out * H_out * W_out, 0.0f);
  for (uint32_t oc = 0; oc < C_out; ++oc)
    for (uint32_t oy = 0; oy < H_out; ++oy)
      for (uint32_t ox = 0; ox < W_out; ++ox) {
        double sum = 0.0;
        for (uint32_t ic = 0; ic < C_in; ++ic)
          for (uint32_t ky = 0; ky < K; ++ky)
            for (uint32_t kx = 0; kx < K; ++kx) {
              uint32_t in_idx = ic * H * Wd + (oy + ky) * Wd + (ox + kx);
              uint32_t wt_idx = ((oc * C_in + ic) * K + ky) * K + kx;
              sum += (double)I[in_idx] * (double)W[wt_idx];
            }
        O[oc * H_out * W_out + oy * W_out + ox] = (float)sum;
      }
}

// ---------------------------------------------------------------------------

const char* kernel_file = "kernel.vxbin";
uint32_t C_in_req  = 1;
uint32_t C_out_req = 8;
uint32_t H_req     = 28;
uint32_t W_req     = 28;
uint32_t K_req     = 3;

vx_device_h device = nullptr;
vx_buffer_h I_buffer = nullptr;
vx_buffer_h W_buffer = nullptr;
vx_buffer_h O_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "akili_cnn — SIMT conv2d benchmark" << std::endl;
  std::cout << "Usage: [-c C_in] [-o C_out] [-h H] [-w W] [-s K_size]" << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "c:o:h:w:s:k:?")) != -1) {
    switch (c) {
      case 'c': C_in_req  = atoi(optarg); break;
      case 'o': C_out_req = atoi(optarg); break;
      case 'h': H_req     = atoi(optarg); break;
      case 'w': W_req     = atoi(optarg); break;
      case 's': K_req     = atoi(optarg); break;
      case 'k': kernel_file = optarg; break;
      default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (I_buffer)    vx_mem_free(I_buffer);
    if (W_buffer)    vx_mem_free(W_buffer);
    if (O_buffer)    vx_mem_free(O_buffer);
    if (krnl_buffer) vx_mem_free(krnl_buffer);
    if (args_buffer) vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

static void read_back_cycles() {
  kernel_arg_t back = {};
  vx_copy_from_dev(&back, args_buffer, 0, sizeof(kernel_arg_t));
  printf("KCYC[CONV,nt=%u]: %lu\n",
         (unsigned)NUM_THREADS, (unsigned long)back.kernel_cycles);
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);
  std::srand(50);

  RT_CHECK(vx_dev_open(&device));
  uint64_t NT_caps = 0;
  vx_dev_caps(device, VX_CAPS_NUM_THREADS, &NT_caps);
  if ((uint32_t)NT_caps != NUM_THREADS) {
    printf("Error: device NUM_THREADS=%u but built with NUM_THREADS=%u\n",
           (unsigned)NT_caps, (unsigned)NUM_THREADS);
    cleanup();
    return -1;
  }

  uint32_t C_in  = C_in_req;
  uint32_t C_out = C_out_req;
  uint32_t H     = H_req;
  uint32_t Wd    = W_req;
  uint32_t K     = K_req;
  uint32_t H_out = H - K + 1;
  uint32_t W_out = Wd - K + 1;
  uint32_t N_out = H_out * W_out;

  std::cout << "akili_cnn — C_in=" << C_in << " C_out=" << C_out
            << " H=" << H << " W=" << Wd << " K=" << K
            << " → H_out=" << H_out << " W_out=" << W_out << std::endl;

  std::vector<float> h_I((size_t)C_in * H * Wd);
  std::vector<float> h_W((size_t)C_out * C_in * K * K);
  for (auto& v : h_I) v = frand();
  for (auto& v : h_W) v = frand();

  std::vector<float> h_O_ref;
  conv2d_cpu(h_O_ref, h_I, h_W, C_in, C_out, H, Wd, K, H_out, W_out);

  size_t I_bytes = h_I.size() * sizeof(float);
  size_t W_bytes = h_W.size() * sizeof(float);
  size_t O_bytes = (size_t)C_out * H_out * W_out * sizeof(float);

  RT_CHECK(vx_mem_alloc(device, I_bytes, VX_MEM_READ_WRITE, &I_buffer));
  RT_CHECK(vx_mem_address(I_buffer, &kernel_arg.I_addr));
  RT_CHECK(vx_mem_alloc(device, W_bytes, VX_MEM_READ_WRITE, &W_buffer));
  RT_CHECK(vx_mem_address(W_buffer, &kernel_arg.W_addr));
  RT_CHECK(vx_mem_alloc(device, O_bytes, VX_MEM_READ_WRITE, &O_buffer));
  RT_CHECK(vx_mem_address(O_buffer, &kernel_arg.O_addr));
  RT_CHECK(vx_copy_to_dev(I_buffer, h_I.data(), 0, I_bytes));
  RT_CHECK(vx_copy_to_dev(W_buffer, h_W.data(), 0, W_bytes));

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_mem_alloc(device, sizeof(kernel_arg_t), VX_MEM_READ_WRITE, &args_buffer));

  kernel_arg.C_in  = C_in;
  kernel_arg.C_out = C_out;
  kernel_arg.H     = H;
  kernel_arg.W     = Wd;
  kernel_arg.K_sz  = K;
  kernel_arg.H_out = H_out;
  kernel_arg.W_out = W_out;
  kernel_arg.grid_dim[0] = N_out;
  kernel_arg.grid_dim[1] = C_out;
  kernel_arg.block_dim[0] = 0;
  kernel_arg.block_dim[1] = 0;
  RT_CHECK(vx_copy_to_dev(args_buffer, &kernel_arg, 0, sizeof(kernel_arg_t)));
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  read_back_cycles();

  std::vector<float> h_O((size_t)C_out * H_out * W_out);
  RT_CHECK(vx_copy_from_dev(h_O.data(), O_buffer, 0, O_bytes));

  int errors = 0;
  const float atol = 1e-3f;
  const float rtol = 1e-3f;
  for (size_t i = 0; i < h_O_ref.size(); ++i) {
    float a = h_O_ref[i];
    float b = h_O[i];
    float diff = std::fabs(a - b);
    float lim  = atol + rtol * std::fabs(a);
    if (diff > lim) {
      if (errors < 10)
        printf("*** O error: [%zu] expected=%f actual=%f\n", i, a, b);
      ++errors;
    }
  }

  cleanup();
  if (errors) { printf("Found %d errors\nFAILED!\n", errors); return errors; }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
