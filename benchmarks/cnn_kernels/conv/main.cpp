// main_conv_layer.cpp

#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <vortex.h>
#include <cmath>
#include <cstdlib>

#include "common.h"

#define FLOAT_ULP 6

#define RT_CHECK(_expr)                                       \
  do {                                                        \
    int _ret = _expr;                                         \
    if (0 == _ret)                                            \
      break;                                                  \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);  \
    cleanup();                                                \
    exit(-1);                                                 \
  } while (false)

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

static const int C_IN  = 3;
static const int C_OUT = 4;
static const int K     = 3;

// -----------------------------------------------------------------------------
// Comparators
// -----------------------------------------------------------------------------

template <typename Type>
class Comparator {};

template <>
class Comparator<int> {
public:
  static const char* type_str() { return "integer"; }
  static int generate() { return rand(); }

  static bool compare(int a, int b, int index, int errors) {
    if (a != b) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%d, actual=%d\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<float> {
public:
  static const char* type_str() { return "float"; }

  static float generate() {
    return static_cast<float>(rand()) / RAND_MAX;
  }

  static bool compare(float a, float b, int index, int errors) {
    union {
      float  f;
      int32_t i;
    } fa, fb;

    fa.f = a;
    fb.f = b;

    auto d = std::abs(fa.i - fb.i);
    if (d > FLOAT_ULP) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%f, actual=%f\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

// -----------------------------------------------------------------------------
// CPU reference convolution (with bias)
// -----------------------------------------------------------------------------

static void convolution_cpu(
    TYPE*       O,
    const TYPE* I,
    const TYPE* W,
    const TYPE* B,
    int32_t     C_in,
    int32_t     C_out,
    int32_t     H,
    int32_t     Wt,
    int32_t     padding,
    int32_t     stride) {

  const int K = 3;
  int H_out = (H + 2 * padding - K) / stride + 1;
  int W_out = (Wt + 2 * padding - K) / stride + 1;

  for (int oc = 0; oc < C_out; ++oc) {
    for (int oy = 0; oy < H_out; ++oy) {
      for (int ox = 0; ox < W_out; ++ox) {

        TYPE sum = B[oc];

        for (int ic = 0; ic < C_in; ++ic) {
          for (int ky = 0; ky < K; ++ky) {
            for (int kx = 0; kx < K; ++kx) {
              int in_y = oy * stride + ky - padding;
              int in_x = ox * stride + kx - padding;

              if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < Wt) {
                int in_idx = ic * H * Wt + in_y * Wt + in_x;
                int wt_idx = ((oc * C_in + ic) * K + ky) * K + kx;
                sum += I[in_idx] * W[wt_idx];
              }
            }
          }
        }

        int out_idx = oc * H_out * W_out + oy * W_out + ox;
        O[out_idx] = sum;
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Globals
// -----------------------------------------------------------------------------

const char* kernel_file = "kernel.vxbin";
int size = 32;
bool use_lmem = false;

vx_device_h device     = nullptr;
vx_buffer_h I_buffer   = nullptr;
vx_buffer_h W_buffer   = nullptr;
vx_buffer_h B_buffer   = nullptr;
vx_buffer_h O_buffer   = nullptr;
vx_buffer_h krnl_buffer  = nullptr;
vx_buffer_h args_buffer  = nullptr;
kernel_arg_t kernel_arg = {};

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

static void show_usage() {
  std::cout << "Vortex Conv Layer Test\n";
  std::cout << "Usage: [-k kernel] [-l] [-n size] [-h]\n";
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:lh")) != -1) {
    switch (c) {
      case 'n': size = atoi(optarg); break;
      case 'l': use_lmem = true;     break;
      case 'k': kernel_file = optarg; break;
      case 'h':
      default:
        show_usage();
        exit(0);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(I_buffer);
    vx_mem_free(W_buffer);
    vx_mem_free(B_buffer);
    vx_mem_free(O_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
  parse_args(argc, argv);
  std::srand(50);

  RT_CHECK(vx_dev_open(&device));

  int padding = 1;
  int stride  = 1;
  int H  = size;
  int Wt = size;

  int H_out = (H  + 2 * padding - K) / stride + 1;
  int W_out = (Wt + 2 * padding - K) / stride + 1;

  kernel_arg = {};
  kernel_arg.C_in    = C_IN;
  kernel_arg.C_out   = C_OUT;
  kernel_arg.height  = H;
  kernel_arg.width   = Wt;
  kernel_arg.padding = padding;
  kernel_arg.stride  = stride;
  kernel_arg.H_out   = H_out;
  kernel_arg.W_out   = W_out;
  kernel_arg.use_lmem = use_lmem;

  kernel_arg.grid_dim[0] = W_out;
  kernel_arg.grid_dim[1] = H_out;
  kernel_arg.grid_dim[2] = C_OUT;

  kernel_arg.block_dim[0] = 1;
  kernel_arg.block_dim[1] = 1;
  kernel_arg.block_dim[2] = 1;

  uint32_t i_points = C_IN  * H     * Wt;
  uint32_t w_points = C_OUT * C_IN * K * K;
  uint32_t b_points = C_OUT;
  uint32_t o_points = C_OUT * H_out * W_out;

  RT_CHECK(vx_mem_alloc(device, i_points * sizeof(TYPE), VX_MEM_READ,  &I_buffer));
  RT_CHECK(vx_mem_address(I_buffer, &kernel_arg.I_addr));

  RT_CHECK(vx_mem_alloc(device, w_points * sizeof(TYPE), VX_MEM_READ,  &W_buffer));
  RT_CHECK(vx_mem_address(W_buffer, &kernel_arg.W_addr));

  RT_CHECK(vx_mem_alloc(device, b_points * sizeof(TYPE), VX_MEM_READ,  &B_buffer));
  RT_CHECK(vx_mem_address(B_buffer, &kernel_arg.B_addr));

  RT_CHECK(vx_mem_alloc(device, o_points * sizeof(TYPE), VX_MEM_WRITE, &O_buffer));
  RT_CHECK(vx_mem_address(O_buffer, &kernel_arg.O_addr));

  std::vector<TYPE> h_I(i_points);
  std::vector<TYPE> h_W(w_points);
  std::vector<TYPE> h_B(b_points);
  std::vector<TYPE> h_O(o_points);

  for (auto& v : h_I) v = Comparator<TYPE>::generate();
  for (auto& v : h_W) v = Comparator<TYPE>::generate();
  for (auto& v : h_B) v = Comparator<TYPE>::generate();

  RT_CHECK(vx_copy_to_dev(I_buffer, h_I.data(), 0, i_points * sizeof(TYPE)));
  RT_CHECK(vx_copy_to_dev(W_buffer, h_W.data(), 0, w_points * sizeof(TYPE)));
  RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, b_points * sizeof(TYPE)));

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  RT_CHECK(vx_copy_from_dev(h_O.data(), O_buffer, 0, o_points * sizeof(TYPE)));

  std::vector<TYPE> h_ref(o_points);
  convolution_cpu(
      h_ref.data(), h_I.data(), h_W.data(), h_B.data(),
      C_IN, C_OUT, H, Wt, padding, stride);

  int errors = 0;
  for (size_t i = 0; i < h_ref.size(); ++i) {
    if (!Comparator<TYPE>::compare(h_O[i], h_ref[i], i, errors))
      ++errors;
  }

  cleanup();

  if (errors) {
    std::cout << "FAILED with " << errors << " errors\n";
    return errors;
  }

  std::cout << "PASSED\n";
  return 0;
}
