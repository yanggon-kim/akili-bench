#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"

// =============================================================================
// akili_cnn — SIMT 2D convolution. One thread per output element.
// grid = {H_out*W_out, C_out}. Each thread loops over C_in × K × K.
// =============================================================================
void kernel_conv_simt(kernel_arg_t* __UNIFORM__ arg) {
  auto I = reinterpret_cast<float*>(arg->I_addr);
  auto W = reinterpret_cast<float*>(arg->W_addr);
  auto O = reinterpret_cast<float*>(arg->O_addr);

  uint32_t C_in  = arg->C_in;
  uint32_t H     = arg->H;
  uint32_t Wd    = arg->W;
  uint32_t K     = arg->K_sz;
  uint32_t H_out = arg->H_out;
  uint32_t W_out = arg->W_out;

  int oc     = blockIdx.y;
  int sp_idx = blockIdx.x;
  int oy = sp_idx / (int)W_out;
  int ox = sp_idx % (int)W_out;

  if (oy >= (int)H_out || ox >= (int)W_out) return;

  float sum = 0.0f;
  uint32_t wt_base = oc * C_in * K * K;
  for (uint32_t ic = 0; ic < C_in; ++ic) {
    for (uint32_t ky = 0; ky < K; ++ky) {
      for (uint32_t kx = 0; kx < K; ++kx) {
        int in_y = oy + (int)ky;
        int in_x = ox + (int)kx;
        if (in_y >= 0 && in_y < (int)H && in_x >= 0 && in_x < (int)Wd) {
          uint32_t in_idx = ic * H * Wd + in_y * Wd + in_x;
          uint32_t wt_idx = wt_base + (ic * K + ky) * K + kx;
          sum += I[in_idx] * W[wt_idx];
        }
      }
    }
  }
  O[oc * H_out * W_out + oy * W_out + ox] = sum;
}

int main() {
  kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  uint64_t t_begin = vx_rdcycle();
  int rc = vx_spawn_threads(2, arg->grid_dim, nullptr,
                            (vx_kernel_func_cb)kernel_conv_simt, arg);
  uint64_t t_end = vx_rdcycle();
  arg->kernel_cycles = t_end - t_begin;
  return rc;
}
