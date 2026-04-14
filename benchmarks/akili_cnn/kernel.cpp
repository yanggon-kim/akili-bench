#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

// =============================================================================
// akili_cnn — SIMT 2D convolution on the KMU launch API.
//
// Grid: {ceil(H_out*W_out / NUM_THREADS), C_out}
// Block: {NUM_THREADS, 1}
// Each CTA has NUM_THREADS threads. Thread `t` in block (bx, by) handles
// spatial position `bx * NUM_THREADS + t` on output channel `by`.
// =============================================================================
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  __rdcycle_time t0 = vx_rdcycle_sync_begin();

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
  int sp_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N_out  = (int)(H_out * W_out);

  if (sp_idx < N_out && oc < (int)arg->C_out) {
    int oy = sp_idx / (int)W_out;
    int ox = sp_idx % (int)W_out;

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

  __rdcycle_time t1 = vx_rdcycle_sync_end();

  // Only thread 0 of each CTA writes its block's cycle count.
  if (threadIdx.x == 0) {
    auto pCycles = reinterpret_cast<uint32_t*>(arg->cycles_addr);
    uint32_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
    pCycles[block_id] = (uint32_t)vx_rdcycle_sync_diff(t0, t1);
  }
}
