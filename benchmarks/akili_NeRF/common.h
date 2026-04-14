#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

// Kernel stage IDs — three SIMT kernels dispatched sequentially from host.
#define KID_RAY_SETUP  0  // non-GEMM: ray-AABB + stratified sampling
#define KID_MLP_FWD    1  // GEMM     : 4-layer tiny NeRF MLP per (ray,sample)
#define KID_COMPOSITE  2  // non-GEMM: alpha compositing along each ray

// Tiny NeRF architecture (fixed at compile time; host & device must agree).
//   pos_enc(L=4)  : 3 + 3*2*4 = 27 input features
//   hidden layers : 4 * width 32 with ReLU
//   output head   : 32 -> 4 (sigma + RGB logits)
#define PE_L        4
#define MLP_IN_DIM  27
#define MLP_W       64
#define MLP_DEPTH   4
#define MLP_OUT_DIM 4

typedef struct {
  // Stage 1 (ray setup) I/O
  uint64_t rays_o_addr;      // float[n_rays*3]
  uint64_t rays_d_addr;      // float[n_rays*3]
  uint64_t aabb_addr;        // float[6]
  uint64_t sample_pos_addr;  // float[n_rays*n_samples*3]  (written)
  uint64_t deltas_addr;      // float[n_rays*n_samples]    (written)

  // Stage 2 (MLP) weights — 4 layers
  uint64_t w0_addr; // float[MLP_IN_DIM * MLP_W]
  uint64_t b0_addr; // float[MLP_W]
  uint64_t w1_addr; // float[MLP_W * MLP_W]
  uint64_t b1_addr; // float[MLP_W]
  uint64_t w2_addr; // float[MLP_W * MLP_W]
  uint64_t b2_addr; // float[MLP_W]
  uint64_t w3_addr; // float[MLP_W * MLP_OUT_DIM]
  uint64_t b3_addr; // float[MLP_OUT_DIM]

  // Stage 2 outputs -> Stage 3 inputs
  uint64_t sigmas_addr;      // float[n_rays*n_samples]
  uint64_t rgbs_addr;        // float[n_rays*n_samples*3]

  // Stage 3 output
  uint64_t image_addr;       // float[n_rays*3]

  // Per-CTA cycle count buffer (uint32_t[num_blocks_max])
  uint64_t cycles_addr;

  uint32_t n_rays;
  uint32_t n_samples;
  float    min_near;
  uint32_t kernel_id;
} kernel_arg_t;

#endif
