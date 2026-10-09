#pragma once
#include <stdint.h>

// NeRF forward pass (Mildenhall et al. MLP): ray-AABB + stratified sampling +
// positional encoding (SIMT) -> 8x256 MLP with a skip at layer 4, feature
// layer, 128-wide view-dependent layer (TCU or FPU GEMMs) -> rgb head,
// softplus/sigmoid and alpha compositing (SIMT).
#define NERF_W      256
#define NERF_WV     128
#define NERF_PE_X   10    // xyz frequencies: 3 + 3*2*10 = 63 features
#define NERF_PE_D   4     // dir frequencies: 3 + 3*2*4 = 27 features

typedef struct {
  uint64_t rays_o, rays_d;     // fp32 [R][3]
  uint64_t x0, x0_sc;          // PE(xyz) operand, K = k0
  uint64_t x5, x5_sc;          // [h4 | PE(xyz)] operand, K = 256 + k0
  uint64_t x8, x8_sc;          // [feature | PE(dir)] operand, K = 256 + kd
  uint64_t deltas;             // fp32 [P]
  uint64_t sigma_raw;          // fp32 [P]
  uint64_t h8;                 // fp32 [P][128]
  uint64_t w9, b9;             // fp32 rgb head [3][128], [3]
  uint64_t rgba;               // fp32 [P][4] color + alpha per sample
  uint64_t image;              // fp32 [R][3]
  float    ts_x0, ts_x5, ts_x8;
  uint32_t n_rays, n_samples, n_points, k0, kd;
  float    near, far;
} nerf_args_t;
