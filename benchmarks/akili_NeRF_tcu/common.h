#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

// Kernel stage IDs — the MLP runs one TCU GEMM kernel per layer, followed
// by a SIMT activation+cast (ReLU for hidden layers, softplus/sigmoid for
// layer 3).  Ray setup and compositing are SIMT.
#define KID_RAY_SETUP     0  // SIMT: slab test + stratified sampling + PE
#define KID_MLP_GEMM      1  // TCU : one layer's W x X -> Y_fp32
#define KID_MLP_ACT       2  // SIMT: ReLU + bias add + fp32->fp16 cast
#define KID_MLP_OUT_ACT   3  // SIMT: softplus(sigma) + sigmoid(rgb)
#define KID_COMPOSITE     4  // SIMT: alpha compositing along each ray

// Tiny NeRF architecture (fixed at compile time; host & device must agree).
//   pos_enc(L=4)    : 3 + 3*2*4 = 27 features, PADDED to 32 for TCU tiles
//   4 hidden layers : width 64 with ReLU
//   output head     : 32 -> 4 (sigma + RGB logits); PADDED to 8 for TCU tiles
#define PE_L             4
#define MLP_IN_DIM       27
#define MLP_IN_DIM_PAD   32     // must be multiple of tileK = 16 (dense)
#define MLP_W            64     // hidden width, multiple of tileM=tileN=8
#define MLP_DEPTH        4
#define MLP_OUT_DIM      4
#define MLP_OUT_DIM_PAD  8      // padded to tileM = 8

typedef struct {
  // ---- Stage 1 (ray setup) I/O ----
  uint64_t rays_o_addr;        // float[n_rays*3]
  uint64_t rays_d_addr;        // float[n_rays*3]
  uint64_t aabb_addr;          // float[6]
  uint64_t pe_buf_addr;        // fp16[MLP_IN_DIM_PAD * n_points] feature-major
  uint64_t deltas_addr;        // float[n_rays * n_samples]

  // ---- Stage 2 (MLP) per-layer pointers; host swaps between layers ----
  uint64_t W_cur_addr;         // fp16 weight matrix [N_out_cur x K_in_cur]
  uint64_t B_cur_addr;         // fp32 bias[N_out_cur] (applied in ACT stage)
  uint64_t X_cur_addr;         // fp16 input  [K_in_cur x n_points] row-major
  uint64_t Y_cur_addr;         // fp32 output [N_out_cur x n_points] row-major
  uint64_t act_out_addr;       // fp16 output for ACT stage (next layer's X)

  // ---- Stage 2 output (layer 3 → sigmas/rgbs for compositing) ----
  uint64_t sigmas_addr;        // float[n_points]
  uint64_t rgbs_addr;          // float[n_points * 3]

  // ---- Stage 3 output ----
  uint64_t image_addr;         // float[n_rays * 3]

  // ---- Per-CTA cycle buffer (uint32_t[num_blocks_max]) ----
  uint64_t cycles_addr;

  // ---- Per-layer GEMM scalars ----
  uint32_t K_in_cur;           // GEMM K = input feature count
  uint32_t N_out_cur;          // GEMM M = output feature count

  // ---- Scalars ----
  uint32_t n_rays;
  uint32_t n_samples;
  uint32_t n_points;           // = n_rays * n_samples
  float    min_near;
  uint32_t layer_idx;          // 0..3, used by ACT to pick activation shape
  uint32_t kernel_id;
} kernel_arg_t;

#endif
