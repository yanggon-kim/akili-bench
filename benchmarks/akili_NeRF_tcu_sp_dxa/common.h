#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

// Kernel stage IDs — the MLP runs one SPARSE TCU GEMM kernel per layer,
// followed by a SIMT activation+cast (ReLU for hidden layers, softplus/
// sigmoid for layer 3).  Ray setup and compositing are SIMT.
#define KID_RAY_SETUP     0  // SIMT: slab + stratified sampling + PE
#define KID_MLP_GEMM      1  // sparse TCU: one layer's W_compressed x X -> Y_fp32
#define KID_MLP_ACT       2  // SIMT: ReLU + bias add + fp32->fp16 cast
#define KID_MLP_OUT_ACT   3  // SIMT: softplus(sigma) + sigmoid(rgb)
#define KID_COMPOSITE     4  // SIMT: alpha compositing along each ray

// Tiny NeRF architecture (fixed at compile time; host & device must agree).
#define PE_L             4
#define MLP_IN_DIM       27
#define MLP_IN_DIM_PAD   32
#define MLP_W            64
#define MLP_DEPTH        4
#define MLP_OUT_DIM      4
#ifndef NUM_THREADS
#define NUM_THREADS 8
#endif
#if NUM_THREADS >= 16
#define MLP_OUT_DIM_PAD  16     // padded to tileM = 16 at NT=16/32
#else
#define MLP_OUT_DIM_PAD  8      // padded to tileM = 8 at NT<=8
#endif

typedef struct {
  // ---- Stage 1 (ray setup) I/O ----
  uint64_t rays_o_addr;
  uint64_t rays_d_addr;
  uint64_t aabb_addr;
  uint64_t pe_buf_addr;
  uint64_t deltas_addr;

  // ---- Stage 2 (MLP) per-layer pointers ----
  uint64_t W_cur_addr;         // COMPRESSED fp16 weight [N_out x K_in/2]
  uint64_t B_cur_addr;         // fp32 bias[N_out]
  uint64_t X_cur_addr;         // fp16 input  [K_in x n_points] row-major
  uint64_t Y_cur_addr;         // fp32 output [N_out x n_points] row-major
  uint64_t act_out_addr;       // fp16 output for ACT stage
  uint64_t meta_cur_addr;      // packed 2:4 metadata for the current layer's W

  // ---- Stage 2 outputs (layer 3 → sigmas/rgbs) ----
  uint64_t sigmas_addr;
  uint64_t rgbs_addr;

  // ---- Stage 3 output ----
  uint64_t image_addr;

  uint64_t cycles_addr;  // uint32_t[num_blocks_max]

  uint32_t K_in_cur;
  uint32_t N_out_cur;

  uint32_t n_rays;
  uint32_t n_samples;
  uint32_t n_points;
  float    min_near;
  uint32_t layer_idx;
  uint32_t kernel_id;
} kernel_arg_t;

#endif
