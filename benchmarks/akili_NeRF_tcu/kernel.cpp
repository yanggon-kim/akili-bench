#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_tensor.h>
#include <math.h>
#include "common.h"

// =============================================================================
// akili_NeRF_tcu — Full NeRF forward pass with TCU-accelerated MLP on the KMU
// launch API. One __kernel entry point dispatches 5 stages via arg->kernel_id.
//
//   KID_RAY_SETUP    : SIMT ray-AABB slab + stratified sampling + PE
//   KID_MLP_GEMM     : dense TCU layer GEMM (2D tile grid)
//   KID_MLP_ACT      : SIMT ReLU + bias + fp32->fp16 cast
//   KID_MLP_OUT_ACT  : SIMT softplus(sigma) + sigmoid(rgb)
//   KID_COMPOSITE    : SIMT per-ray alpha compositing
// =============================================================================

namespace vt = vortex::tensor;
using tcu_ctx = vt::wmma_context<NUM_TCU_LANES, vt::fp16, vt::fp32, false>;

// fp32 -> fp16 (IEEE) packing helper.
static inline uint16_t d_f2h(float x) {
  uint32_t bits;
  __builtin_memcpy(&bits, &x, sizeof(bits));
  uint16_t sign = (uint16_t)((bits >> 16) & 0x8000);
  uint16_t mant = (uint16_t)((bits >> 13) & 0x03FF);
  int32_t  ex   = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
  if (ex >= 31) return (uint16_t)(sign | 0x7C00);
  if (ex <= 0)  return sign;
  return (uint16_t)(sign | ((uint16_t)ex << 10) | mant);
}

// -----------------------------------------------------------------------------
// Stage 1: ray setup + positional encoding (SIMT, one task per point).
// -----------------------------------------------------------------------------
static inline void ray_setup_body(kernel_arg_t* arg) {
  auto rays_o  = reinterpret_cast<float*>(arg->rays_o_addr);
  auto rays_d  = reinterpret_cast<float*>(arg->rays_d_addr);
  auto aabb    = reinterpret_cast<float*>(arg->aabb_addr);
  auto pe_out  = reinterpret_cast<uint16_t*>(arg->pe_buf_addr);
  auto dlt_out = reinterpret_cast<float*>(arg->deltas_addr);

  uint32_t pt = blockIdx.x * blockDim.x + threadIdx.x;
  if (pt >= arg->n_points) return;

  uint32_t S      = arg->n_samples;
  uint32_t n_pts  = arg->n_points;
  uint32_t ray    = pt / S;
  uint32_t s      = pt - ray * S;
  float min_near  = arg->min_near;

  float ox = rays_o[ray*3+0], oy = rays_o[ray*3+1], oz = rays_o[ray*3+2];
  float dx = rays_d[ray*3+0], dy = rays_d[ray*3+1], dz = rays_d[ray*3+2];

  float inv_dx = 1.0f / dx, inv_dy = 1.0f / dy, inv_dz = 1.0f / dz;
  float tx1 = (aabb[0] - ox) * inv_dx, tx2 = (aabb[3] - ox) * inv_dx;
  float ty1 = (aabb[1] - oy) * inv_dy, ty2 = (aabb[4] - oy) * inv_dy;
  float tz1 = (aabb[2] - oz) * inv_dz, tz2 = (aabb[5] - oz) * inv_dz;
  float tmin_x = (tx1 < tx2) ? tx1 : tx2;
  float tmax_x = (tx1 > tx2) ? tx1 : tx2;
  float tmin_y = (ty1 < ty2) ? ty1 : ty2;
  float tmax_y = (ty1 > ty2) ? ty1 : ty2;
  float tmin_z = (tz1 < tz2) ? tz1 : tz2;
  float tmax_z = (tz1 > tz2) ? tz1 : tz2;
  float tmin = tmin_x;
  if (tmin_y > tmin) tmin = tmin_y;
  if (tmin_z > tmin) tmin = tmin_z;
  float tmax = tmax_x;
  if (tmax_y < tmax) tmax = tmax_y;
  if (tmax_z < tmax) tmax = tmax_z;
  if (tmin < min_near) tmin = min_near;
  float step = 0.0f;
  if (tmax > tmin) step = (tmax - tmin) / (float)S;

  float t  = tmin + ((float)s + 0.5f) * step;
  float px = ox + t * dx;
  float py = oy + t * dy;
  float pz = oz + t * dz;
  dlt_out[pt] = step;

  pe_out[0 * n_pts + pt] = d_f2h(px);
  pe_out[1 * n_pts + pt] = d_f2h(py);
  pe_out[2 * n_pts + pt] = d_f2h(pz);
  float freq = 1.0f;
  for (int L = 0; L < PE_L; ++L) {
    uint32_t base = 3 + L*6;
    pe_out[(base+0) * n_pts + pt] = d_f2h(sinf(freq * px));
    pe_out[(base+1) * n_pts + pt] = d_f2h(cosf(freq * px));
    pe_out[(base+2) * n_pts + pt] = d_f2h(sinf(freq * py));
    pe_out[(base+3) * n_pts + pt] = d_f2h(cosf(freq * py));
    pe_out[(base+4) * n_pts + pt] = d_f2h(sinf(freq * pz));
    pe_out[(base+5) * n_pts + pt] = d_f2h(cosf(freq * pz));
    freq *= 2.0f;
  }
  for (uint32_t f = MLP_IN_DIM; f < MLP_IN_DIM_PAD; ++f)
    pe_out[f * n_pts + pt] = 0;
}

// -----------------------------------------------------------------------------
// Stage 2: dense TCU MLP layer (2D tile grid).
// -----------------------------------------------------------------------------
static inline void mlp_gemm_body(kernel_arg_t* arg) {
  auto pA = reinterpret_cast<tcu_ctx::input_t*>(arg->W_cur_addr);
  auto pB = reinterpret_cast<tcu_ctx::input_t*>(arg->X_cur_addr);
  auto pC = reinterpret_cast<tcu_ctx::output_t*>(arg->Y_cur_addr);

  uint32_t K_in  = arg->K_in_cur;
  uint32_t n_pts = arg->n_points;

  tcu_ctx::fragment_a   fragA;
  tcu_ctx::fragment_b   fragB;
  tcu_ctx::fragment_acc fragC;

  uint32_t tile_row = blockIdx.y * tcu_ctx::tileM;
  uint32_t tile_col = blockIdx.x * tcu_ctx::tileN;
  tcu_ctx::fill_fragment(fragC, 0);

  for (int i = 0; i < (int)K_in; i += (int)tcu_ctx::tileK) {
    auto pTileA = pA + tile_row * K_in + i;
    auto pTileB = pB + i * n_pts + tile_col;
    tcu_ctx::load_matrix_sync(fragA, pTileA, K_in);
    tcu_ctx::load_matrix_sync(fragB, pTileB, n_pts);
    tcu_ctx::mma_sync(fragC, fragA, fragB, fragC);
  }

  auto pTileC = pC + tile_row * n_pts + tile_col;
  tcu_ctx::store_matrix_sync(pTileC, fragC, n_pts);
}

// -----------------------------------------------------------------------------
// Stage 2: SIMT hidden-layer activation (ReLU + bias + fp32→fp16).
// -----------------------------------------------------------------------------
static inline void mlp_act_body(kernel_arg_t* arg) {
  auto pY_fp32 = reinterpret_cast<float*>(arg->Y_cur_addr);
  auto pB      = reinterpret_cast<float*>(arg->B_cur_addr);
  auto pOut    = reinterpret_cast<uint16_t*>(arg->act_out_addr);

  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t total = arg->N_out_cur * arg->n_points;
  if (tid >= total) return;

  uint32_t n_pts = arg->n_points;
  uint32_t row = tid / n_pts;
  uint32_t col = tid - row * n_pts;

  float v = pY_fp32[row * n_pts + col] + pB[row];
  if (v < 0.0f) v = 0.0f;
  pOut[row * n_pts + col] = d_f2h(v);
}

// -----------------------------------------------------------------------------
// Stage 2 final: softplus(sigma) + sigmoid(rgb) (SIMT, one task per point).
// -----------------------------------------------------------------------------
static inline void mlp_out_act_body(kernel_arg_t* arg) {
  auto pY_fp32 = reinterpret_cast<float*>(arg->Y_cur_addr);
  auto pB      = reinterpret_cast<float*>(arg->B_cur_addr);
  auto pSig    = reinterpret_cast<float*>(arg->sigmas_addr);
  auto pRGB    = reinterpret_cast<float*>(arg->rgbs_addr);

  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= arg->n_points) return;

  uint32_t n_pts = arg->n_points;
  float s  = pY_fp32[0 * n_pts + tid] + pB[0];
  float r  = pY_fp32[1 * n_pts + tid] + pB[1];
  float g  = pY_fp32[2 * n_pts + tid] + pB[2];
  float b  = pY_fp32[3 * n_pts + tid] + pB[3];
  pSig[tid]        = logf(1.0f + expf(s));
  pRGB[tid*3 + 0]  = 1.0f / (1.0f + expf(-r));
  pRGB[tid*3 + 1]  = 1.0f / (1.0f + expf(-g));
  pRGB[tid*3 + 2]  = 1.0f / (1.0f + expf(-b));
}

// -----------------------------------------------------------------------------
// Stage 3: SIMT alpha compositing (one task per ray).
// -----------------------------------------------------------------------------
static inline void composite_body(kernel_arg_t* arg) {
  auto sigmas = reinterpret_cast<float*>(arg->sigmas_addr);
  auto rgbs   = reinterpret_cast<float*>(arg->rgbs_addr);
  auto deltas = reinterpret_cast<float*>(arg->deltas_addr);
  auto image  = reinterpret_cast<float*>(arg->image_addr);

  uint32_t ray = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray >= arg->n_rays) return;

  uint32_t S   = arg->n_samples;
  uint32_t base = ray * S;

  float T = 1.0f;
  float R = 0.0f, G = 0.0f, B = 0.0f;
  for (uint32_t s = 0; s < S; ++s) {
    float sigma = sigmas[base + s];
    float delta = deltas[base + s];
    float alpha = 1.0f - expf(-sigma * delta);
    float w = alpha * T;
    R += w * rgbs[(base + s) * 3 + 0];
    G += w * rgbs[(base + s) * 3 + 1];
    B += w * rgbs[(base + s) * 3 + 2];
    T *= (1.0f - alpha);
  }
  image[ray * 3 + 0] = R;
  image[ray * 3 + 1] = G;
  image[ray * 3 + 2] = B;
}

// =============================================================================
// KMU kernel entry
// =============================================================================
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  __rdcycle_time t0 = vx_rdcycle_sync_begin();

  switch (arg->kernel_id) {
    case KID_RAY_SETUP:    ray_setup_body(arg);    break;
    case KID_MLP_GEMM:     mlp_gemm_body(arg);     break;
    case KID_MLP_ACT:      mlp_act_body(arg);      break;
    case KID_MLP_OUT_ACT:  mlp_out_act_body(arg);  break;
    case KID_COMPOSITE:    composite_body(arg);    break;
    default: break;
  }

  __rdcycle_time t1 = vx_rdcycle_sync_end();
  if (threadIdx.x == 0) {
    auto pCycles = reinterpret_cast<uint32_t*>(arg->cycles_addr);
    uint32_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
    pCycles[block_id] = (uint32_t)vx_rdcycle_sync_diff(t0, t1);
  }
}
