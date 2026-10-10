#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_tensor.h>
#include <math.h>
#include "common.h"


static inline float fast_exp(float x) {
  x = x < -80.0f ? -80.0f : x;
  float u = x * 0.03125f;
  float y = 1.0f + u*(1.0f + u*(0.5f + u*(0.166666667f + u*(0.041666667f + u*0.008333333f))));
  y = y*y; y = y*y; y = y*y; y = y*y; y = y*y;
  return y;
}

// Polynomial sin/cos/log for the SIMT stages. libm sinf/cosf hang
// non-deterministically on this core at >= ~2k points (reproduced on SimX),
// so the positional encoding and softplus avoid libm entirely.
static inline float fast_sin(float x) {
  const float kTwoPi = 6.28318530718f, kPi = 3.14159265359f, kHalfPi = 1.57079632679f;
  float q = x * (1.0f / kTwoPi);
  int n = (int)(q + (q >= 0.0f ? 0.5f : -0.5f));
  x -= (float)n * kTwoPi;                  // [-pi, pi]
  if (x > kHalfPi) x = kPi - x;            // fold to [-pi/2, pi/2]
  else if (x < -kHalfPi) x = -kPi - x;
  float x2 = x * x;
  return x * (1.0f + x2 * (-1.0f / 6.0f + x2 * (1.0f / 120.0f
           + x2 * (-1.0f / 5040.0f + x2 * (1.0f / 362880.0f)))));
}
static inline float fast_cos(float x) { return fast_sin(x + 1.57079632679f); }

// log(x) for x >= 1 (softplus argument): exponent from the float bits plus an
// atanh series on the mantissa in [1, 2).
static inline float fast_log(float x) {
  if (!(x < 3.0e38f)) return x;
  union { float f; uint32_t u; } v = { x };
  int e = (int)((v.u >> 23) & 0xff) - 127;
  v.u = (v.u & 0x007fffffu) | 0x3f800000u;
  float z = (v.f - 1.0f) / (v.f + 1.0f), z2 = z * z;
  float lm = 2.0f * z * (1.0f + z2 * (1.0f / 3.0f + z2 * (1.0f / 5.0f + z2 * (1.0f / 7.0f + z2 * (1.0f / 9.0f)))));
  return (float)e * 0.69314718056f + lm;
}

// =============================================================================
// akili_NeRF_tcu_sp — Full NeRF forward pass with the MLP GEMMs on the SPARSE
// (2:4) tensor core. KMU launch API.
// =============================================================================

namespace vt = vortex::tensor;
using sp_ctx = vt::wmma_context<VX_CFG_NUM_TCU_LANES, vt::fp16, vt::fp32, true>;

// Extra elements per activation row. The rows are n_points fp16 apart, a
// power-of-two stride that maps every B row of a tile to the same cache set.
#ifndef AKILI_NERF_XPAD
#define AKILI_NERF_XPAD 0
#endif
#ifndef AKILI_NERF_NB
#define AKILI_NERF_NB 1  // output tiles per block along N, sharing each A (and metadata) load
#endif

// Index of activation element (k, n) of a K x n_points B operand. With
// AKILI_NERF_TILEDB the producers write B as contiguous column-major
// tileK x tileN blocks in the order the GEMM consumes them (column tile, then
// K tile); otherwise B is row-major with row stride ldx.
static inline uint32_t x_index(uint32_t k, uint32_t n, uint32_t K, uint32_t ldx) {
#ifdef AKILI_NERF_TILEDB
  constexpr uint32_t TKb = sp_ctx::tileK, TNb = sp_ctx::tileN;
  return ((n / TNb) * (K / TKb) + k / TKb) * (TKb * TNb) + (n % TNb) * TKb + k % TKb;
#else
  (void)K;
  return k * ldx + n;
#endif
}

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
// Stage 1: ray setup + PE (SIMT, one task per point).
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
  uint32_t ldx    = n_pts + AKILI_NERF_XPAD;
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

  pe_out[x_index(0, pt, MLP_IN_DIM_PAD, ldx)] = d_f2h(px);
  pe_out[x_index(1, pt, MLP_IN_DIM_PAD, ldx)] = d_f2h(py);
  pe_out[x_index(2, pt, MLP_IN_DIM_PAD, ldx)] = d_f2h(pz);
  float freq = 1.0f;
  for (int L = 0; L < PE_L; ++L) {
    uint32_t base = 3 + L*6;
    pe_out[x_index((base+0), pt, MLP_IN_DIM_PAD, ldx)] = d_f2h(fast_sin(freq * px));
    pe_out[x_index((base+1), pt, MLP_IN_DIM_PAD, ldx)] = d_f2h(fast_cos(freq * px));
    pe_out[x_index((base+2), pt, MLP_IN_DIM_PAD, ldx)] = d_f2h(fast_sin(freq * py));
    pe_out[x_index((base+3), pt, MLP_IN_DIM_PAD, ldx)] = d_f2h(fast_cos(freq * py));
    pe_out[x_index((base+4), pt, MLP_IN_DIM_PAD, ldx)] = d_f2h(fast_sin(freq * pz));
    pe_out[x_index((base+5), pt, MLP_IN_DIM_PAD, ldx)] = d_f2h(fast_cos(freq * pz));
    freq *= 2.0f;
  }
  for (uint32_t f = MLP_IN_DIM; f < MLP_IN_DIM_PAD; ++f)
    pe_out[x_index(f, pt, MLP_IN_DIM_PAD, ldx)] = 0;
}

// -----------------------------------------------------------------------------
// Stage 2: SPARSE TCU MLP layer.
// -----------------------------------------------------------------------------
static inline void mlp_gemm_body(kernel_arg_t* arg) {
  auto pA = reinterpret_cast<sp_ctx::input_t*>(arg->W_cur_addr);
  auto pB = reinterpret_cast<sp_ctx::input_t*>(arg->X_cur_addr);
  auto pC = reinterpret_cast<sp_ctx::output_t*>(arg->Y_cur_addr);
  auto pMetaBase = reinterpret_cast<const float*>(arg->meta_cur_addr);

  uint32_t K_in  = arg->K_in_cur;
  uint32_t n_pts = arg->n_points;
  uint32_t ldx = n_pts + AKILI_NERF_XPAD;

  sp_ctx::fragment_a   fragA;
  sp_ctx::fragment_b   fragB;
  sp_ctx::fragment_acc fragC[AKILI_NERF_NB];

  uint32_t tile_row = blockIdx.y * sp_ctx::tileM;
  uint32_t tile_col = blockIdx.x * sp_ctx::tileN * AKILI_NERF_NB;
#pragma unroll
  for (uint32_t j = 0; j < AKILI_NERF_NB; ++j)
    sp_ctx::fill_fragment(fragC[j], 0);

  constexpr uint32_t rtl_i_ratio = 32 / vt::fp16::bits;
  constexpr uint32_t meta_cols = (VX_CFG_NUM_TCU_LANES * 2 * rtl_i_ratio + 31) / 32;
  using kcfg = vt::wmma_config_t<VX_CFG_NUM_TCU_LANES>;
  constexpr uint32_t PD = kcfg::m_steps * (kcfg::k_steps / 2);
  constexpr uint32_t num_meta_loads = (PD * meta_cols + VX_CFG_NUM_TCU_LANES - 1) / VX_CFG_NUM_TCU_LANES;
  constexpr uint32_t per_k_tile_words = num_meta_loads * VX_CFG_NUM_TCU_LANES;

  uint32_t num_k_tiles = K_in / sp_ctx::tileK;
  uint32_t tile_row_idx = blockIdx.y;
  uint32_t stride_A = K_in / 2;

  auto pMetaSp = pMetaBase + tile_row_idx * num_k_tiles * per_k_tile_words;
  auto pTileA = pA + tile_row * stride_A;
  constexpr uint32_t a_k_stride = sp_ctx::tileK / 2;
  auto pTileB = pB + tile_col;

#ifdef AKILI_NERF_META_ONCE
  sp_ctx::load_sp_metadata(fragA, pMetaSp);  // timing experiment only
#endif
  for (int i = 0; i < (int)K_in; i += (int)sp_ctx::tileK) {
    sp_ctx::load_matrix_sync<vt::row_major>(fragA, pTileA, stride_A);
#ifndef AKILI_NERF_META_ONCE
    sp_ctx::load_sp_metadata(fragA, pMetaSp);
#endif
#pragma unroll
    for (uint32_t j = 0; j < AKILI_NERF_NB; ++j) {
#ifdef AKILI_NERF_TILEDB
      sp_ctx::load_matrix_sync<vt::col_major>(fragB, pB + x_index(i, tile_col + j * sp_ctx::tileN, K_in, ldx),
                                              sp_ctx::tileK);
#else
      sp_ctx::load_matrix_sync(fragB, pTileB + j * sp_ctx::tileN, ldx);
#endif
      sp_ctx::mma_sync(fragC[j], fragA, fragB, fragC[j]);
    }

    pMetaSp += per_k_tile_words;
    pTileA  += a_k_stride;
    pTileB  += sp_ctx::tileK * ldx;
  }

#pragma unroll
  for (uint32_t j = 0; j < AKILI_NERF_NB; ++j)
    sp_ctx::store_matrix_sync(pC + tile_row * n_pts + tile_col + j * sp_ctx::tileN, fragC[j], n_pts);
}

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
  pOut[x_index(row, col, arg->N_out_cur, n_pts + AKILI_NERF_XPAD)] = d_f2h(v);
}

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
  pSig[tid]        = fast_log(1.0f + fast_exp(s));
  pRGB[tid*3 + 0]  = 1.0f / (1.0f + fast_exp(-r));
  pRGB[tid*3 + 1]  = 1.0f / (1.0f + fast_exp(-g));
  pRGB[tid*3 + 2]  = 1.0f / (1.0f + fast_exp(-b));
}

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
    float alpha = 1.0f - fast_exp(-sigma * delta);
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
