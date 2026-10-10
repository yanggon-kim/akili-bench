#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include "common.h"
#include "fa_softmax.h"

namespace vt = vortex::tensor;
#ifdef FA_SPARSE
using qk_ctx = vt::wmma_context<VX_CFG_NUM_TCU_LANES, vt::fp16, vt::fp32, true>;
#else
using qk_ctx = vt::wmma_context<VX_CFG_NUM_TCU_LANES, vt::fp16, vt::fp32, false>;
#endif
using pv_ctx = vt::wmma_context<VX_CFG_NUM_TCU_LANES, vt::fp16, vt::fp32, false>;

constexpr uint32_t TM = pv_ctx::tileM;
constexpr uint32_t TN = pv_ctx::tileN;
constexpr uint32_t TK = pv_ctx::tileK;
static_assert(TM == VX_CFG_NUM_THREADS, "one lane per query row within a warp");
static_assert(FA_BR % TM == 0 && FA_BC % TN == 0 && FA_BC % TK == 0, "tile shape");
static_assert(FA_D % TK == 0 && FA_D % TN == 0, "head dim must be a multiple of the tile");

constexpr uint32_t kWarpS = TM * FA_BC * 4;
constexpr uint32_t kWarpP = TM * FA_BC * 2;
constexpr uint32_t kWarpO = TM * FA_D * 4;
#if defined(FA_SPARSE) && !defined(FA_META_GLOBAL)
constexpr uint32_t kMetaWords = (FA_D / TK) * qk_ctx::meta_stride;
constexpr uint32_t kWarpMeta  = kMetaWords * 4;
#else
constexpr uint32_t kWarpMeta  = 0;
#endif
constexpr uint32_t kWarpBytes = kWarpS + kWarpP + kWarpO + kWarpMeta;

static inline float fast_exp(float x) {
  x = x < -80.0f ? -80.0f : x;
  float u = x * 0.03125f;
  float y = 1.0f + u*(1.0f + u*(0.5f + u*(0.166666667f + u*(0.041666667f + u*0.008333333f))));
  y = y*y; y = y*y; y = y*y; y = y*y; y = y*y;
  return y;
}

static inline uint16_t f2h(float x) {
  uint32_t bits;
  __builtin_memcpy(&bits, &x, sizeof(bits));
  uint16_t sign = (uint16_t)((bits >> 16) & 0x8000);
  uint16_t mant = (uint16_t)((bits >> 13) & 0x03FF);
  int32_t  ex   = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
  if (ex >= 31) return (uint16_t)(sign | 0x7C00);
  if (ex <= 0)  return sign;
  return (uint16_t)(sign | ((uint16_t)ex << 10) | mant);
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t N         = arg->N;
  const uint32_t lane      = threadIdx.x % VX_CFG_NUM_THREADS;
  const uint32_t warp_rank = threadIdx.x / VX_CFG_NUM_THREADS;
  const uint32_t wrow0     = blockIdx.x * FA_BR + warp_rank * TM;
  const uint32_t num_tiles = N / FA_BC;

  auto smem = reinterpret_cast<uint8_t*>(__local_mem()) + warp_rank * kWarpBytes;
  auto Sw = reinterpret_cast<float*>(smem);
  auto Pw = reinterpret_cast<pv_ctx::input_t*>(smem + kWarpS);
  auto Ow = reinterpret_cast<float*>(smem + kWarpS + kWarpP);

  auto K = reinterpret_cast<const qk_ctx::input_t*>(arg->K_addr);
  auto V = reinterpret_cast<const pv_ctx::input_t*>(arg->V_addr);
#ifdef FA_SPARSE
  auto Qw = reinterpret_cast<const qk_ctx::input_t*>(arg->Q_addr) + wrow0 * (FA_D / 2);
  auto Mw = reinterpret_cast<const uint32_t*>(arg->meta_addr)
          + (wrow0 / TM) * (FA_D / TK) * qk_ctx::meta_stride;
#ifndef FA_META_GLOBAL
  // Q's metadata is the same for every key tile: stage it in shared memory
  // once instead of fetching it from global memory before every MMA.
  auto Ms = reinterpret_cast<uint32_t*>(smem + kWarpS + kWarpP + kWarpO);
  for (uint32_t i = lane; i < kMetaWords; i += VX_CFG_NUM_THREADS)
    Ms[i] = Mw[i];
  vx_fence();
#else
  auto Ms = Mw;
#endif
#else
  auto Qw = reinterpret_cast<const qk_ctx::input_t*>(arg->Q_addr) + wrow0 * FA_D;
#endif

  float* srow = Sw + lane * FA_BC;
  uint16_t* prow = Pw + lane * FA_BC;
  float* orow = Ow + lane * FA_D;
  for (uint32_t j = 0; j < FA_D; ++j)
    orow[j] = 0.0f;
  float m_i = -1.0e30f;
  float l_i = 0.0f;

#if FA_D <= 32
  // Q stays in registers across all key tiles.
  qk_ctx::fragment_a   fragQ[FA_D / TK];
  for (uint32_t k = 0; k < FA_D; k += TK) {
#ifdef FA_SPARSE
    qk_ctx::load_matrix_sync(fragQ[k / TK], Qw + k / 2, FA_D / 2);
#else
    qk_ctx::load_matrix_sync(fragQ[k / TK], Qw + k, FA_D);
#endif
  }
#else
  // Wider heads do not fit 32 FP registers: Q fragments are reloaded per MMA.
  qk_ctx::fragment_a   fQl;
#endif
  qk_ctx::fragment_b   fragK;
  qk_ctx::fragment_acc fragS;
  pv_ctx::fragment_a   fragP;
  pv_ctx::fragment_b   fragV;
  pv_ctx::fragment_acc fragO;

  for (uint32_t t = 0; t < num_tiles; ++t) {
    const uint32_t key0 = t * FA_BC;

    // S = Q Kᵀ over the head dimension.
    for (uint32_t n = 0; n < FA_BC; n += TN) {
      qk_ctx::fill_fragment(fragS, 0);
      for (uint32_t k = 0; k < FA_D; k += TK) {
#if FA_D <= 32
        auto& fQ = fragQ[k / TK];
#else
        auto& fQ = fQl;
#ifdef FA_SPARSE
        qk_ctx::load_matrix_sync(fQ, Qw + k / 2, FA_D / 2);
#else
        qk_ctx::load_matrix_sync(fQ, Qw + k, FA_D);
#endif
#endif
#ifdef FA_SPARSE
        qk_ctx::load_sp_metadata(fQ, Ms + (k / TK) * qk_ctx::meta_stride);
#endif
        qk_ctx::load_matrix_sync<vt::col_major>(fragK, K + (key0 + n) * FA_D + k, FA_D);
        qk_ctx::mma_sync(fragS, fQ, fragK, fragS);
      }
      qk_ctx::store_matrix_sync(Sw + n, fragS, FA_BC);
    }
    vx_fence();

    // Online softmax: this lane owns query row wrow0 + lane.
#ifdef FA_SOFTMAX_REF
    float m_new = m_i;
    for (uint32_t j = 0; j < FA_BC; ++j)
      m_new = srow[j] > m_new ? srow[j] : m_new;
    float scale = fast_exp(m_i - m_new);
    float sum = 0.0f;
    for (uint32_t j = 0; j < FA_BC; ++j) {
      float p = fast_exp(srow[j] - m_new);
      sum += p;
      prow[j] = f2h(p);
    }
    l_i = l_i * scale + sum;
    m_i = m_new;
#else
    const float scale = fa_softmax_tile(srow, prow, m_i, l_i);
#endif
#pragma unroll
    for (uint32_t j = 0; j < FA_D; ++j)
      orow[j] *= scale;
    vx_fence();

    // O += P V over this tile's keys.
    for (uint32_t n = 0; n < FA_D; n += TN) {
      pv_ctx::load_matrix_sync(fragO, Ow + n, FA_D);
      for (uint32_t k = 0; k < FA_BC; k += TK) {
        pv_ctx::load_matrix_sync(fragP, Pw + k, FA_BC);
        pv_ctx::load_matrix_sync<vt::row_major>(fragV, V + (key0 + k) * FA_D + n, FA_D);
        pv_ctx::mma_sync(fragO, fragP, fragV, fragO);
      }
      pv_ctx::store_matrix_sync(Ow + n, fragO, FA_D);
    }
    vx_fence();
  }

  const float inv_l = 1.0f / l_i;
  auto pO = reinterpret_cast<float*>(arg->O_addr) + (wrow0 + lane) * FA_D;
  for (uint32_t j = 0; j < FA_D; ++j)
    pO[j] = orow[j] * inv_l;
}
