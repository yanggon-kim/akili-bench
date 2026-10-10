// akili_fa_tcu_dxa / akili_fa_tcu_sp_dxa — fused FlashAttention-2 forward on
// WGMMA + DXA.
//
// One CTA owns FA_BR query rows (4 warps x xtileM). Q is fetched once into
// LMEM; K/V tiles of FA_BC keys stream through a 2-stage DXA pipeline. Per
// tile: S = Q Kᵀ (WGMMA), online softmax per query row (one lane per row),
// rescale the O accumulator, O += P V (WGMMA). The N x N score matrix never
// leaves the core.
//
// LMEM (16 KB at d = 32): Q 2 KB | 2 x (K 2 KB + Vᵀ 2 KB) | P 2 KB | S 4 KB.
// Under FA_SPARSE, Q is the 2:4-compressed [Br][d/2] (1 KB).
// Each row's softmax rescale factor (and finally 1/l) is published in that
// row's first S slot, which the row's lane has already consumed.

#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>
#include "common.h"
#include "fa_softmax.h"

namespace vt = vortex::tensor;
using ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::fp16, vt::fp32, false, 32>;
#ifdef FA_SPARSE
using qk_ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::fp16, vt::fp32, true, FA_BC>;
constexpr uint32_t kQCols = FA_D / 2;
#else
using qk_ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::fp16, vt::fp32, false, FA_BC>;
constexpr uint32_t kQCols = FA_D;
#endif

constexpr uint32_t kDescQ = 0;
constexpr uint32_t kDescK = 1;
constexpr uint32_t kDescV = 2;

static_assert(qk_ctx::xtileN == FA_BC, "K/V tile width must equal the S accumulator width");
static_assert(ctx::xtileN == FA_D, "P·V output width must equal xtileN");
static_assert(FA_BR % ctx::xtileM == 0 && ctx::xtileM == VX_CFG_NUM_THREADS,
              "one lane per query row within a warp");
static_assert(FA_D % qk_ctx::tileK == 0 && FA_BC % ctx::tileK == 0, "K multiple of tileK");

constexpr uint32_t kQBytes     = FA_BR * kQCols * 2;
constexpr uint32_t kKBytes     = FA_KV * FA_D * 2;
constexpr uint32_t kStageBytes = 2 * kKBytes;
constexpr uint32_t kKVOff      = kQBytes;
constexpr uint32_t kPOff       = kKVOff + FA_STAGES * kStageBytes;
#ifdef FA_REGSM
// The softmax runs on the S accumulator registers, so S needs no LMEM.
#if defined(FA_OLMEM)
// O is parked in LMEM between tiles instead of round-tripping the stack.
constexpr uint32_t kOOff       = kPOff + FA_BR * FA_BC * 2;
constexpr uint32_t kSmemBytes  = kOOff + FA_BR * FA_D * 4;
#else
constexpr uint32_t kSmemBytes  = kPOff + FA_BR * FA_BC * 2;
#endif
#else
constexpr uint32_t kSOff       = kPOff + FA_BR * FA_BC * 2;
constexpr uint32_t kSmemBytes  = kSOff + FA_BR * FA_BC * 4;
#endif
#if defined(FA_SPARSE) && !defined(FA_META_GLOBAL)
constexpr uint32_t kMetaWordsWarp = (FA_D / qk_ctx::tileK) * (qk_ctx::wg_meta_total_bytes / 4);
constexpr uint32_t kMetaBytes = (FA_BR / ctx::xtileM) * kMetaWordsWarp * 4;
#else
constexpr uint32_t kMetaBytes = 0;
#endif
static_assert(kSmemBytes + kMetaBytes <= 16384, "FlashAttention tiles exceed the 16 KB LMEM");

// Fast fp32 exp: Taylor(x/32) then y^32, clamped. Same as akili_attn's softmax.
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

// Row (within the warp's xtileM rows) held by accumulator register r of this
// lane; mirrors wgmma_context::store_matrix_sync's n-major register layout.
static inline uint32_t acc_row(uint32_t lane, uint32_t r) {
  return lane / ctx::tcN + (r % ctx::m_steps) * ctx::tcM;
}

#ifdef FA_REGSM
static_assert(ctx::tcN == 2 && ctx::m_steps == 2,
              "register softmax assumes each row is split across a lane pair");

#ifdef FA_OLMEM
// Loads an accumulator fragment with the register layout store_matrix_sync
// writes (n-major: r = n * m_steps + m).
static inline void load_acc(ctx::fragment_acc& f, const float* src, uint32_t ldm,
                            uint32_t lane) {
  const float* base = src + (lane / ctx::tcN) * ldm + (lane % ctx::tcN);
#pragma unroll
  for (uint32_t r = 0; r < ctx::fragment_acc::NR; ++r)
    f.data[r] = base[(r % ctx::m_steps) * ctx::tcM * ldm + (r / ctx::m_steps) * ctx::tcN];
}
#endif

// Combines a per-row partial across the lane pair (l, l^1) that shares the row.
static inline float pair_max(float v) {
  return __builtin_fmaxf(v, fa_bits_to_float(vx_quad_swap(fa_float_to_bits(v), 1)));
}
static inline float pair_sum(float v) {
  return v + fa_bits_to_float(vx_quad_swap(fa_float_to_bits(v), 1));
}
#endif

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t N         = arg->N;
  const uint32_t lane      = threadIdx.x % VX_CFG_NUM_THREADS;
  const uint32_t warp_rank = threadIdx.x / VX_CFG_NUM_THREADS;
  const uint32_t num_warps = blockDim.x / VX_CFG_NUM_THREADS;
  const uint32_t row0      = blockIdx.x * FA_BR;
  const uint32_t num_tiles = N / FA_KV;

  auto smem = reinterpret_cast<uint8_t*>(__local_mem());
  auto Q = reinterpret_cast<ctx::input_t*>(smem);
  auto stage_K = [&](uint32_t s) { return reinterpret_cast<ctx::input_t*>(smem + kKVOff + s * kStageBytes); };
  auto stage_V = [&](uint32_t s) { return reinterpret_cast<ctx::input_t*>(smem + kKVOff + s * kStageBytes + kKBytes); };
  auto Pw = reinterpret_cast<uint16_t*>(smem + kPOff) + warp_rank * ctx::xtileM * FA_BC;
#ifndef FA_REGSM
  auto Sw = reinterpret_cast<float*>(smem + kSOff) + warp_rank * ctx::xtileM * FA_BC;
#endif
  auto Qw = Q + warp_rank * ctx::xtileM * kQCols;
#ifdef FA_SPARSE
  constexpr uint32_t meta_words = qk_ctx::wg_meta_total_bytes / 4;
  auto Mw = reinterpret_cast<const uint32_t*>(arg->meta_addr)
          + (row0 / ctx::xtileM + warp_rank) * (FA_D / qk_ctx::tileK) * meta_words;
#ifndef FA_META_GLOBAL
  // Q's metadata is the same for every key tile: stage it in shared memory.
  auto Ms = reinterpret_cast<uint32_t*>(smem + kSmemBytes) + warp_rank * kMetaWordsWarp;
  for (uint32_t i = lane; i < kMetaWordsWarp; i += VX_CFG_NUM_THREADS)
    Ms[i] = Mw[i];
  vx_fence();
#else
  auto Ms = Mw;
#endif
#endif

  const uint32_t cta = get_local_group_id();
  auto bar_kv = [&](uint32_t s) { return cta | (s << 8); };
  const uint32_t bar_q    = cta | (2u << 8);
  const uint32_t bar_sync = cta | (3u << 8);
  const bool dxa_warp = (get_sub_group_id() == 0);

  auto issue_kv = [&](uint32_t s, uint32_t tile) {
    vx_barrier_expect_tx(bar_kv(s), 2);
    vx_dxa_issue_2d_wg(kDescK, bar_kv(s), stage_K(s), 0, tile * FA_KV);
    vx_dxa_issue_2d_wg(kDescV, bar_kv(s), stage_V(s), 0, tile * FA_KV);
  };

  if (dxa_warp) {
    vx_barrier_expect_tx(bar_q, 1);
    vx_dxa_issue_2d_wg(kDescQ, bar_q, Q, 0, row0);
    issue_kv(0, 0);
  }
  vx_barrier(bar_q, num_warps);

#ifdef FA_OLMEM
  auto Ow = reinterpret_cast<float*>(smem + kOOff) + warp_rank * ctx::xtileM * FA_D;
  {
    ctx::fragment_acc zero;
    ctx::fill_fragment(zero, 0);
    ctx::store_matrix_sync(Ow, zero, FA_D);
  }
#else
  ctx::fragment_acc fragO;
  ctx::fill_fragment(fragO, 0);
#endif
#ifdef FA_REGSM
  // This lane holds rows lane/2 and lane/2 + 4 of the warp's 8, in registers
  // r % 2 == 0 and 1, at columns lane % 2 + 2 * (r / 2).
  // The running max and sum cross the P·V WGMMA, which pins all 32 FP
  // registers; LLVM cannot spill around it, so they are carried as bits.
  uint32_t mbits[2] = {fa_float_to_bits(-1.0e30f), fa_float_to_bits(-1.0e30f)};
  uint32_t lbits[2] = {0u, 0u};
  uint16_t* pbase = Pw + (lane / ctx::tcN) * FA_BC + (lane % ctx::tcN);
#else
  float m_i = -1.0e30f;
  float l_i = 0.0f;
  float* srow = Sw + lane * FA_BC;
  uint16_t* prow = Pw + lane * FA_BC;
#endif

  for (uint32_t t = 0; t < num_tiles; ++t) {
    const uint32_t cur = FA_STAGES == 1 ? 0u : (t & 1u);
#ifdef FA_ONE_BAR
    static_assert(FA_STAGES == 2, "FA_ONE_BAR needs the 2-stage pipeline");
    // Every warp arriving here has finished P·V of tile t-1, so the other
    // stage is free to refill without a separate release barrier.
    vx_barrier(bar_kv(cur), num_warps);
    if (t + 1 < num_tiles && dxa_warp)
      issue_kv(cur ^ 1u, t + 1);
#else
    if (FA_STAGES == 1) {
      if (t > 0 && dxa_warp)
        issue_kv(0, t);
    } else if (t + 1 < num_tiles && dxa_warp) {
      issue_kv(cur ^ 1u, t + 1);
    }
    vx_barrier(bar_kv(cur), num_warps);
#endif

    for (uint32_t sub = 0; sub < FA_SUB; ++sub) {
      // S = Q Kᵀ over the head dimension.
      qk_ctx::fragment_acc fragS;
      qk_ctx::fill_fragment(fragS, 0);
      for (uint32_t k = 0; k < FA_D; k += qk_ctx::tileK) {
#ifdef FA_SPARSE
        qk_ctx::fragment_a fragM;
        qk_ctx::load_sp_metadata(fragM, Ms + (k / qk_ctx::tileK) * meta_words);
        auto desc_a = vt::vx_make_smem_desc(Qw + k / 2, kQCols * sizeof(ctx::input_t));
#else
        auto desc_a = vt::vx_make_smem_desc(Qw + k, kQCols * sizeof(ctx::input_t));
#endif
        auto desc_b = vt::vx_make_smem_desc(stage_K(cur) + sub * FA_BC * FA_D + k, FA_D * sizeof(ctx::input_t));
        qk_ctx::wgmma_sync(fragS, desc_a, desc_b, fragS);
      }
#ifdef FA_REGSM
      constexpr uint32_t NR = qk_ctx::fragment_acc::NR;
      constexpr float kLog2e = 1.4426950409f;
      const float m_old[2] = {fa_bits_to_float(mbits[0]), fa_bits_to_float(mbits[1])};
      float mx[2] = {m_old[0], m_old[1]};
#pragma unroll
      for (uint32_t r = 0; r < NR; ++r)
        mx[r % 2] = __builtin_fmaxf(mx[r % 2], fragS.data[r]);
#ifdef FA_XCHG_LMEM
      // The row's other half lives in lane ^ 1; swap partial maxima through
      // this warp's P tile, which is dead until this tile's P is written.
      {
        auto xs = reinterpret_cast<float*>(Pw);
        xs[lane * 2] = mx[0];
        xs[lane * 2 + 1] = mx[1];
#ifndef FA_XCHG_NOFENCE
        vx_fence();
#endif
        mx[0] = __builtin_fmaxf(mx[0], xs[(lane ^ 1) * 2]);
        mx[1] = __builtin_fmaxf(mx[1], xs[(lane ^ 1) * 2 + 1]);
#ifndef FA_XCHG_NOFENCE
        vx_fence();
#endif
      }
#else
      mx[0] = pair_max(mx[0]);
      mx[1] = pair_max(mx[1]);
#endif
      float scale[2], sum[2] = {0.0f, 0.0f};
#pragma unroll
      for (uint32_t i = 0; i < 2; ++i)
        scale[i] = (mx[i] == m_old[i]) ? 1.0f : fa_exp2_nonpos((m_old[i] - mx[i]) * kLog2e);
#pragma unroll
      for (uint32_t r = 0; r < NR; ++r) {
        const float p = fa_exp2_nonpos((fragS.data[r] - mx[r % 2]) * kLog2e);
        sum[r % 2] += p;
        pbase[(r % 2) * ctx::tcM * FA_BC + (r / 2) * ctx::tcN] = fa_f2h_unit(p);
      }
#pragma unroll
      for (uint32_t i = 0; i < 2; ++i) {
#ifdef FA_XCHG_LMEM
        // Both lanes of a row apply the same scale, so each keeps the sum of
        // its own columns and the halves are added once at the end.
        lbits[i] = fa_float_to_bits(fa_bits_to_float(lbits[i]) * scale[i] + sum[i]);
#else
        lbits[i] = fa_float_to_bits(fa_bits_to_float(lbits[i]) * scale[i] + pair_sum(sum[i]));
#endif
        mbits[i] = fa_float_to_bits(mx[i]);
      }
      __asm__ volatile("" : "+r"(lbits[0]), "+r"(lbits[1]), "+r"(mbits[0]), "+r"(mbits[1]));
      vx_fence();
#ifdef FA_OLMEM
      ctx::fragment_acc fragO;
      load_acc(fragO, Ow, FA_D, lane);
#endif
#pragma unroll
      for (uint32_t r = 0; r < ctx::fragment_acc::NR; ++r)
        fragO.data[r] *= scale[r % 2];
#else
      qk_ctx::store_matrix_sync(Sw, fragS, FA_BC);
      vx_fence();
#ifdef FA_BAR_REF
      vx_barrier(bar_sync, num_warps);
#endif

      // Online softmax: this lane owns query row (warp_rank * xtileM + lane).
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
      srow[0] = scale;
      vx_fence();
#ifdef FA_BAR_REF
      vx_barrier(bar_sync, num_warps);
#endif

      for (uint32_t r = 0; r < ctx::fragment_acc::NR; ++r)
        fragO.data[r] *= Sw[acc_row(lane, r) * FA_BC];
#endif

      // O += P V over this tile's keys.
      for (uint32_t k = 0; k < FA_BC; k += ctx::tileK) {
        auto desc_a = vt::vx_make_smem_desc(Pw + k, FA_BC * sizeof(uint16_t));
        auto desc_b = vt::vx_make_smem_desc(stage_V(cur) + sub * FA_BC + k, FA_KV * sizeof(ctx::input_t));
        ctx::wgmma_sync(fragO, desc_a, desc_b, fragO);
      }
#ifdef FA_OLMEM
      ctx::store_matrix_sync(Ow, fragO, FA_D);
#endif
    }

#ifndef FA_ONE_BAR
    vx_barrier(bar_kv(cur), num_warps);
#endif
  }

#ifdef FA_REGSM
#ifdef FA_OLMEM
  ctx::fragment_acc fragO;
  load_acc(fragO, Ow, FA_D, lane);
#endif
#ifdef FA_XCHG_LMEM
  {
    auto xs = reinterpret_cast<float*>(Pw);
    xs[lane * 2] = fa_bits_to_float(lbits[0]);
    xs[lane * 2 + 1] = fa_bits_to_float(lbits[1]);
    vx_fence();
    lbits[0] = fa_float_to_bits(fa_bits_to_float(lbits[0]) + xs[(lane ^ 1) * 2]);
    lbits[1] = fa_float_to_bits(fa_bits_to_float(lbits[1]) + xs[(lane ^ 1) * 2 + 1]);
  }
#endif
  const float inv_l[2] = {1.0f / fa_bits_to_float(lbits[0]), 1.0f / fa_bits_to_float(lbits[1])};
  for (uint32_t r = 0; r < ctx::fragment_acc::NR; ++r)
    fragO.data[r] *= inv_l[r % 2];
#else
  srow[0] = 1.0f / l_i;
  vx_fence();
#ifdef FA_BAR_REF
  vx_barrier(bar_sync, num_warps);
#endif
  for (uint32_t r = 0; r < ctx::fragment_acc::NR; ++r)
    fragO.data[r] *= Sw[acc_row(lane, r) * FA_BC];
#endif

  auto pO = reinterpret_cast<ctx::output_t*>(arg->O_addr)
          + (row0 + warp_rank * ctx::xtileM) * FA_D;
  ctx::store_matrix_sync(pO, fragO, FA_D);
}
