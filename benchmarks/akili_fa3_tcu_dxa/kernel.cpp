// akili_fa3_tcu_dxa / akili_fa3_tcu_sp_dxa — fused FlashAttention on WGMMA with
// the head dimension tiled for the 16 KB shared memory.
//
// One warp (8 query rows) per CTA. Per FA_BC-key block:
//   S = Q Kᵀ        K-loop over the whole head dimension, accumulator in regs
//   softmax         one lane per query row; scale published in the S slot
//   O_j = α O_j + P V_j   for every FA_BC-wide output block j; O lives in
//                   shared memory and each block is loaded, extended, stored
// K and V are single 4 KB DXA tiles. K(j+1) streams during the softmax and
// P·V of block j, V(j+1) during S and softmax of block j+1.

#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>
#include "common.h"
#include "fa_softmax.h"

namespace vt = vortex::tensor;
constexpr uint32_t NRC = FA_BC;
using ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::fp16, vt::fp32, false, NRC>;
#ifdef FA_SPARSE
using qk_ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::fp16, vt::fp32, true, NRC>;
constexpr uint32_t kQCols = FA_D / 2;
#else
using qk_ctx = ctx;
constexpr uint32_t kQCols = FA_D;
#endif

constexpr uint32_t kDescQ = 0;
constexpr uint32_t kDescK = 1;
constexpr uint32_t kDescV = 2;

static_assert(ctx::xtileM == VX_CFG_NUM_THREADS && FA_BR == ctx::xtileM,
              "one warp, one lane per query row");
static_assert(ctx::xtileN == FA_BC, "key tile width must equal the accumulator width");
static_assert(FA_D % qk_ctx::tileK == 0 && FA_D % FA_BC == 0 && FA_BC % ctx::tileK == 0,
              "head dim and key tile multiples of the WGMMA tile");

constexpr uint32_t kQBytes = FA_BR * kQCols * 2;
constexpr uint32_t kKBytes = FA_BC * FA_D * 2;
constexpr uint32_t kOBytes = FA_BR * FA_D * 4;
constexpr uint32_t kOffQ = 0;
constexpr uint32_t kOffK = kOffQ + kQBytes;
constexpr uint32_t kOffV = kOffK + kKBytes;
constexpr uint32_t kOffO = kOffV + kKBytes;
constexpr uint32_t kOffS = kOffO + kOBytes;
constexpr uint32_t kOffP = kOffS + FA_BR * FA_BC * 4;
constexpr uint32_t kOffM = kOffP + FA_BR * FA_BC * 2;
#ifdef FA_SPARSE
constexpr uint32_t kMetaWords = (FA_D / qk_ctx::tileK) * (qk_ctx::wg_meta_total_bytes / 4);
#else
constexpr uint32_t kMetaWords = 0;
#endif
static_assert(kOffM + kMetaWords * 4 <= 16384, "FlashAttention tiles exceed the 16 KB LMEM");

// Loads an accumulator fragment from memory with the register layout that
// store_matrix_sync writes (n-major: r = n * m_steps + m).
static inline void load_acc(ctx::fragment_acc& f, const float* src, uint32_t ldm,
                            uint32_t lane) {
  const float* base = src + (lane / ctx::tcN) * ldm + (lane % ctx::tcN);
#pragma unroll
  for (uint32_t r = 0; r < ctx::fragment_acc::NR; ++r)
    f.data[r] = base[(r % ctx::m_steps) * ctx::tcM * ldm + (r / ctx::m_steps) * ctx::tcN];
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t N    = arg->N;
  const uint32_t lane = threadIdx.x % VX_CFG_NUM_THREADS;
  const uint32_t row0 = blockIdx.x * FA_BR;
  const uint32_t nb   = N / FA_BC;

  auto smem = reinterpret_cast<uint8_t*>(__local_mem());
  auto Q    = reinterpret_cast<ctx::input_t*>(smem + kOffQ);
  auto Kt   = reinterpret_cast<ctx::input_t*>(smem + kOffK);
  auto Vt   = reinterpret_cast<ctx::input_t*>(smem + kOffV);
  auto Ob   = reinterpret_cast<float*>(smem + kOffO);
  auto Sw   = reinterpret_cast<float*>(smem + kOffS);
  auto Pw   = reinterpret_cast<uint16_t*>(smem + kOffP);

  const uint32_t cta   = get_local_group_id();
  const uint32_t bar_k = cta;
  const uint32_t bar_v = cta | (1u << 8);
  const uint32_t bar_q = cta | (2u << 8);

#ifdef FA_SPARSE
  auto Mg = reinterpret_cast<const uint32_t*>(arg->meta_addr)
          + (row0 / ctx::xtileM) * kMetaWords;
  auto Ms = reinterpret_cast<uint32_t*>(smem + kOffM);
  for (uint32_t i = lane; i < kMetaWords; i += VX_CFG_NUM_THREADS)
    Ms[i] = Mg[i];
  vx_fence();
  constexpr uint32_t meta_words = qk_ctx::wg_meta_total_bytes / 4;
#endif

  vx_barrier_expect_tx(bar_q, 1);
  vx_dxa_issue_2d_wg(kDescQ, bar_q, Q, 0, row0);
  vx_barrier_expect_tx(bar_k, 1);
  vx_dxa_issue_2d_wg(kDescK, bar_k, Kt, 0, 0);
  vx_barrier_expect_tx(bar_v, 1);
  vx_dxa_issue_2d_wg(kDescV, bar_v, Vt, 0, 0);
  vx_barrier(bar_q, 1);

  float m_i = -1.0e30f;
  float l_i = 0.0f;
  float* srow = Sw + lane * FA_BC;
  uint16_t* prow = Pw + lane * FA_BC;
  auto pO = reinterpret_cast<ctx::output_t*>(arg->O_addr) + row0 * FA_D;

  for (uint32_t blk = 0; blk < nb; ++blk) {
    vx_barrier(bar_k, 1);

    qk_ctx::fragment_acc fragS;
    qk_ctx::fill_fragment(fragS, 0);
#pragma unroll
    for (uint32_t k = 0; k < FA_D; k += qk_ctx::tileK) {
#ifdef FA_SPARSE
      qk_ctx::fragment_a fragM;
      qk_ctx::load_sp_metadata(fragM, Ms + (k / qk_ctx::tileK) * meta_words);
      auto desc_a = vt::vx_make_smem_desc(Q + k / 2, kQCols * sizeof(ctx::input_t));
#else
      auto desc_a = vt::vx_make_smem_desc(Q + k, kQCols * sizeof(ctx::input_t));
#endif
      auto desc_b = vt::vx_make_smem_desc(Kt + k, FA_D * sizeof(ctx::input_t));
      qk_ctx::wgmma_sync(fragS, desc_a, desc_b, fragS);
    }
    qk_ctx::store_matrix_sync(Sw, fragS, FA_BC);
    vx_fence();

    if (blk + 1 < nb) {
      vx_barrier_expect_tx(bar_k, 1);
      vx_dxa_issue_2d_wg(kDescK, bar_k, Kt, 0, (blk + 1) * FA_BC);
    }

    const float scale = fa_softmax_tile(srow, prow, m_i, l_i);
    const bool last = (blk + 1 == nb);
    srow[0] = scale;
    if (last)
      srow[1] = 1.0f / l_i;
    vx_fence();

    float alpha[ctx::m_steps], inv_l[ctx::m_steps];
#pragma unroll
    for (uint32_t mi = 0; mi < ctx::m_steps; ++mi) {
      const float* row = Sw + ((lane / ctx::tcN) + mi * ctx::tcM) * FA_BC;
      alpha[mi] = row[0];
      inv_l[mi] = row[1];
    }

    vx_barrier(bar_v, 1);

    for (uint32_t j = 0; j < FA_D / FA_BC; ++j) {
      ctx::fragment_acc fragO;
      if (blk == 0) {
        ctx::fill_fragment(fragO, 0);
      } else {
        load_acc(fragO, Ob + j * FA_BC, FA_D, lane);
#pragma unroll
        for (uint32_t r = 0; r < ctx::fragment_acc::NR; ++r)
          fragO.data[r] *= alpha[r % ctx::m_steps];
      }
#pragma unroll
      for (uint32_t ks = 0; ks < FA_BC; ks += ctx::tileK) {
        auto desc_a = vt::vx_make_smem_desc(Pw + ks, FA_BC * sizeof(uint16_t));
        auto desc_b = vt::vx_make_smem_desc(Vt + j * FA_BC * FA_BC + ks,
                                            FA_BC * sizeof(ctx::input_t));
        ctx::wgmma_sync(fragO, desc_a, desc_b, fragO);
      }
      if (last) {
#pragma unroll
        for (uint32_t r = 0; r < ctx::fragment_acc::NR; ++r)
          fragO.data[r] *= inv_l[r % ctx::m_steps];
        ctx::store_matrix_sync(pO + j * FA_BC, fragO, FA_D);
      } else {
        ctx::store_matrix_sync(Ob + j * FA_BC, fragO, FA_D);
      }
    }
    vx_fence();

    if (blk + 1 < nb) {
      vx_barrier_expect_tx(bar_v, 1);
      vx_dxa_issue_2d_wg(kDescV, bar_v, Vt, 0, (blk + 1) * FA_BC);
    }
  }
}
