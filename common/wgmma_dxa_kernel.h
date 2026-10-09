#ifndef AKILI_WGMMA_DXA_KERNEL_H
#define AKILI_WGMMA_DXA_KERNEL_H

// PIPE_N-stage software-pipelined WGMMA+DXA GEMM (DEEP_BUF), ported from
// tests/regression/sgemm_tcu_wg_dxa_deepbuf and sgemm_tcu_wg_sp_dxa.
//
// The first warp of the CTA prefetches PIPE_N-1 stages ahead; every stage
// holds KB K-tiles and is consumed by KB back-to-back WGMMAs whose smem
// descriptors step through the stage. A is always row-major [M][K]; B is
// staged K-major ([N][K] in smem). B_KN selects how B sits in global memory:
//   false: [N][K] row-major, RowMajor descriptor, coords (k, n)
//   true : [K][N] row-major, KMajor descriptor (DXA transposes), coords (n, k)
// The stage count rounds up; a ragged last stage stops at the last K-tile
// (K is padded to a whole K-tile by the host), since out-of-range stage data
// is not guaranteed to be zero.

#include <vx_barrier.h>
#include <vx_dxa.h>
#include <vx_intrinsics.h>
#include <vx_tensor.h>
#include "wgmma_dxa_layout.h"
#include "akili_ablock.h"

#if !defined(WGMMA_SS)
#error "akili WGMMA kernels require WGMMA_SS (A from shared memory)"
#endif

namespace akili {

template <bool B_KN>
inline __attribute__((always_inline)) void wg_issue_b(
    uint32_t desc_b, uint32_t bar, void* dst, uint32_t k, uint32_t col) {
  if constexpr (B_KN)
    vx_dxa_issue_2d_wg(desc_b, bar, dst, col, k);
  else
    vx_dxa_issue_2d_wg(desc_b, bar, dst, k, col);
}

template <typename Ctx, bool B_KN = false>
inline __attribute__((always_inline)) void wgmma_dxa(
    typename Ctx::output_t* output, uint32_t output_stride, uint32_t K,
    uint32_t desc_a, uint32_t desc_b) {
  using in_t = typename Ctx::input_t;
  constexpr uint32_t bk = kWgKB * Ctx::tileK;

  uint32_t warp_rank = threadIdx.x / VX_CFG_NUM_THREADS;
  uint32_t num_warps = blockDim.x / VX_CFG_NUM_THREADS;
  uint32_t cta_m = num_warps * Ctx::xtileM;
  uint32_t tile_row = blockIdx.y * cta_m;
  uint32_t tile_col = blockIdx.x * Ctx::xtileN;

  auto smem = reinterpret_cast<uint8_t*>(__local_mem());
  const uint32_t stage_bytes = wg_dense_stage_bytes(cta_m, Ctx::tileK, Ctx::xtileN);
  auto stage_a = [&](uint32_t s) {
    return reinterpret_cast<in_t*>(smem + s * stage_bytes);
  };
  auto stage_b = [&](uint32_t s) { return stage_a(s) + cta_m * bk; };

  const uint32_t cta_id = get_local_group_id();
  auto bar_id = [&](uint32_t s) { return cta_id | (s << 8); };
  const bool is_dxa_warp = (get_sub_group_id() == 0);
  const uint32_t num_k_tiles = (K + Ctx::tileK - 1) / Ctx::tileK;
  const uint32_t num_steps = (num_k_tiles + kWgKB - 1) / kWgKB;

  auto issue = [&](uint32_t s, uint32_t k) {
    vx_barrier_expect_tx(bar_id(s), 2);
#ifdef AKILI_A_BLOCKED
    static_assert(bk == kABlockCols, "blocked A needs one column-block per stage");
    vx_dxa_issue_2d_wg(desc_a, bar_id(s), stage_a(s), k * kABlockRows, tile_row / kABlockRows);
#else
    vx_dxa_issue_2d_wg(desc_a, bar_id(s), stage_a(s), k, tile_row);
#endif
    wg_issue_b<B_KN>(desc_b, bar_id(s), stage_b(s), k, tile_col);
  };

  typename Ctx::fragment_acc frag_c;
  Ctx::fill_fragment(frag_c, 0);

  if (is_dxa_warp) {
    for (uint32_t s = 0; s < kWgPipeN - 1 && s < num_steps; ++s)
      issue(s, s * bk);
  }

  for (uint32_t i = 0; i < num_steps; ++i) {
    const uint32_t cur = i % kWgPipeN;
    const uint32_t issue_i = i + kWgPipeN - 1;
    if (issue_i < num_steps && is_dxa_warp)
      issue(issue_i % kWgPipeN, issue_i * bk);

    vx_barrier(bar_id(cur), num_warps);

    for (uint32_t ki = 0; ki < kWgKB; ++ki) {
      if (i * kWgKB + ki >= num_k_tiles)
        break;
      auto a_warp = stage_a(cur) + warp_rank * Ctx::xtileM * bk + ki * Ctx::tileK;
      auto desc_a_smem = vortex::tensor::vx_make_smem_desc(a_warp, bk * sizeof(in_t));
      auto desc_b_smem = vortex::tensor::vx_make_smem_desc(
          stage_b(cur) + ki * Ctx::tileK, bk * sizeof(in_t));
      Ctx::wgmma_sync(frag_c, desc_a_smem, desc_b_smem, frag_c);
    }

    vx_barrier(bar_id(cur), num_warps);
  }

  auto tile_c = output
      + (tile_row + warp_rank * Ctx::xtileM) * output_stride + tile_col;
  Ctx::store_matrix_sync(tile_c, frag_c, output_stride);
}

// metadata: per (warp tile-row, K-tile) blocks of wg_meta_total_bytes, packed
// by akili_pack_wgmma_metadata. K must be a multiple of Ctx::tileK.
template <typename Ctx, bool B_KN = false>
inline __attribute__((always_inline)) void sparse_wgmma_dxa(
    typename Ctx::output_t* output, const uint32_t* metadata,
    uint32_t output_stride, uint32_t K, uint32_t desc_a, uint32_t desc_b) {
  using in_t = typename Ctx::input_t;
  constexpr uint32_t bk = kWgKB * Ctx::tileK;
  constexpr uint32_t meta_words = Ctx::wg_meta_total_bytes / 4;

  uint32_t warp_rank = threadIdx.x / VX_CFG_NUM_THREADS;
  uint32_t num_warps = blockDim.x / VX_CFG_NUM_THREADS;
  uint32_t cta_m = num_warps * Ctx::xtileM;
  uint32_t tile_row = blockIdx.y * cta_m;
  uint32_t tile_col = blockIdx.x * Ctx::xtileN;
  uint32_t num_k_tiles = K / Ctx::tileK;
  uint32_t tile_row_w = blockIdx.y * num_warps + warp_rank;

  auto smem = reinterpret_cast<uint8_t*>(__local_mem());
  const uint32_t stage_bytes = wg_sp_stage_bytes(cta_m, Ctx::tileK, Ctx::xtileN, Ctx::xtileM);
  const uint32_t b_off = wg_sp_stage_b_off(cta_m, Ctx::tileK);
  auto stage_a = [&](uint32_t s) {
    return reinterpret_cast<in_t*>(smem + s * stage_bytes);
  };
  auto stage_b = [&](uint32_t s) {
    return reinterpret_cast<in_t*>(smem + s * stage_bytes + b_off);
  };
#ifdef AKILI_META_DXA
  const uint32_t meta_off = wg_sp_stage_meta_off(cta_m, Ctx::tileK, Ctx::xtileN);
  auto stage_meta = [&](uint32_t s) {
    return reinterpret_cast<const uint32_t*>(smem + s * stage_bytes + meta_off);
  };
  constexpr uint32_t stage_tx = 3;
#else
  constexpr uint32_t stage_tx = 2;
#endif

  const uint32_t cta_id = get_local_group_id();
  auto bar_id = [&](uint32_t s) { return cta_id | (s << 8); };
  const bool is_dxa_warp = (get_sub_group_id() == 0);
  const uint32_t num_steps = (num_k_tiles + kWgKB - 1) / kWgKB;

  auto issue = [&](uint32_t s, uint32_t k) {
    vx_barrier_expect_tx(bar_id(s), stage_tx);
#ifdef AKILI_A_BLOCKED
    static_assert(bk / 2 == kABlockCols, "blocked A needs one column-block per stage");
    vx_dxa_issue_2d_wg(desc_a, bar_id(s), stage_a(s), (k / 2) * kABlockRows, tile_row / kABlockRows);
#else
    vx_dxa_issue_2d_wg(desc_a, bar_id(s), stage_a(s), k / 2, tile_row);
#endif
    wg_issue_b<B_KN>(desc_b, bar_id(s), stage_b(s), k, tile_col);
#ifdef AKILI_META_DXA
    vx_dxa_issue_2d_wg(wg_meta_desc(desc_a), bar_id(s),
                       const_cast<uint32_t*>(stage_meta(s)),
                       (k / bk) * kWgKB * meta_words, blockIdx.y * num_warps);
#endif
  };

  typename Ctx::fragment_acc frag_c;
  Ctx::fill_fragment(frag_c, 0);

  if (is_dxa_warp) {
    for (uint32_t s = 0; s < kWgPipeN - 1 && s < num_steps; ++s)
      issue(s, s * bk);
  }

  for (uint32_t i = 0; i < num_steps; ++i) {
    const uint32_t cur = i % kWgPipeN;
    const uint32_t issue_i = i + kWgPipeN - 1;
    if (issue_i < num_steps && is_dxa_warp)
      issue(issue_i % kWgPipeN, issue_i * bk);

    vx_barrier(bar_id(cur), num_warps);

    for (uint32_t ki = 0; ki < kWgKB; ++ki) {
      const uint32_t kt = i * kWgKB + ki;
      if (kt >= num_k_tiles)
        break;
      typename Ctx::fragment_a frag_a;
#ifdef AKILI_META_DXA
      (void)metadata; (void)tile_row_w;
      Ctx::load_sp_metadata(frag_a,
          stage_meta(cur) + (warp_rank * kWgKB + ki) * meta_words);
#else
      Ctx::load_sp_metadata(frag_a,
          metadata + (tile_row_w * num_k_tiles + kt) * meta_words);
#endif
      auto a_warp = stage_a(cur) + warp_rank * Ctx::xtileM * (bk / 2)
                  + ki * (Ctx::tileK / 2);
      auto desc_a_smem = vortex::tensor::vx_make_smem_desc(
          a_warp, (bk / 2) * sizeof(in_t));
      auto desc_b_smem = vortex::tensor::vx_make_smem_desc(
          stage_b(cur) + ki * Ctx::tileK, bk * sizeof(in_t));
      Ctx::wgmma_sync(frag_c, desc_a_smem, desc_b_smem, frag_c);
    }

    vx_barrier(bar_id(cur), num_warps);
  }

  auto tile_c = output
      + (tile_row + warp_rank * Ctx::xtileM) * output_stride + tile_col;
  Ctx::store_matrix_sync(tile_c, frag_c, output_stride);
}

} // namespace akili

#endif
