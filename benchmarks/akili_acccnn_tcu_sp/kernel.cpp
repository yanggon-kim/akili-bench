#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_tensor.h>
#include "common.h"

namespace vt = vortex::tensor;
using sp_ctx = vt::wmma_context<NUM_TCU_LANES, vt::fp16, vt::fp32, true>;

// =============================================================================
// akili_acccnn_tcu_sp — Sparse (2:4) TCU conv2d via im2col + sgemm_tcu_sp-style
// GEMM.  Host prunes W 2:4 along K_gemm, compresses, packs metadata.
// =============================================================================
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  __rdcycle_time t0 = vx_rdcycle_sync_begin();

  auto pA = reinterpret_cast<sp_ctx::input_t*>(arg->W_addr);
  auto pB = reinterpret_cast<sp_ctx::input_t*>(arg->B_addr);
  auto pC = reinterpret_cast<sp_ctx::output_t*>(arg->O_addr);
  auto pMetaSpBase = reinterpret_cast<const float*>(arg->meta_W_addr);

  uint32_t M = arg->M_gemm;
  uint32_t N = arg->N_gemm;
  uint32_t K = arg->K_gemm;
  (void)M;
  uint32_t stride_A = K / 2;
  (void)M;

  sp_ctx::fragment_a   fragA;
  sp_ctx::fragment_b   fragB;
  sp_ctx::fragment_acc fragC;

  uint32_t tile_row = blockIdx.y * sp_ctx::tileM;
  uint32_t tile_col = blockIdx.x * sp_ctx::tileN;
  sp_ctx::fill_fragment(fragC, 0);

  constexpr uint32_t rtl_i_ratio = 32 / vt::fp16::bits;
  constexpr uint32_t meta_cols = (NUM_TCU_LANES * 2 * rtl_i_ratio + 31) / 32;
  using kcfg = vt::wmma_config_t<NUM_TCU_LANES>;
  constexpr uint32_t PD = kcfg::m_steps * (kcfg::k_steps / 2);
  constexpr uint32_t num_meta_loads = (PD * meta_cols + NUM_TCU_LANES - 1) / NUM_TCU_LANES;
  constexpr uint32_t per_k_tile_words = num_meta_loads * NUM_TCU_LANES;

  uint32_t num_k_tiles = K / sp_ctx::tileK;
  uint32_t tile_row_idx = blockIdx.y;

  auto pMetaSp = pMetaSpBase + tile_row_idx * num_k_tiles * per_k_tile_words;
  auto pTileA = pA + tile_row * stride_A;
  constexpr uint32_t a_k_stride = sp_ctx::tileK / 2;

  // B as plain col-major (sgemm_tcu_sp pattern). pTileB advances by sp_ctx::tileK each K-iter.
  auto pTileB = pB + tile_col * K;
  for (int i = 0; i < (int)K; i += (int)sp_ctx::tileK) {
    sp_ctx::load_matrix_sync<vt::row_major>(fragA, pTileA, stride_A, nullptr, pMetaSp);
    sp_ctx::load_matrix_sync<vt::col_major>(fragB, pTileB, K);
    sp_ctx::mma_sync(fragC, fragA, fragB, fragC);
    pMetaSp += per_k_tile_words;
    pTileA  += a_k_stride;
    pTileB  += sp_ctx::tileK;
  }

  auto pTileC = pC + tile_row * N + tile_col;
  sp_ctx::store_matrix_sync(pTileC, fragC, N);

  __rdcycle_time t1 = vx_rdcycle_sync_end();
  if (threadIdx.x == 0) {
    auto pCycles = reinterpret_cast<uint32_t*>(arg->cycles_addr);
    uint32_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
    pCycles[block_id] = (uint32_t)vx_rdcycle_sync_diff(t0, t1);
  }
}
