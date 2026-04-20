// akili_acccnn_tcu_sp_dxa — Task 10 HYBRID variant:
// Use DXA only for matrix B (SMEM-staged). Load compressed-A + meta directly
// from gmem via inline load_matrix_sync (same pattern as non-DXA sparse kernel).
//
//   W compressed row-major [M × K/2]  — inline gmem load (LSU → DCache/L2)
//   B col-major   [N × K]             — DXA-staged into SMEM
//   Meta (packed uint32)              — inline gmem load via pMetaDDR
//   O dense  row-major [M × N]

#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>
#include "common.h"

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_TCU_LANES, vt::fp16, vt::fp32, true>;

// Only kDescB is used in the hybrid variant; kDescA and kDescMeta are unused.
constexpr uint32_t kDescA    = 0;  // unused in hybrid
constexpr uint32_t kDescB    = 1;  // col-major B (only DXA descriptor we use)

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<ctx::input_t*>(arg->W_addr);
  auto pC = reinterpret_cast<ctx::output_t*>(arg->O_addr);

  uint32_t N = arg->N_gemm;
  uint32_t K = arg->K_gemm;

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragC;

  uint32_t tile_row = blockIdx.y * ctx::tileM;
  uint32_t tile_col = blockIdx.x * ctx::tileN;
  ctx::fill_fragment(fragC, 0);

  // Meta sizing — matches baseline pack_metadata.
  constexpr uint32_t rtl_i_ratio = 32 / vt::fp16::bits;
  constexpr uint32_t meta_cols = (NUM_TCU_LANES * 2 * rtl_i_ratio + 31) / 32;
  using kcfg = vt::wmma_config_t<NUM_TCU_LANES>;
  constexpr uint32_t PD = kcfg::m_steps * (kcfg::k_steps / 2);
  constexpr uint32_t num_meta_loads = (PD * meta_cols + NUM_TCU_LANES - 1) / NUM_TCU_LANES;
  constexpr uint32_t per_k_tile_words = num_meta_loads * NUM_TCU_LANES;

  uint32_t num_k_tiles = K / ctx::tileK;
  uint32_t stride_A = K / 2;

  // Inline gmem pointers for A and meta (non-DXA pattern).
  auto tileA    = pA + tile_row * stride_A;
  auto pMetaDDR = reinterpret_cast<const float*>(arg->meta_W_addr)
                + blockIdx.y * num_k_tiles * per_k_tile_words;

  // SMEM: only B tile (tileN × tileK fp16, col-major). No A/meta SMEM regions.
  auto B_smem = reinterpret_cast<ctx::input_t*>(__local_mem());

  vortex::barrier bar(0);
  const bool is_dxa_warp = (csr_read(VX_CSR_CTA_RANK) == 0);

  for (uint32_t i = 0; i < K; i += ctx::tileK) {
    if (is_dxa_warp) {
      vx_dxa_issue_2d_wg(kDescB, bar.id(), B_smem, i, tile_col);
    }
    bar.arrive_and_wait();

    // A + meta loaded directly from gmem (non-DXA pattern).
    ctx::load_matrix_sync<vt::row_major>(fragA, tileA, stride_A, nullptr, pMetaDDR);
    // B loaded from SMEM (DXA-staged).
    ctx::load_matrix_sync<vt::col_major>(fragB, B_smem, ctx::tileK);
    ctx::mma_sync(fragC, fragA, fragB, fragC);

    tileA    += ctx::tileK / 2;
    pMetaDDR += per_k_tile_words;
    bar.arrive_and_wait();
  }

  auto pTileC = pC + tile_row * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}
