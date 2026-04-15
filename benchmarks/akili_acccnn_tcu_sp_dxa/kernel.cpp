// akili_acccnn_tcu_sp_dxa — DXA-staged smem variant of akili_acccnn_tcu_sp.
// Sparse TCU conv2d (2:4 on W) with compressed A, col-major B, packed meta.
// A is fp16 [M×K/2], B is fp16 [N×K] (col-major view), Meta is uint32 words.
// Mirrors sgemm_tcu_sp_smem_dxa.

#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>
#include "common.h"

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_TCU_LANES, vt::fp16, vt::fp32, true>;

constexpr uint32_t kDescA    = 0;  // compressed A
constexpr uint32_t kDescB    = 1;
// Option B: no kDescMeta — metadata is read directly from DDR via the
// load_matrix_sync fast-path, not staged through LMEM.

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
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

  constexpr uint32_t stride_A_smem = ctx::tileK / 2;

  // smem layout: [A tile (tileM × tileK/2) fp16] [B tile (tileN × tileK) fp16]
  // Option B: no LMEM region for Meta — it stays in DDR.
  auto smem   = reinterpret_cast<ctx::input_t*>(__local_mem());
  auto A_smem = smem;
  auto B_smem = smem + ctx::tileM * stride_A_smem;

  vortex::barrier bar(0);
  const bool is_dxa_warp = (csr_read(VX_CSR_CTA_RANK) == 0);

  uint32_t num_k_tiles = K / ctx::tileK;
  // Option B: walk a DDR metadata pointer, matching the WMMA reference.
  auto pMetaDDR = reinterpret_cast<const float*>(arg->meta_W_addr)
                + blockIdx.y * num_k_tiles * per_k_tile_words;

  for (uint32_t i = 0; i < K; i += ctx::tileK) {
    if (is_dxa_warp) {
      vx_dxa_issue_2d_wg(kDescA, bar.id(), A_smem, i / 2, tile_row);
      vx_dxa_issue_2d_wg(kDescB, bar.id(), B_smem, i,     tile_col);
    }
    bar.arrive_and_wait();

    ctx::load_matrix_sync<vt::row_major>(fragA, A_smem, stride_A_smem, nullptr, pMetaDDR);
    ctx::load_matrix_sync<vt::col_major>(fragB, B_smem, ctx::tileK);
    ctx::mma_sync(fragC, fragA, fragB, fragC);

    pMetaDDR += per_k_tile_words;
    bar.arrive_and_wait();
  }

  auto pTileC = pC + tile_row * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}
