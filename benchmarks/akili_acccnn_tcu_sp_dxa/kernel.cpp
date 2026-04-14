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
constexpr uint32_t kDescMeta = 2;  // packed 2:4 meta

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

  // smem layout: [A tile (tileM × tileK/2) fp16] [B tile (tileN × tileK) fp16] [Meta tile (per_k_tile_words) uint32]
  auto smem    = reinterpret_cast<ctx::input_t*>(__local_mem());
  auto A_smem  = smem;
  auto B_smem  = smem + ctx::tileM * stride_A_smem;
  auto Meta_smem = reinterpret_cast<uint32_t*>(B_smem + ctx::tileN * ctx::tileK);

  vortex::barrier bar(0);
  const bool is_dxa_warp = (csr_read(VX_CSR_CTA_RANK) == 0);

  uint32_t num_k_tiles = K / ctx::tileK;
  (void)num_k_tiles;

  for (uint32_t i = 0, kt = 0; i < K; i += ctx::tileK, ++kt) {
    if (is_dxa_warp) {
      // A: compressed, col-coord is in units of fp16 elements in the compressed
      // buffer, so k_compressed = i / 2.
      vx_dxa_issue_2d_wg(kDescA,    bar.id(), A_smem,    i / 2,                    tile_row);
      vx_dxa_issue_2d_wg(kDescB,    bar.id(), B_smem,    i,                        tile_col);
      vx_dxa_issue_2d_wg(kDescMeta, bar.id(), Meta_smem, kt * per_k_tile_words,    blockIdx.y);
    }
    bar.arrive_and_wait();

    ctx::load_matrix_sync<vt::row_major>(fragA, A_smem, stride_A_smem, nullptr, Meta_smem);
    ctx::load_matrix_sync<vt::col_major>(fragB, B_smem, ctx::tileK);
    ctx::mma_sync(fragC, fragA, fragB, fragC);

    bar.arrive_and_wait();
  }

  auto pTileC = pC + tile_row * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}
