// akili_attn_tcu_sp_dxa / akili_flash_tcu_sp_dxa — DXA-staged smem variant of
// the sparse TCU attention kernel. Three stages dispatched via arg->kernel_id:
//   KID_QK_SPARSE — sparse TCU GEMM S = Q·Kᵀ  (DXA + smem, A=Qsp / B=K / Meta=Qmeta)
//   KID_SOFTMAX   — SIMT row-wise softmax (no DXA, no smem)
//   KID_PV_SPARSE — sparse TCU GEMM O = P·V   (DXA + smem, A=Psp / B=V / Meta=Pmeta)
//
// Separate DXA descriptor slots per GEMM stage (6 total: A/B/Meta × QK,PV).

#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>
#include "common.h"
#include <cmath>
#include <algorithm>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_TCU_LANES, vt::fp16, vt::fp32, true>;

// Descriptor slots. Option B: no Meta descriptors — metadata is read
// directly from DDR via the load_matrix_sync fast-path.
constexpr uint32_t kDescA_QK    = 0;
constexpr uint32_t kDescB_QK    = 1;
constexpr uint32_t kDescA_PV    = 2;
constexpr uint32_t kDescB_PV    = 3;

// =============================================================================
// Stage 2: SIMT softmax on fp32 S. No DXA, no smem.
// =============================================================================
static void softmax_body(kernel_arg_t* __UNIFORM__ arg) {
  // Launched with grid={NUM_WARPS,1}, block={NUM_THREADS,1}: 1 CTA per warp,
  // full NUM_WARPS × NUM_THREADS hardware threads striped across N rows.
  // Matches baseline's vortex1 vx_spawn_threads parallelism (which distributes
  // N logical threads across all available HW threads). Earlier attempts with
  // grid={N,1}/block={1,1} paid ~70K cycles KMU dispatch overhead per CTA;
  // grid={1,1}/block={NUM_THREADS,1} only got 8-way parallelism (vs 64-way
  // baseline).
  auto S = reinterpret_cast<float*>(arg->S_addr);
  auto P = reinterpret_cast<float*>(arg->P_addr);
  uint32_t N = arg->N;

  uint32_t hw_tid = blockIdx.x * NUM_THREADS + threadIdx.x;
  uint32_t stride = gridDim.x * NUM_THREADS;

  for (uint32_t row = hw_tid; row < N; row += stride) {
    float max_val = S[row * N];
    for (uint32_t col = 1; col < N; ++col)
      max_val = std::max(max_val, S[row * N + col]);

    float local_P[512];
    float exp_sum = 0;
    for (uint32_t col = 0; col < N; ++col) {
      float e = std::exp(S[row * N + col] - max_val);
      local_P[col] = e;
      exp_sum += e;
    }
    for (uint32_t col = 0; col < N; ++col)
      P[row * N + col] = local_P[col] / exp_sum;
  }
}

// =============================================================================
// Shared DXA-staged sparse GEMM helper.
//   A is compressed [M × K/2] row-major, B is col-major (K_walk), Meta is
//   the packed 2:4 metadata per (tile_row, k_tile) in uint32 words.
//   A_desc / B_desc / Meta_desc select the DXA descriptor slots for this stage.
// =============================================================================
static inline void sparse_mma_dxa(ctx::input_t*  pA_unused,
                                  ctx::input_t*  pB_unused,
                                  ctx::output_t* pC_base,
                                  const float*   pMeta_base,
                                  uint32_t K_walk, uint32_t c_stride,
                                  uint32_t A_desc, uint32_t B_desc) {
  (void)pA_unused; (void)pB_unused;

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragC;

  uint32_t tile_row = blockIdx.y * ctx::tileM;
  uint32_t tile_col = blockIdx.x * ctx::tileN;
  ctx::fill_fragment(fragC, 0);

  // Meta sizing — matches host pack_metadata.
  constexpr uint32_t rtl_i_ratio = 32 / vt::fp16::bits;
  constexpr uint32_t meta_cols = (NUM_TCU_LANES * 2 * rtl_i_ratio + 31) / 32;
  using kcfg = vt::wmma_config_t<NUM_TCU_LANES>;
  constexpr uint32_t PD = kcfg::m_steps * (kcfg::k_steps / 2);
  constexpr uint32_t num_meta_loads = (PD * meta_cols + NUM_TCU_LANES - 1) / NUM_TCU_LANES;
  constexpr uint32_t per_k_tile_words = num_meta_loads * NUM_TCU_LANES;

  constexpr uint32_t stride_A_smem = ctx::tileK / 2;

  // Option B: smem layout is just [A tile][B tile] — Meta stays in DDR.
  auto smem   = reinterpret_cast<ctx::input_t*>(__local_mem());
  auto A_smem = smem;
  auto B_smem = smem + ctx::tileM * stride_A_smem;

  vortex::barrier bar(0);
  const bool is_dxa_warp = (csr_read(VX_CSR_CTA_RANK) == 0);

  // Option B: per-CTA DDR Meta pointer — walks per k-tile.
  uint32_t num_k_tiles = K_walk / ctx::tileK;
  (void)num_k_tiles;
  auto pMetaDDR = pMeta_base + blockIdx.y * num_k_tiles * per_k_tile_words;

  for (uint32_t i = 0; i < K_walk; i += ctx::tileK) {
    if (is_dxa_warp) {
      vx_dxa_issue_2d_wg(A_desc, bar.id(), A_smem, i / 2, tile_row);
      vx_dxa_issue_2d_wg(B_desc, bar.id(), B_smem, i,     tile_col);
    }
    bar.arrive_and_wait();

    ctx::load_matrix_sync<vt::row_major>(fragA, A_smem, stride_A_smem, nullptr, pMetaDDR);
    ctx::load_matrix_sync<vt::col_major>(fragB, B_smem, ctx::tileK);
    ctx::mma_sync(fragC, fragA, fragB, fragC);

    pMetaDDR += per_k_tile_words;
    bar.arrive_and_wait();
  }

  auto pTileC = pC_base + tile_row * c_stride + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, c_stride);
}

// =============================================================================
// Stage 1: sparse TCU S = Q·Kᵀ   (K_walk = d_attn)
// =============================================================================
static void qk_sparse_body(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<ctx::input_t*>(arg->Q_addr);
  auto pB = reinterpret_cast<ctx::input_t*>(arg->K_addr);
  auto pC = reinterpret_cast<ctx::output_t*>(arg->S_addr);
  auto pM = reinterpret_cast<const float*>(arg->meta_Q_addr);
  sparse_mma_dxa(pA, pB, pC, pM, arg->d, arg->N,
                 kDescA_QK, kDescB_QK);
}

// =============================================================================
// Stage 3: sparse TCU O = P·V   (K_walk = N_attn)
// =============================================================================
static void pv_sparse_body(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<ctx::input_t*>(arg->P_addr);
  auto pB = reinterpret_cast<ctx::input_t*>(arg->V_addr);
  auto pC = reinterpret_cast<ctx::output_t*>(arg->O_addr);
  auto pM = reinterpret_cast<const float*>(arg->meta_P_addr);
  sparse_mma_dxa(pA, pB, pC, pM, arg->N, arg->d,
                 kDescA_PV, kDescB_PV);
}

// =============================================================================
// Single entry point — host selects the active stage via arg->kernel_id.
// =============================================================================
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  switch (arg->kernel_id) {
    case KID_QK_SPARSE: qk_sparse_body(arg); break;
    case KID_SOFTMAX:   softmax_body(arg);   break;
    case KID_PV_SPARSE: pv_sparse_body(arg); break;
    default: break;
  }
}
