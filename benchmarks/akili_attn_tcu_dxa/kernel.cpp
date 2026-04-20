// akili_attn_tcu_dxa / akili_flash_tcu_dxa — DXA-staged smem variant of the
// dense TCU attention kernel. Three stages dispatched via arg->kernel_id:
//   KID_QK_TCU  — dense TCU GEMM S = Q·Kᵀ  (DXA + smem, A=Q / B=K)
//   KID_SOFTMAX — SIMT row-wise softmax (no DXA, no smem)
//   KID_PV_TCU  — dense TCU GEMM O = P·V   (DXA + smem, A=P / B=V)
//
// Separate DXA descriptor slots per GEMM stage so host can program them once.

#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>
#include "common.h"
#include <cmath>
#include <algorithm>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_TCU_LANES, vt::fp16, vt::fp32, false>;

// Descriptor slots.
constexpr uint32_t kDescA_QK = 0;
constexpr uint32_t kDescB_QK = 1;
constexpr uint32_t kDescA_PV = 2;
constexpr uint32_t kDescB_PV = 3;

// Fast fp32 exp: Taylor(x/32) then y^32. Branchless, fp32-only.
// Avoids libm expf's softfloat-double fallback on rv32if.
static inline float fast_exp(float x) {
  x = x < -80.0f ? -80.0f : x;
  float u = x * 0.03125f;
  float y = 1.0f + u*(1.0f + u*(0.5f + u*(0.166666667f + u*(0.041666667f + u*0.008333333f))));
  y = y*y; y = y*y; y = y*y; y = y*y; y = y*y;
  return y;
}

// =============================================================================
// Stage 2: SIMT softmax on fp32 S. One thread per row. No DXA, no smem.
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
      float e = fast_exp(S[row * N + col] - max_val);
      local_P[col] = e;
      exp_sum += e;
    }
    for (uint32_t col = 0; col < N; ++col)
      P[row * N + col] = local_P[col] / exp_sum;
  }
}

// =============================================================================
// DXA-staged dense GEMM bodies, inlined directly into each stage.
// The previous helper `dense_mma_dxa(…, A_desc, B_desc)` took descriptor slots
// as runtime arguments, which prevented full inlining and forced per-k-iter
// call overhead. Inlining each stage mirrors the `llama2_tcu_dxa` pattern that
// wins DXA speedup at NT=4/8, and matches the earlier sparse_mma_dxa inline
// fix applied to the _sp_dxa counterparts.
// =============================================================================
static void qk_tcu_body(kernel_arg_t* __UNIFORM__ arg) {
  auto pC_base = reinterpret_cast<ctx::output_t*>(arg->S_addr);
  uint32_t K_walk = arg->d;
  uint32_t c_stride = arg->N;

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragC;

  uint32_t tile_row = blockIdx.y * ctx::tileM;
  uint32_t tile_col = blockIdx.x * ctx::tileN;
  ctx::fill_fragment(fragC, 0);

  auto smem   = reinterpret_cast<ctx::input_t*>(__local_mem());
  auto A_smem = smem;
  auto B_smem = smem + ctx::tileM * ctx::tileK;

  vortex::barrier bar(0);
  const bool is_dxa_warp = (csr_read(VX_CSR_CTA_RANK) == 0);

  for (uint32_t i = 0; i < K_walk; i += ctx::tileK) {
    if (is_dxa_warp) {
      vx_dxa_issue_2d_wg(kDescA_QK, bar.id(), A_smem, i, tile_row);
      vx_dxa_issue_2d_wg(kDescB_QK, bar.id(), B_smem, i, tile_col);
    }
    bar.arrive_and_wait();
    ctx::load_matrix_sync(fragA, A_smem, ctx::tileK);
    ctx::load_matrix_sync<vt::col_major>(fragB, B_smem, ctx::tileK);
    ctx::mma_sync(fragC, fragA, fragB, fragC);
    bar.arrive_and_wait();
  }

  auto pTileC = pC_base + tile_row * c_stride + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, c_stride);
}

static void pv_tcu_body(kernel_arg_t* __UNIFORM__ arg) {
  auto pC_base = reinterpret_cast<ctx::output_t*>(arg->O_addr);
  uint32_t K_walk = arg->N;
  uint32_t c_stride = arg->d;

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragC;

  uint32_t tile_row = blockIdx.y * ctx::tileM;
  uint32_t tile_col = blockIdx.x * ctx::tileN;
  ctx::fill_fragment(fragC, 0);

  auto smem   = reinterpret_cast<ctx::input_t*>(__local_mem());
  auto A_smem = smem;
  auto B_smem = smem + ctx::tileM * ctx::tileK;

  vortex::barrier bar(0);
  const bool is_dxa_warp = (csr_read(VX_CSR_CTA_RANK) == 0);

  for (uint32_t i = 0; i < K_walk; i += ctx::tileK) {
    if (is_dxa_warp) {
      vx_dxa_issue_2d_wg(kDescA_PV, bar.id(), A_smem, i, tile_row);
      vx_dxa_issue_2d_wg(kDescB_PV, bar.id(), B_smem, i, tile_col);
    }
    bar.arrive_and_wait();
    ctx::load_matrix_sync(fragA, A_smem, ctx::tileK);
    ctx::load_matrix_sync<vt::col_major>(fragB, B_smem, ctx::tileK);
    ctx::mma_sync(fragC, fragA, fragB, fragC);
    bar.arrive_and_wait();
  }

  auto pTileC = pC_base + tile_row * c_stride + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, c_stride);
}

// =============================================================================
// Single entry point — host selects the active stage via arg->kernel_id.
// =============================================================================
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  switch (arg->kernel_id) {
    case KID_QK_TCU:  qk_tcu_body(arg);  break;
    case KID_SOFTMAX: softmax_body(arg); break;
    case KID_PV_TCU:  pv_tcu_body(arg);  break;
    default: break;
  }
}
