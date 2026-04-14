#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_tensor.h>
#include "common.h"

namespace vt = vortex::tensor;
using tcu_ctx = vt::wmma_context<NUM_TCU_LANES, vt::fp16, vt::fp32, false>;

// =============================================================================
// akili_acccnn_tcu — Dense TCU conv2d via GEMM over (W_gemm × Icol).
// One output tile per block. KMU-dispatched per-CTA via vx_start_g.
// =============================================================================
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<tcu_ctx::input_t*>(arg->W_addr);
  auto pB = reinterpret_cast<tcu_ctx::input_t*>(arg->B_addr);
  auto pC = reinterpret_cast<tcu_ctx::output_t*>(arg->O_addr);

  uint32_t M = arg->M_gemm;
  uint32_t N = arg->N_gemm;
  uint32_t K = arg->K_gemm;
  (void)M;

  tcu_ctx::fragment_a   fragA;
  tcu_ctx::fragment_b   fragB;
  tcu_ctx::fragment_acc fragC;

  uint32_t tile_row = blockIdx.y * tcu_ctx::tileM;
  uint32_t tile_col = blockIdx.x * tcu_ctx::tileN;
  tcu_ctx::fill_fragment(fragC, 0);

  for (int i = 0; i < (int)K; i += (int)tcu_ctx::tileK) {
    auto pTileA = pA + tile_row * K + i;
    auto pTileB = pB + tile_col * K + i;
    tcu_ctx::load_matrix_sync(fragA, pTileA, K);
    tcu_ctx::load_matrix_sync<vt::col_major>(fragB, pTileB, K);
    tcu_ctx::mma_sync(fragC, fragA, fragB, fragC);
  }

  auto pTileC = pC + tile_row * N + tile_col;
  tcu_ctx::store_matrix_sync(pTileC, fragC, N);
}
