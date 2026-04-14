#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_tensor.h>
#include "common.h"

namespace vt = vortex::tensor;
using tcu_ctx = vt::wmma_context<NUM_TCU_LANES, vt::fp16, vt::fp32, false>;

// =============================================================================
// akili_acccnn_tcu — Dense TCU conv2d via GEMM over (W_gemm × Icol).
// Mirrors tests/regression/sgemm_tcu/kernel.cpp. One output tile per block.
// =============================================================================
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  __rdcycle_time t0 = vx_rdcycle_sync_begin();

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

  __rdcycle_time t1 = vx_rdcycle_sync_end();
  if (threadIdx.x == 0) {
    auto pCycles = reinterpret_cast<uint32_t*>(arg->cycles_addr);
    uint32_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
    pCycles[block_id] = (uint32_t)vx_rdcycle_sync_diff(t0, t1);
  }
}
