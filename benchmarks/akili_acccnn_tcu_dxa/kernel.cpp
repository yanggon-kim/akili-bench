// akili_acccnn_tcu_dxa — DXA-staged smem variant of akili_acccnn_tcu.
// A [M×K] row-major, B [N×K] col-major (host im2col output), C [M×N] fp32.
// Mirrors sgemm_tcu_smem_dxa.

#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>
#include "common.h"

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_TCU_LANES, vt::fp16, vt::fp32, false>;

constexpr uint32_t kDescA = 0;
constexpr uint32_t kDescB = 1;

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

  auto smem   = reinterpret_cast<ctx::input_t*>(__local_mem());
  auto A_smem = smem;
  auto B_smem = smem + ctx::tileM * ctx::tileK;

  vortex::barrier bar(0);
  const bool is_dxa_warp = (csr_read(VX_CSR_CTA_RANK) == 0);

  for (uint32_t i = 0; i < K; i += ctx::tileK) {
    if (is_dxa_warp) {
      vx_dxa_issue_2d_wg(kDescA, bar.id(), A_smem, i, tile_row);
      vx_dxa_issue_2d_wg(kDescB, bar.id(), B_smem, i, tile_col);
    }
    bar.arrive_and_wait();

    ctx::load_matrix_sync(fragA, A_smem, ctx::tileK);
    ctx::load_matrix_sync<vt::col_major>(fragB, B_smem, ctx::tileK);
    ctx::mma_sync(fragC, fragA, fragB, fragC);

    bar.arrive_and_wait();
  }

  auto pTileC = pC + tile_row * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}
