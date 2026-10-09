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
#include "wgmma_dxa_kernel.h"
#include "common.h"

namespace vt = vortex::tensor;
using ctx = vt::wgmma_context<VX_CFG_NUM_TCU_LANES, vt::fp16, vt::fp32, true,
                              WGMMA_NRC>;

constexpr uint32_t kDescA = 0;
constexpr uint32_t kDescB = 1;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pC = reinterpret_cast<ctx::output_t*>(arg->O_addr);
  auto meta = reinterpret_cast<const uint32_t*>(arg->meta_W_addr);
  akili::sparse_wgmma_dxa<ctx>(pC, meta, arg->N_gemm, arg->K_gemm,
                                kDescA, kDescB);
}
