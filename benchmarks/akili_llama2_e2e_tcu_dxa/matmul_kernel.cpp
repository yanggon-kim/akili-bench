// llama2_tcu_dxa — DXA-staged smem variant of llama2_tcu.
// Dense TCU GEMM with:
//   A row-major [M × K], tile (tileM × tileK), stride K
//   B row-major [K × N], tile (tileK × tileN), stride N   (ROW-MAJOR B)
//   C row-major [M × N], tile (tileM × tileN), stride N
//
// Mirrors sgemm_tcu_smem_dxa but adapts the B descriptor / fragB load to
// row-major (llama2's weight-activation layout is already row-major for both
// operands, so we do not pre-transpose B on the host).

#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>
#include "wgmma_dxa_kernel.h"
#include "common.h"

namespace vt = vortex::tensor;
using ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, false,
                              WGMMA_NRC>;

// DXA descriptor slots (programmed by host in vxmath.cpp).
constexpr uint32_t kDescA = 0;
constexpr uint32_t kDescB = 1;

// ============================================================================
// DXA-sparse root-cause Phase 2 instrumentation (REMOVE AFTER INVESTIGATION).
// Measures per-region cycles for ONE representative CTA on ONE launch.
// Only CTA(0,0) lane 0 prints. Guarded by AKILI_DXA_INSTRUMENT macro.
// ============================================================================
#ifdef AKILI_DXA_INSTRUMENT
#include <vx_print.h>
#endif

__kernel void kernel_main(matmul_kernel_args_t* __UNIFORM__ args) {
    auto pC = reinterpret_cast<ctx::output_t*>(args->C_addr);
    akili::wgmma_dxa<ctx, true>(pC, args->N, args->K, kDescA, kDescB);
}
