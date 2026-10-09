// llama2_tcu_sp_dxa — Task 8 HYBRID variant:
// Use DXA only for matrix B (SMEM-staged). Load compressed-A + meta directly
// from gmem via inline load_matrix_sync (same pattern as non-DXA sparse kernel).
//
//   A compressed row-major [M × K/2]  — inline gmem load (LSU → DCache/L2)
//   B dense     row-major [K × N]     — DXA-staged into SMEM
//   Meta (packed uint32)              — inline gmem load via pMetaCur
//   C dense     row-major [M × N]

#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>
#include "wgmma_dxa_kernel.h"
#include "common.h"

namespace vt = vortex::tensor;
using ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, true,
                              WGMMA_NRC>;

// DXA descriptor slots (programmed by host in vxmath.cpp).
constexpr uint32_t kDescA = 0;
constexpr uint32_t kDescB = 1;

__kernel void kernel_main(matmul_kernel_args_t* __UNIFORM__ args) {
    auto pC = reinterpret_cast<ctx::output_t*>(args->C_addr);
    auto meta = reinterpret_cast<const uint32_t*>(args->meta_sp_addr);
    akili::sparse_wgmma_dxa<ctx, true>(pC, meta, args->N, args->K,
                                  kDescA, kDescB);
}
