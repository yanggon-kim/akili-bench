// akili_acccnn_tcu_dxa — DXA-staged smem variant of akili_acccnn_tcu.
// A [M×K] row-major, B [N×K] col-major (host im2col output), C [M×N] fp32.
// Mirrors sgemm_tcu_smem_dxa.

#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>
#include "wgmma_dxa_kernel.h"
#include "common.h"

namespace vt = vortex::tensor;
using ctx = vt::wgmma_context<VX_CFG_NUM_TCU_LANES, vt::fp16, vt::fp32, false,
                              WGMMA_NRC>;

constexpr uint32_t kDescA = 0;
constexpr uint32_t kDescB = 1;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pC = reinterpret_cast<ctx::output_t*>(arg->O_addr);

  akili::wgmma_dxa<ctx>(pC, arg->N_gemm, arg->K_gemm, kDescA, kDescB);
}
