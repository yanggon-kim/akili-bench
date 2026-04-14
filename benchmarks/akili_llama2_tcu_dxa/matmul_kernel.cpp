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
#include "common.h"

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

// DXA descriptor slots (programmed by host in vxmath.cpp).
constexpr uint32_t kDescA = 0;
constexpr uint32_t kDescB = 1;

extern "C" void kernel_main(matmul_kernel_args_t* __UNIFORM__ args) {
    auto pC = reinterpret_cast<ctx::output_t*>(args->C_addr);

    uint32_t N = args->N;
    uint32_t K = args->K;

    ctx::fragment_a   fragA;
    ctx::fragment_b   fragB;
    ctx::fragment_acc fragC;

    // CTA tile origin in the output matrix.
    uint32_t tile_row = blockIdx.y * ctx::tileM;
    uint32_t tile_col = blockIdx.x * ctx::tileN;

    ctx::fill_fragment(fragC, 0);

    // SMEM layout (both row-major, matching llama2 global layouts):
    //   A_smem: [tileM × tileK] row-major, stride = tileK
    //   B_smem: [tileK × tileN] row-major, stride = tileN
    auto smem   = reinterpret_cast<ctx::input_t*>(__local_mem());
    auto A_smem = smem;
    auto B_smem = smem + ctx::tileM * ctx::tileK;

    vortex::barrier bar(0);
    const bool is_dxa_warp = (csr_read(VX_CSR_CTA_RANK) == 0);

    for (uint32_t k = 0; k < K; k += ctx::tileK) {
        // Issue DXA copies in the descriptor's 2D index space:
        //   A (row-major M×K): tile at (row=tile_row, col=k).
        //   B (row-major K×N): tile at (row=k,        col=tile_col).
        if (is_dxa_warp) {
            vx_dxa_issue_2d_wg(kDescA, bar.id(), A_smem, k,        tile_row);
            vx_dxa_issue_2d_wg(kDescB, bar.id(), B_smem, tile_col, k);
        }

        bar.arrive_and_wait();

        // Fragment loads + MMA from smem. B is row-major here (default),
        // stride = tileN.
        ctx::load_matrix_sync(fragA, A_smem, ctx::tileK);
        ctx::load_matrix_sync(fragB, B_smem, ctx::tileN);
        ctx::mma_sync(fragC, fragA, fragB, fragC);

        bar.arrive_and_wait();
    }

    auto pTileC = pC + tile_row * N + tile_col;
    ctx::store_matrix_sync(pTileC, fragC, N);
}
