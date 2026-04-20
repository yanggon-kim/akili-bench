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

// ============================================================================
// DXA-sparse root-cause Phase 2 instrumentation (REMOVE AFTER INVESTIGATION).
// Measures per-region cycles for ONE representative CTA on ONE launch.
// Only CTA(0,0) lane 0 prints. Guarded by AKILI_DXA_INSTRUMENT macro.
// ============================================================================
#ifdef AKILI_DXA_INSTRUMENT
#include <vx_print.h>
#endif

extern "C" void kernel_main(matmul_kernel_args_t* __UNIFORM__ args) {
    auto pC = reinterpret_cast<ctx::output_t*>(args->C_addr);

    uint32_t N = args->N;
    uint32_t K = args->K;

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

#ifdef AKILI_DXA_INSTRUMENT
    // Finer-grained per-iter breakdown:
    //   issue   — DXA issue instructions
    //   bar1    — DRAM→SMEM wait (first arrive_and_wait after issue)
    //   s2r     — SMEM→register: the two load_matrix_sync calls
    //   tcu     — mma_sync (register→TCU execution)
    //   bar2    — release barrier at end of iter
    uint32_t acc_issue = 0, acc_bar1 = 0, acc_s2r = 0, acc_tcu = 0, acc_bar2 = 0;
    uint32_t n_iter = 0;
    uint32_t t_start = csr_read(0xB00);
#endif

    for (uint32_t k = 0; k < K; k += ctx::tileK) {
#ifdef AKILI_DXA_INSTRUMENT
        uint32_t t0 = csr_read(0xB00);
#endif
        if (is_dxa_warp) {
            vx_dxa_issue_2d_wg(kDescA, bar.id(), A_smem, k,        tile_row);
            vx_dxa_issue_2d_wg(kDescB, bar.id(), B_smem, tile_col, k);
        }
#ifdef AKILI_DXA_INSTRUMENT
        uint32_t t1 = csr_read(0xB00);
#endif

        bar.arrive_and_wait();
#ifdef AKILI_DXA_INSTRUMENT
        uint32_t t2 = csr_read(0xB00);
#endif

        ctx::load_matrix_sync(fragA, A_smem, ctx::tileK);
        ctx::load_matrix_sync(fragB, B_smem, ctx::tileN);
#ifdef AKILI_DXA_INSTRUMENT
        uint32_t t2b = csr_read(0xB00);
#endif
        ctx::mma_sync(fragC, fragA, fragB, fragC);
#ifdef AKILI_DXA_INSTRUMENT
        uint32_t t3 = csr_read(0xB00);
#endif

        bar.arrive_and_wait();
#ifdef AKILI_DXA_INSTRUMENT
        uint32_t t4 = csr_read(0xB00);
        acc_issue += (t1 - t0);
        acc_bar1  += (t2 - t1);
        acc_s2r   += (t2b - t2);
        acc_tcu   += (t3 - t2b);
        acc_bar2  += (t4 - t3);
        ++n_iter;
#endif
    }

#ifdef AKILI_DXA_INSTRUMENT
    uint32_t t_kloop_end = csr_read(0xB00);
#endif

    auto pTileC = pC + tile_row * N + tile_col;
    ctx::store_matrix_sync(pTileC, fragC, N);

#ifdef AKILI_DXA_INSTRUMENT
    uint32_t t_store_end = csr_read(0xB00);
  #ifdef AKILI_ALL_CTAS
    #ifndef AKILI_CTA_Y_TARGET
    #define AKILI_CTA_Y_TARGET 0
    #endif
    if (blockIdx.x == 0 && blockIdx.y == AKILI_CTA_Y_TARGET && threadIdx.x == 0) {
        uint32_t v_K = K;
        uint32_t v_bx = blockIdx.x;
        uint32_t v_by = blockIdx.y;
        uint32_t v_total = t_store_end - t_start;
        uint32_t v_kloop = t_kloop_end - t_start;
        uint32_t v_store = t_store_end - t_kloop_end;
        vx_printf("DXA_DENSE_CTA K=%u bx=%u by=%u total=%u kloop=%u store=%u\n",
                  v_K, v_bx, v_by, v_total, v_kloop, v_store);
    }
  #else
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        vx_printf("DXA_DENSE[K=%u,niter=%u] kloop=%u store=%u | avg/iter issue=%u bar1=%u s2r=%u tcu=%u bar2=%u\n",
                  K, n_iter,
                  t_kloop_end - t_start,
                  t_store_end - t_kloop_end,
                  n_iter ? acc_issue / n_iter : 0,
                  n_iter ? acc_bar1  / n_iter : 0,
                  n_iter ? acc_s2r   / n_iter : 0,
                  n_iter ? acc_tcu   / n_iter : 0,
                  n_iter ? acc_bar2  / n_iter : 0);
    }
  #endif
#endif
}
