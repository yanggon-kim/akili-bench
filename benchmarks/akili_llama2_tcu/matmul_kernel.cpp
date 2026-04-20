#include <vx_spawn2.h>
#include <vx_tensor.h>
#include "common.h"

// Task C instrumentation (guarded by AKILI_NONDXA_INSTRUMENT). Mirrors the
// DXA instrumentation in akili_llama2_tcu_dxa so per-iter compute (load_A +
// load_B + mma) is directly comparable to the DXA kernel's post-barrier
// compute region.
#ifdef AKILI_NONDXA_INSTRUMENT
#include <vx_print.h>
#include <vx_intrinsics.h>
#endif

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

extern "C" void kernel_main(matmul_kernel_args_t* __UNIFORM__ args) {
    auto A = reinterpret_cast<ctx::input_t*>(args->A_addr);
    auto B = reinterpret_cast<ctx::input_t*>(args->B_addr);
    auto C = reinterpret_cast<ctx::output_t*>(args->C_addr);

    ctx::fragment_a fragA;
    ctx::fragment_b fragB;
    ctx::fragment_acc fragC;

    uint32_t tile_row = blockIdx.y * ctx::tileM;
    uint32_t tile_col = blockIdx.x * ctx::tileN;

    ctx::fill_fragment(fragC, 0);

#ifdef AKILI_NONDXA_INSTRUMENT
    // Finer breakdown:
    //   gmem2r — DRAM→register: load_matrix_sync(A) + load_matrix_sync(B)
    //   tcu    — mma_sync (register→TCU execution)
    uint32_t acc_gmem2r = 0, acc_tcu = 0;
    uint32_t n_iter = 0;
    uint32_t t_start = csr_read(0xB00);
#endif

    for (uint32_t k = 0; k < args->K; k += ctx::tileK) {
        auto tileA = A + tile_row * args->K + k;
        auto tileB = B + k * args->N + tile_col;

#ifdef AKILI_NONDXA_INSTRUMENT
        uint32_t t0 = csr_read(0xB00);
#endif
        ctx::load_matrix_sync(fragA, tileA, args->K);
        ctx::load_matrix_sync(fragB, tileB, args->N);
#ifdef AKILI_NONDXA_INSTRUMENT
        uint32_t t0b = csr_read(0xB00);
#endif
        ctx::mma_sync(fragC, fragA, fragB, fragC);
#ifdef AKILI_NONDXA_INSTRUMENT
        uint32_t t1 = csr_read(0xB00);
        acc_gmem2r += (t0b - t0);
        acc_tcu    += (t1 - t0b);
        ++n_iter;
#endif
    }

#ifdef AKILI_NONDXA_INSTRUMENT
    uint32_t t_kloop_end = csr_read(0xB00);
#endif

    auto tileC = C + tile_row * args->N + tile_col;
    ctx::store_matrix_sync(tileC, fragC, args->N);

#ifdef AKILI_NONDXA_INSTRUMENT
    uint32_t t_store_end = csr_read(0xB00);
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        vx_printf("TCU_DENSE_NDXA[K=%u,niter=%u] kloop=%u store=%u | avg/iter gmem2r=%u tcu=%u\n",
                  args->K, n_iter,
                  t_kloop_end - t_start,
                  t_store_end - t_kloop_end,
                  n_iter ? acc_gmem2r / n_iter : 0,
                  n_iter ? acc_tcu    / n_iter : 0);
    }
#endif
}
