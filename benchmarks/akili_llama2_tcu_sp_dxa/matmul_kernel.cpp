// llama2_tcu_sp_dxa — DXA-staged smem variant of llama2_tcu_sp.
// Sparse TCU GEMM (2:4 on A) with:
//   A compressed row-major [M × K/2]  (stride K/2), tile (tileM × tileK/2)
//   B dense     row-major [K × N]     (stride N),   tile (tileK × tileN)
//   Meta packed: per M-tile row, one linear run of
//                num_k_tiles × per_k_tile_words uint32 words.
//   C dense     row-major [M × N].
//
// Mirrors akili_acccnn_tcu_sp_dxa but with row-major B (llama2 layout).

#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>
#include "common.h"

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE, true>;
using kcfg = vt::wmma_config_t<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

// DXA descriptor slots (programmed by host in vxmath.cpp).
constexpr uint32_t kDescA    = 0;  // compressed A
constexpr uint32_t kDescB    = 1;  // dense row-major B
constexpr uint32_t kDescMeta = 2;  // packed 2:4 meta

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

    // Meta sizing — matches host-side pack_metadata.
    constexpr uint32_t pd = kcfg::m_steps * (kcfg::k_steps / 2);
    constexpr uint32_t meta_cols = kcfg::meta_cols;
    constexpr uint32_t num_meta_loads = (pd * meta_cols + NUM_THREADS - 1) / NUM_THREADS;
    constexpr uint32_t per_k_tile_words = num_meta_loads * NUM_THREADS;

    constexpr uint32_t stride_A_smem = ctx::tileK / 2;

    // SMEM layout:
    //   A_smem   : [tileM × tileK/2] fp16 row-major (compressed)
    //   B_smem   : [tileK × tileN]   fp16 row-major
    //   Meta_smem: per_k_tile_words uint32 words
    auto smem      = reinterpret_cast<ctx::input_t*>(__local_mem());
    auto A_smem    = smem;
    auto B_smem    = smem + ctx::tileM * stride_A_smem;
    auto Meta_smem = reinterpret_cast<uint32_t*>(B_smem + ctx::tileK * ctx::tileN);

    vortex::barrier bar(0);
    const bool is_dxa_warp = (csr_read(VX_CSR_CTA_RANK) == 0);

    for (uint32_t k = 0, kt = 0; k < K; k += ctx::tileK, ++kt) {
        // Issue DXA copies in each descriptor's 2D index space:
        //   A compressed (M × K/2 row-major): (row=tile_row, col=k/2).
        //   B row-major  (K × N):             (row=k,        col=tile_col).
        //   Meta: one row per output M-tile, column stride per_k_tile_words.
        if (is_dxa_warp) {
            vx_dxa_issue_2d_wg(kDescA,    bar.id(), A_smem,    k / 2,                  tile_row);
            vx_dxa_issue_2d_wg(kDescB,    bar.id(), B_smem,    tile_col,               k);
            vx_dxa_issue_2d_wg(kDescMeta, bar.id(), Meta_smem, kt * per_k_tile_words,  blockIdx.y);
        }
        bar.arrive_and_wait();

        // Sparse fragA load consumes the packed meta tile from smem.
        // B fragment load is row-major (default), stride = tileN.
        ctx::load_matrix_sync<vt::row_major>(fragA, A_smem, stride_A_smem, nullptr, Meta_smem);
        ctx::load_matrix_sync<vt::row_major>(fragB, B_smem, ctx::tileN);
        ctx::mma_sync(fragC, fragA, fragB, fragC);

        bar.arrive_and_wait();
    }

    auto pTileC = pC + tile_row * N + tile_col;
    ctx::store_matrix_sync(pTileC, fragC, N);
}
