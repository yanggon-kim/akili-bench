// llama2_tcu_sp_dxa — DXA-staged smem variant of llama2_tcu_sp.
// Sparse TCU GEMM (2:4 on A) with:
//   A compressed row-major [M × K/2]  (stride K/2), tile (tileM × tileK/2)
//   B dense     row-major [K × N]     (stride N),   tile (tileK × tileN)
//   Meta: all per-CTA 2:4 metadata is prefetched into SMEM via a single
//         DXA issue at CTA entry (Fix M2). The k-loop then reads meta
//         from SMEM instead of DDR, which otherwise dominated the LSU
//         stall at 80-cycle avg load latency (vs. 40-cycle dense).
//   C dense     row-major [M × N].

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
// Fix M2: re-add kDescMeta, but issued ONCE per CTA with a full per-M-tile
// meta slab. This is distinct from the original buggy pattern which issued
// kDescMeta per-k-iter inside the hot loop.
constexpr uint32_t kDescA    = 0;  // compressed A
constexpr uint32_t kDescB    = 1;  // dense row-major B
constexpr uint32_t kDescMeta = 2;  // full per-M-tile meta slab (issued once)

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

    // SMEM layout: [A tile][B tile][Meta (all k-tiles for this M-tile row)].
    auto smem      = reinterpret_cast<ctx::input_t*>(__local_mem());
    auto A_smem    = smem;
    auto B_smem    = smem + ctx::tileM * stride_A_smem;
    auto Meta_smem = reinterpret_cast<uint32_t*>(B_smem + ctx::tileK * ctx::tileN);

    vortex::barrier bar(0);
    const bool is_dxa_warp = (csr_read(VX_CSR_CTA_RANK) == 0);

    // Fix M2: one-shot DXA prefetch of the full per-M-tile meta slab
    // at CTA entry. Costs one extra barrier but keeps Meta DXA isolated
    // from the A/B DXA pipeline so the k-loop sees no Meta stall.
    if (is_dxa_warp) {
        vx_dxa_issue_2d_wg(kDescMeta, bar.id(), Meta_smem, 0, blockIdx.y);
    }
    bar.arrive_and_wait();

    auto pMetaCur = reinterpret_cast<const float*>(Meta_smem);

    for (uint32_t k = 0; k < K; k += ctx::tileK) {
        if (is_dxa_warp) {
            vx_dxa_issue_2d_wg(kDescA, bar.id(), A_smem, k / 2,    tile_row);
            vx_dxa_issue_2d_wg(kDescB, bar.id(), B_smem, tile_col, k);
        }
        bar.arrive_and_wait();

        ctx::load_matrix_sync<vt::row_major>(fragA, A_smem, stride_A_smem, nullptr, pMetaCur);
        ctx::load_matrix_sync<vt::row_major>(fragB, B_smem, ctx::tileN);
        ctx::mma_sync(fragC, fragA, fragB, fragC);

        pMetaCur += per_k_tile_words;  // const float* walks in 4-byte units
        bar.arrive_and_wait();
    }

    auto pTileC = pC + tile_row * N + tile_col;
    ctx::store_matrix_sync(pTileC, fragC, N);
}
