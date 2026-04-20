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
#include "common.h"

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE, true>;
using kcfg = vt::wmma_config_t<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

// DXA descriptor slots (programmed by host in vxmath.cpp).
// Only kDescB is used in the hybrid variant; kDescA and kDescMeta are unused.
constexpr uint32_t kDescA    = 0;  // unused in hybrid
constexpr uint32_t kDescB    = 1;  // dense row-major B (only DXA descriptor we use)
constexpr uint32_t kDescMeta = 2;  // unused in hybrid

extern "C" void kernel_main(matmul_kernel_args_t* __UNIFORM__ args) {
    auto pA = reinterpret_cast<ctx::input_t*>(args->A_addr);
    auto pC = reinterpret_cast<ctx::output_t*>(args->C_addr);
    auto pMetaBase = reinterpret_cast<const float*>(args->meta_sp_addr);

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

    uint32_t num_k_tiles = K / ctx::tileK;
    uint32_t stride_A = K / 2;

    // Inline gmem pointers for A and meta (non-DXA pattern, same as akili_llama2_tcu_sp).
    auto tileA    = pA + tile_row * stride_A;
    auto tileMeta = pMetaBase + blockIdx.y * num_k_tiles * per_k_tile_words;

    // SMEM: only B tile (tileK × tileN fp16). No A/meta SMEM regions.
    auto B_smem = reinterpret_cast<ctx::input_t*>(__local_mem());

    vortex::barrier bar(0);
    const bool is_dxa_warp = (csr_read(VX_CSR_CTA_RANK) == 0);

    for (uint32_t k = 0; k < K; k += ctx::tileK) {
        if (is_dxa_warp) {
            vx_dxa_issue_2d_wg(kDescB, bar.id(), B_smem, tile_col, k);
        }
        bar.arrive_and_wait();

        // A + meta loaded directly from gmem (non-DXA pattern).
        ctx::load_matrix_sync<vt::row_major>(fragA, tileA, stride_A, nullptr, tileMeta);
        // B loaded from SMEM (DXA-staged).
        ctx::load_matrix_sync<vt::row_major>(fragB, B_smem, ctx::tileN);
        ctx::mma_sync(fragC, fragA, fragB, fragC);

        tileA    += ctx::tileK / 2;
        tileMeta += per_k_tile_words;
        bar.arrive_and_wait();
    }

    auto pTileC = pC + tile_row * N + tile_col;
    ctx::store_matrix_sync(pTileC, fragC, N);
}
