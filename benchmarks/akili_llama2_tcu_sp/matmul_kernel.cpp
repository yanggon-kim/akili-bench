#include <vx_spawn.h>
#include <vx_tensor.h>

#include "common.h"

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE, true>;
using kcfg = vt::wmma_config_t<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

static void kernel_main(matmul_kernel_args_t* __UNIFORM__ args) {
    auto A = reinterpret_cast<ctx::input_t*>(args->A_addr);
    auto B = reinterpret_cast<ctx::input_t*>(args->B_addr);
    auto C = reinterpret_cast<ctx::output_t*>(args->C_addr);
    auto meta_base = reinterpret_cast<const float*>(args->meta_sp_addr);

    ctx::fragment_a fragA;
    ctx::fragment_b fragB;
    ctx::fragment_acc fragC;

    uint32_t tile_row = blockIdx.y * ctx::tileM;
    uint32_t tile_col = blockIdx.x * ctx::tileN;

    ctx::fill_fragment(fragC, 0);

    constexpr uint32_t pd = kcfg::m_steps * (kcfg::k_steps / 2);
    constexpr uint32_t meta_cols = kcfg::meta_cols;
    constexpr uint32_t num_meta_loads = (pd * meta_cols + NUM_THREADS - 1) / NUM_THREADS;
    constexpr uint32_t per_k_tile_words = num_meta_loads * NUM_THREADS;

    uint32_t num_k_tiles = args->K / ctx::tileK;
    uint32_t stride_A = args->K / 2;

    auto tileA = A + tile_row * stride_A;
    auto tileB = B + tile_col;
    auto tileMeta = meta_base + blockIdx.y * num_k_tiles * per_k_tile_words;

    for (uint32_t k = 0; k < args->K; k += ctx::tileK) {
        ctx::load_matrix_sync<vt::row_major>(fragA, tileA, stride_A, nullptr, tileMeta);
        ctx::load_matrix_sync<vt::row_major>(fragB, tileB, args->N);
        ctx::mma_sync(fragC, fragA, fragB, fragC);
        tileA += ctx::tileK / 2;
        tileB += ctx::tileK * args->N;
        tileMeta += per_k_tile_words;
    }

    auto tileC = C + tile_row * args->N + tile_col;
    ctx::store_matrix_sync(tileC, fragC, args->N);
}

int main() {
    auto arg = (matmul_kernel_args_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t grid_dim[2] = {arg->N / ctx::tileN, arg->M / ctx::tileM};
    uint32_t block_dim[2] = {NUM_THREADS, 1};
    return vx_spawn_threads(2, grid_dim, block_dim,
                            (vx_kernel_func_cb)kernel_main, arg);
}
