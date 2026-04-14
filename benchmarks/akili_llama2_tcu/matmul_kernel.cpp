#include <vx_spawn.h>
#include <vx_tensor.h>
#include "common.h"

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

static void kernel_main(matmul_kernel_args_t* __UNIFORM__ args) {
    auto A = reinterpret_cast<ctx::input_t*>(args->A_addr);
    auto B = reinterpret_cast<ctx::input_t*>(args->B_addr);
    auto C = reinterpret_cast<ctx::output_t*>(args->C_addr);

    ctx::fragment_a fragA;
    ctx::fragment_b fragB;
    ctx::fragment_acc fragC;

    uint32_t tile_row = blockIdx.y * ctx::tileM;
    uint32_t tile_col = blockIdx.x * ctx::tileN;

    ctx::fill_fragment(fragC, 0);

    for (uint32_t k = 0; k < args->K; k += ctx::tileK) {
        auto tileA = A + tile_row * args->K + k;
        auto tileB = B + k * args->N + tile_col;

        ctx::load_matrix_sync(fragA, tileA, args->K);
        ctx::load_matrix_sync(fragB, tileB, args->N);
        ctx::mma_sync(fragC, fragA, fragB, fragC);
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
