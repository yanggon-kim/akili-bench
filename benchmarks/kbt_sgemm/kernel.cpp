#include <vx_spawn.h>
#include <vx_tensor.h>
#include <cmath>
#include <cstring>
#include "common.h"

namespace vt = vortex::tensor;
using tcu_ctx = vt::wmma_context<NUM_TCU_LANES, vt::fp16, vt::fp32>;
static constexpr uint32_t TM = tcu_ctx::tileM;  // 8
static constexpr uint32_t TN = tcu_ctx::tileN;  // 8
static constexpr uint32_t TK = tcu_ctx::tileK;  // 16 (fp16 elements)

static inline uint16_t f2h(float x) {
    __fp16 h = (__fp16)x;
    uint16_t out;
    memcpy(&out, &h, sizeof(out));
    return out;
}

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto a_ptr = reinterpret_cast<float*>(arg->a_addr);
    auto b_ptr = reinterpret_cast<float*>(arg->b_addr);
    auto c_ptr = reinterpret_cast<float*>(arg->c_addr);

    uint32_t N = arg->N;
    uint32_t K = arg->K;
    uint32_t tile_row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t row_start = tile_row * TM;

    auto local_ptr = __local_mem(
        2 * TK * TM * sizeof(uint16_t) +
        TM * TN * sizeof(float)
    );
    uint16_t* local_A = (uint16_t*)local_ptr;
    uint16_t* local_B = local_A + TK * TM;
    float*    local_C = (float*)(local_B + TK * TN);

    for (uint32_t cblk = 0; cblk < N; cblk += TN) {
        tcu_ctx::fragment_acc fragC;
        tcu_ctx::fill_fragment(fragC, 0);

        for (uint32_t kb = 0; kb < K; kb += TK) {
            // Zero local tiles
            uint32_t a_elems = TM * TK;
            uint32_t b_elems = TN * TK;
            for (uint32_t i = tid; i < a_elems; i += blockDim.x)
                local_A[i] = 0;
            for (uint32_t i = tid; i < b_elems; i += blockDim.x)
                local_B[i] = 0;
            __syncthreads();

            // Load A tile [TM x TK] - only valid K range
            uint32_t k_valid = (kb + TK <= K) ? TK : (K - kb);
            for (uint32_t r = 0; r < TM; r++)
                for (uint32_t c = tid; c < k_valid; c += blockDim.x)
                    local_A[r * TK + c] = f2h(a_ptr[(row_start + r) * K + kb + c]);

            // Load B tile [TK x TN] stored as col-major [TN x TK]
            for (uint32_t r = 0; r < k_valid; r++)
                for (uint32_t c = tid; c < TN; c += blockDim.x)
                    local_B[c * TK + r] = f2h(b_ptr[(kb + r) * N + cblk + c]);

            __syncthreads();

            tcu_ctx::fragment_a fragA;
            tcu_ctx::fragment_b fragB;
            tcu_ctx::load_matrix_sync(fragA, local_A, TK);
            tcu_ctx::load_matrix_sync<vt::col_major>(fragB, local_B, TK);
            tcu_ctx::mma_sync(fragC, fragA, fragB, fragC);

            __syncthreads();
        }

        tcu_ctx::store_matrix_sync(local_C, fragC, TN);
        __syncthreads();

        if (tid < TM) {
            for (uint32_t c = 0; c < TN; c++)
                c_ptr[(row_start + tid) * N + cblk + c] = local_C[tid * TN + c];
        }
        __syncthreads();
    }
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
