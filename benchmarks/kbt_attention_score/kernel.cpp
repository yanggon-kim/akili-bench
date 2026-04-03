#include <vx_spawn.h>
#include <vx_tensor.h>
#include <cmath>
#include <cstring>
#include "common.h"

// TCU attention score: C = softmax(Q @ K^T / sqrt(D))
// Q [M x D], K [N x D], C [M x N]
// Phase 1: TCU GEMM for Q * K^T, scale by 1/sqrt(D), track row max
// Phase 2: SIMT row-wise softmax

namespace vt = vortex::tensor;
using tcu_ctx = vt::wmma_context<NUM_TCU_LANES, vt::fp16, vt::fp32>;
static constexpr uint32_t TM = tcu_ctx::tileM;
static constexpr uint32_t TN = tcu_ctx::tileN;
static constexpr uint32_t TK = tcu_ctx::tileK;

static inline uint16_t f2h(float x) {
    __fp16 h = (__fp16)x;
    uint16_t out;
    memcpy(&out, &h, sizeof(out));
    return out;
}

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto q_ptr = reinterpret_cast<float*>(arg->q_addr);
    auto k_ptr = reinterpret_cast<float*>(arg->k_addr);
    auto c_ptr = reinterpret_cast<float*>(arg->c_addr);

    uint32_t N = arg->N;
    uint32_t D = arg->D;
    uint32_t tile_row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t row_start = tile_row * TM;

    float inv_sqrt_d = 1.0f / sqrtf((float)D);

    auto local_ptr = __local_mem(
        2 * TK * TM * sizeof(uint16_t) +
        TM * TN * sizeof(float)
    );
    uint16_t* local_A = (uint16_t*)local_ptr;
    uint16_t* local_B = local_A + TM * TK;
    float*    local_C = (float*)(local_B + TN * TK);

    float row_max = -INFINITY;

    // Phase 1: TCU GEMM for Q * K^T, scaled by 1/sqrt(D)
    // Output column blocks correspond to rows of K
    for (uint32_t cblk = 0; cblk < N; cblk += TN) {
        tcu_ctx::fragment_acc fragC;
        tcu_ctx::fill_fragment(fragC, 0);

        // Accumulate over D dimension
        for (uint32_t db = 0; db < D; db += TK) {
            uint32_t a_elems = TM * TK;
            uint32_t b_elems = TN * TK;
            for (uint32_t i = tid; i < a_elems; i += blockDim.x)
                local_A[i] = 0;
            for (uint32_t i = tid; i < b_elems; i += blockDim.x)
                local_B[i] = 0;
            __syncthreads();

            // Load Q tile [TM x k_valid]: rows [row_start..+TM], cols [db..db+k_valid]
            uint32_t k_valid = (db + TK <= D) ? TK : (D - db);
            for (uint32_t r = 0; r < TM; r++)
                for (uint32_t c = tid; c < k_valid; c += blockDim.x)
                    local_A[r * TK + c] = f2h(q_ptr[(row_start + r) * D + db + c]);

            // Load K tile [TN x k_valid]: rows [cblk..+TN], cols [db..db+k_valid]
            // Store as col-major [TN x TK] for fragB = K_tile^T
            for (uint32_t r = 0; r < k_valid; r++)
                for (uint32_t c = tid; c < TN; c += blockDim.x)
                    local_B[c * TK + r] = f2h(k_ptr[(cblk + c) * D + db + r]);

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
            for (uint32_t c = 0; c < TN; c++) {
                float val = local_C[tid * TN + c] * inv_sqrt_d;
                c_ptr[(row_start + tid) * N + cblk + c] = val;
                if (val > row_max) row_max = val;
            }
        }
        __syncthreads();
    }

    // Phase 2: SIMT row-wise softmax
    if (tid < TM) {
        uint32_t row = row_start + tid;
        float sum = 0;
        for (uint32_t j = 0; j < N; j++) {
            float v = expf(c_ptr[row * N + j] - row_max);
            c_ptr[row * N + j] = v;
            sum += v;
        }
        float inv = 1.0f / sum;
        for (uint32_t j = 0; j < N; j++)
            c_ptr[row * N + j] *= inv;
    }
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
