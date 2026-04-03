#include <vx_spawn.h>
#include <vx_tensor.h>
#include <cmath>
#include <cstring>
#include "common.h"

// silu(RMSNorm(x) @ Wg) * (RMSNorm(x) @ Wu)
// Phase 1: RMSNorm x -> norm_ptr (SIMT)
// Phase 2: gate = norm @ Wg (TCU GEMM)
// Phase 3: up   = norm @ Wu (TCU GEMM)
// Phase 4: c = silu(gate) * up (SIMT)

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

// Helper: run tiled TCU GEMM from src_a[M x K] and src_b[K x N], write to dst[M x N]
// Only writes dst for rows [row_start..row_start+TM)
static void tcu_gemm_tile(
    float* src_a, float* src_b, float* dst,
    uint32_t row_start, uint32_t N, uint32_t K,
    uint32_t tid,
    uint16_t* local_A, uint16_t* local_B, float* local_C)
{
    for (uint32_t cblk = 0; cblk < N; cblk += TN) {
        tcu_ctx::fragment_acc fragC;
        tcu_ctx::fill_fragment(fragC, 0);
        for (uint32_t kb = 0; kb < K; kb += TK) {
            uint32_t a_elems = TM * TK;
            uint32_t b_elems = TN * TK;
            for (uint32_t i = tid; i < a_elems; i += blockDim.x)
                local_A[i] = 0;
            for (uint32_t i = tid; i < b_elems; i += blockDim.x)
                local_B[i] = 0;
            __syncthreads();
            uint32_t k_valid = (kb + TK <= K) ? TK : (K - kb);
            for (uint32_t r = 0; r < TM; r++)
                for (uint32_t c = tid; c < k_valid; c += blockDim.x)
                    local_A[r * TK + c] = f2h(src_a[(row_start + r) * K + kb + c]);
            for (uint32_t r = 0; r < k_valid; r++)
                for (uint32_t c = tid; c < TN; c += blockDim.x)
                    local_B[c * TK + r] = f2h(src_b[(kb + r) * N + cblk + c]);
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
                dst[(row_start + tid) * N + cblk + c] = local_C[tid * TN + c];
        }
        __syncthreads();
    }
}

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto x_ptr     = reinterpret_cast<float*>(arg->x_addr);
    auto rms_w_ptr = reinterpret_cast<float*>(arg->rms_w_addr);
    auto wg_ptr    = reinterpret_cast<float*>(arg->wg_addr);
    auto wu_ptr    = reinterpret_cast<float*>(arg->wu_addr);
    auto norm_ptr  = reinterpret_cast<float*>(arg->norm_addr);
    auto gate_ptr  = reinterpret_cast<float*>(arg->gate_addr);
    auto c_ptr     = reinterpret_cast<float*>(arg->c_addr);

    uint32_t N   = arg->N;
    uint32_t K   = arg->K;
    float    eps = arg->eps;
    uint32_t tile_row  = blockIdx.x;
    uint32_t tid       = threadIdx.x;
    uint32_t row_start = tile_row * TM;

    auto local_ptr = __local_mem(
        2 * TK * TM * sizeof(uint16_t) +
        TM * TN * sizeof(float)
    );
    uint16_t* local_A = (uint16_t*)local_ptr;
    uint16_t* local_B = local_A + TM * TK;
    float*    local_C = (float*)(local_B + TN * TK);

    // Phase 1: RMSNorm (SIMT) — normalize x into norm_ptr
    if (tid < TM) {
        uint32_t row = row_start + tid;
        float ss = 0;
        for (uint32_t k = 0; k < K; k++) {
            float v = x_ptr[row * K + k];
            ss += v * v;
        }
        float inv_rms = 1.0f / sqrtf(ss / (float)K + eps);
        for (uint32_t k = 0; k < K; k++)
            norm_ptr[row * K + k] = x_ptr[row * K + k] * inv_rms * rms_w_ptr[k];
    }
    __syncthreads();

    // Phase 2: gate = norm @ Wg (TCU GEMM)
    tcu_gemm_tile(norm_ptr, wg_ptr, gate_ptr, row_start, N, K, tid, local_A, local_B, local_C);

    // Phase 3: up = norm @ Wu (TCU GEMM)
    tcu_gemm_tile(norm_ptr, wu_ptr, c_ptr, row_start, N, K, tid, local_A, local_B, local_C);

    // Phase 4: c = silu(gate) * up (SIMT)
    if (tid < TM) {
        uint32_t row = row_start + tid;
        for (uint32_t j = 0; j < N; j++) {
            float g = gate_ptr[row * N + j];
            float u = c_ptr[row * N + j];
            float silu_g = g / (1.0f + expf(-g));
            c_ptr[row * N + j] = silu_g * u;
        }
    }
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
