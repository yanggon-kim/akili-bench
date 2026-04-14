#include <vx_spawn2.h>
#include "common.h"

// =============================================================================
// akili_llama2 matmul — plain SIMT GEMM. One block covers a BLOCK_SIZE × BLOCK_SIZE
// tile of C; threads within the block stripe across rows and columns via
// blockIdx/blockDim/threadIdx as provided by the KMU. KMU-dispatched via vx_start_g.
// =============================================================================
__kernel void kernel_main(matmul_kernel_args_t* __UNIFORM__ args) {
    auto A = reinterpret_cast<float*>(args->A_addr);
    auto B = reinterpret_cast<float*>(args->B_addr);
    auto C = reinterpret_cast<float*>(args->C_addr);

    int M = args->M;
    int N = args->N;
    int K = args->K;

    // 2D indexing - each thread computes one output element
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
