#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// SGEMM: C = A @ B
// A is [M x K], B is [K x N], C is [M x N]
// One task per row of C (i.e., one task per row of A).

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto a_ptr = reinterpret_cast<float*>(arg->a_addr);
    auto b_ptr = reinterpret_cast<float*>(arg->b_addr);
    auto c_ptr = reinterpret_cast<float*>(arg->c_addr);

    uint32_t row = blockIdx.x;
    uint32_t N = arg->N;
    uint32_t K = arg->K;

    for (uint32_t j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < K; ++k) {
            sum += a_ptr[row * K + k] * b_ptr[k * N + j];
        }
        c_ptr[row * N + j] = sum;
    }
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->M, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
