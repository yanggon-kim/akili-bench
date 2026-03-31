#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// Matrix-vector multiply: y = A @ x
// A is [M x K], x is [K], y is [M]
// One task per row of A.

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto a_ptr = reinterpret_cast<float*>(arg->a_addr);
    auto x_ptr = reinterpret_cast<float*>(arg->x_addr);
    auto y_ptr = reinterpret_cast<float*>(arg->y_addr);

    uint32_t row = blockIdx.x;
    uint32_t K = arg->K;

    float sum = 0.0f;
    for (uint32_t k = 0; k < K; ++k) {
        sum += a_ptr[row * K + k] * x_ptr[k];
    }
    y_ptr[row] = sum;
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->M, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
