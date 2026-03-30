#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

// SwiGLU Forward: c = silu(a) * b, where silu(x) = x * sigmoid(x)
// Matches Liger/Unsloth Triton _swiglu_forward_kernel.
// One task per row, each thread processes num_cols/threads_per_warp elements.

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto a_ptr = reinterpret_cast<float*>(arg->a_addr);
    auto b_ptr = reinterpret_cast<float*>(arg->b_addr);
    auto c_ptr = reinterpret_cast<float*>(arg->c_addr);

    uint32_t row = blockIdx.x;
    uint32_t num_cols = arg->num_cols;
    uint32_t offset = row * num_cols;

    for (uint32_t col = 0; col < num_cols; ++col) {
        float av = a_ptr[offset + col];
        float bv = b_ptr[offset + col];
        float sig = 1.0f / (1.0f + expf(-av));
        float silu = av * sig;
        c_ptr[offset + col] = silu * bv;
    }
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->num_rows, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
