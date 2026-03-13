#include <vx_spawn.h>
#include "../common.h" 

void kernel_body(kernel_arg_relu_t* __UNIFORM__ arg) {
    auto X = reinterpret_cast<float*>(arg->X_addr);
    int total = arg->total;

    int idx = blockIdx.x;
    if (idx >= total) return;

    float v = X[idx];
    if (v < 0) v = 0;
    X[idx] = v;
}

int main() {
    auto arg = (kernel_arg_relu_t*)csr_read(VX_CSR_MSCRATCH);

    return vx_spawn_threads(
        1,
        arg->grid_dim,
        arg->block_dim,
        (vx_kernel_func_cb)kernel_body,
        arg
    );
}
