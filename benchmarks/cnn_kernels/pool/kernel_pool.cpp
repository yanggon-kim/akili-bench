#include <vx_spawn.h>
#include "common.h"

void kernel_body(kernel_arg_pool_t* __UNIFORM__ arg) {
    TYPE* I = reinterpret_cast<TYPE*>(arg->I_addr);
    TYPE* O = reinterpret_cast<TYPE*>(arg->O_addr);

    const int C = arg->C;
    const int H = arg->H;
    const int W = arg->W;

    const int H_out = H / 2;
    const int W_out = W / 2;

    int ox = blockIdx.x;
    int oy = blockIdx.y;

    if (ox >= W_out || oy >= H_out)
        return;

    for (int oc = 0; oc < C; ++oc) {
        TYPE m = static_cast<TYPE>(-1e30f);

        // 2×2 window with stride 2
        for (int ky = 0; ky < 2; ++ky) {
            for (int kx = 0; kx < 2; ++kx) {
                int iy = oy * 2 + ky;
                int ix = ox * 2 + kx;

                TYPE v = I[oc * H * W + iy * W + ix];
                if (v > m) {
                    m = v;
                }
            }
        }

        O[oc * H_out * W_out + oy * W_out + ox] = m;
    }
}

int main() {
    auto arg = (kernel_arg_pool_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(
        2,            // dimension = 2D launch
        arg->grid_dim,
        arg->block_dim,
        (vx_kernel_func_cb)kernel_body,
        arg
    );
}
