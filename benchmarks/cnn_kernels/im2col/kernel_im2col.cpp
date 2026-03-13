#include <vx_spawn.h>
#include "common.h"

void kernel_body(im2col_arg_t* __UNIFORM__ arg) {

    float* I = reinterpret_cast<float*>(arg->I_addr);
    float* A = reinterpret_cast<float*>(arg->O_addr); // output matrix

    int C_in   = arg->C_in;
    int H      = arg->H;
    int W      = arg->W;
    int K      = 3;
    int stride = arg->stride;
    int pad    = arg->padding;

    int H_out  = arg->H_out;
    int W_out  = arg->W_out;

    int ox = blockIdx.x;
    int oy = blockIdx.y;

    if (ox >= W_out || oy >= H_out)
        return;

    int row = oy * W_out + ox;
    int col = 0;

    for (int ic = 0; ic < C_in; ic++) {
        for (int ky = 0; ky < K; ky++) {
            for (int kx = 0; kx < K; kx++) {

                int iy = oy * stride + ky - pad;
                int ix = ox * stride + kx - pad;

                float v = 0.0f;
                if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                    int in_idx = (ic * H + iy) * W + ix;
                    v = I[in_idx];
                }

                A[row * (C_in*K*K) + col] = v;
                col++;
            }
        }
    }
}

int main() {
    auto arg = (im2col_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(
        2,
        arg->grid_dim,
        arg->block_dim,
        (vx_kernel_func_cb)kernel_body,
        arg
    );
}
