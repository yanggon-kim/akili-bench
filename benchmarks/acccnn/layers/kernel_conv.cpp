#include <vx_spawn.h>
#include "../common.h" 
#include <cstdio> 

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto I = reinterpret_cast<TYPE*>(arg->I_addr);
    auto W = reinterpret_cast<TYPE*>(arg->use_lmem ? __local_mem(0) : (void*)arg->W_addr);
    auto O = reinterpret_cast<TYPE*>(arg->O_addr);

    const int C_in    = arg->C_in;
    const int C_out   = arg->C_out;
    const int H       = arg->height;
    const int Wt      = arg->width;
    const int padding = arg->padding;
    const int stride  = arg->stride;

    const int K = 3;

    const int H_out = arg->H_out;
    const int W_out = arg->W_out;

    // Thread's output coordinates:
    int ox = blockIdx.x;  // output x (width)
    int oy = blockIdx.y;  // output y (height)

    if (ox >= W_out || oy >= H_out)
        return;

    float sum = 0.0f;

    for (int oc = 0; oc < C_out; ++oc) {
        float sum = 0.0f;
        // Loop over input channels and 3x3 kernel
        for (int ic = 0; ic < C_in; ++ic) {
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    int in_y = oy * stride + ky - padding;
                    int in_x = ox * stride + kx - padding;

                    // Zero-padding by skipping out-of-bounds
                    if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < Wt) {
                        int in_idx = ic * H * Wt + in_y * Wt + in_x;
                        int wt_idx = ((oc * C_in + ic) * K + ky) * K + kx;

                        sum += I[in_idx] * W[wt_idx];
                    }
                }
            }
        }
        // Store result: [C_out][H_out][W_out]
        int out_idx = oc * H_out * W_out + oy * W_out + ox;
        O[out_idx] = static_cast<TYPE>(sum);
    }
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    // printf("DEVICE sizeof(kernel_arg_t) = %d\n", sizeof(kernel_arg_t));
    // printf("Test printing inside kernel.cpp\n");
    if (arg->use_lmem) {
        // Copy all weights to local memory
        auto W_global = reinterpret_cast<TYPE*>(arg->W_addr);
        auto W_local  = reinterpret_cast<TYPE*>(__local_mem(0));

        int total_w = arg->C_out * arg->C_in * 3 * 3;
        for (int i = 0; i < total_w; ++i) {
            W_local[i] = W_global[i];
        }
    }

    // Launch threads: grid_dim was filled by host
    return vx_spawn_threads(2, arg->grid_dim, arg->block_dim,
                            (vx_kernel_func_cb)kernel_body, arg);
}
