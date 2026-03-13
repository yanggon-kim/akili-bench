#include <vx_spawn.h>
#include "../common.h" 

void kernel_body(kernel_arg_pool_t* __UNIFORM__ arg) {
    auto I = reinterpret_cast<float*>(arg->I_addr);
    auto O = reinterpret_cast<float*>(arg->O_addr);

    int C = arg->C;
    int H = arg->H;
    int W = arg->W;

    int H_out = H/2;
    int W_out = W/2;

    int ox = blockIdx.x;
    int oy = blockIdx.y;

    if (ox>=W_out || oy>=H_out) return;
    for (int oc = 0; oc < 4; oc++) {
        float m = -1e30f;
        for(int ky=0; ky<2; ky++){
            for(int kx=0; kx<2; kx++){
                int iy = oy*2 + ky;
                int ix = ox*2 + kx;
                float v = I[oc*H*W + iy*W + ix];
                if(v > m) m = v;
            }
        }

        O[oc*H_out*W_out + oy*W_out + ox] = m;

    }
}

int main(){
    auto arg = (kernel_arg_pool_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(2, arg->grid_dim, nullptr,
                            (vx_kernel_func_cb)kernel_body, arg);
}
