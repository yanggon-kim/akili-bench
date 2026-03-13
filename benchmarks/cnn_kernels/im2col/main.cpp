#include <vortex.h>
#include <vector>
#include <iostream>
#include <cmath>
#include "common.h"

#define CHECK(x) if (!(x)) { std::cerr << "FAIL\n"; exit(1); }

int main() {

    vx_device_h dev;
    vx_dev_open(&dev);

    const int C = 1, H = 28, W = 28, K = 3;
    const int pad = 0, stride = 1;
    const int H_out = 26, W_out = 26;
    const int rows = H_out * W_out;
    const int cols = C * K * K;
    std::cout << "output dimensions:" << rows << " x " << cols << std::endl;

    std::vector<float> I(C*H*W);
    std::vector<float> A(rows * cols);
    std::vector<float> A_ref(rows * cols);

    for (auto& x : I) x = drand48();

    // CPU reference
    for (int oy = 0; oy < H_out; oy++) {
        for (int ox = 0; ox < W_out; ox++) {
            int r = oy * W_out + ox;
            int c = 0;
            for (int ic = 0; ic < C; ic++)
                for (int ky = 0; ky < K; ky++)
                    for (int kx = 0; kx < K; kx++) {
                        int iy = oy + ky;
                        int ix = ox + kx;
                        A_ref[r*cols + c++] = I[(ic*H + iy)*W + ix];
                    }
        }
    }

    im2col_arg_t arg{};
    arg.C_in = C;
    arg.H = H;
    arg.W = W;
    arg.padding = pad;
    arg.stride = stride;
    arg.H_out = H_out;
    arg.W_out = W_out;

    arg.grid_dim[0] = W_out;
    arg.grid_dim[1] = H_out;
    arg.grid_dim[2] = 1;

    arg.block_dim[0] = 1;
    arg.block_dim[1] = 1;
    arg.block_dim[2] = 1;

    vx_buffer_h I_buf, A_buf, arg_buf, bin_buf;

    vx_mem_alloc(dev, I.size()*4, VX_MEM_READ, &I_buf);
    vx_mem_alloc(dev, A.size()*4, VX_MEM_WRITE, &A_buf);

    vx_mem_address(I_buf, &arg.I_addr);
    vx_mem_address(A_buf, &arg.O_addr);

    vx_copy_to_dev(I_buf, I.data(), 0, I.size()*4);
    vx_upload_bytes(dev, &arg, sizeof(arg), &arg_buf);
    vx_upload_kernel_file(dev, "kernel.vxbin", &bin_buf);

    vx_start(dev, bin_buf, arg_buf);
    vx_ready_wait(dev, VX_MAX_TIMEOUT);
    vx_copy_from_dev(A.data(), A_buf, 0, A.size()*4);

    for (int i = 0; i < rows*cols; i++)
        CHECK(std::fabs(A[i] - A_ref[i]) < 1e-4);

    if(dev){
        // I_buf, A_buf, arg_buf, bin_buf;
        vx_mem_free(A_buf);
        vx_mem_free(I_buf);
        vx_mem_free(arg_buf);
        vx_mem_free(bin_buf);
        vx_dev_close(dev);
    }

    std::cout << "im2col PASSED\n";
}
