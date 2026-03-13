#include "../cnn_pipeline.h"
#include "../common.h"
#include <iostream>
#include <vortex.h>

// ------------------------------------------------------------
// GPU ReLU Wrapper
// ------------------------------------------------------------
//
// input:  Tensor(C, H, W)
// output: Tensor(C, H, W)
//
// kernel_file: e.g. "relu_kernel.vxbin"
//
// Your device kernel should do: O[i] = max(0, I[i])
// Grid: 2D = (W, H), channel loop inside kernel
// ------------------------------------------------------------

Tensor relu_gpu(VortexDevice& vx,
                const Tensor& input,
                const char* kernel_file)
{
    int C = input.C;
    int H = input.H;
    int W = input.W;

    Tensor output(C, H, W);

    //
    // --- Upload input & allocate output ---
    //
    vx_buffer_h I_buf, O_buf;

    uint64_t I_addr = upload_tensor_to_device(vx, input, &I_buf);

    size_t o_nbytes = output.data.size() * sizeof(float);
    vx.check(vx_mem_alloc(vx.dev, o_nbytes, VX_MEM_WRITE, &O_buf),
             "relu alloc O_buf");
    uint64_t O_addr;
    vx.check(vx_mem_address(O_buf, &O_addr), "relu O_addr");

    //
    // --- Fill kernel_arg_relu_t ---
    //
    kernel_arg_relu_t arg{};
    arg.I_addr = I_addr;
    arg.O_addr = O_addr;
    arg.C      = C;
    arg.H      = H;
    arg.W      = W;

    // grid = {W, H, 1}
    arg.grid_dim[0] = W;
    arg.grid_dim[1] = H;
    arg.grid_dim[2] = 1;

    // block = {1,1,1}
    arg.block_dim[0] = 1;
    arg.block_dim[1] = 1;
    arg.block_dim[2] = 1;

    //
    // Upload args
    //
    vx_buffer_h arg_buf;
    vx.check(vx_upload_bytes(vx.dev, &arg, sizeof(arg), &arg_buf),
             "upload relu args");

    //
    // Upload kernel bin
    //
    vx_buffer_h bin_buf;
    vx.check(vx_upload_kernel_file(vx.dev, kernel_file, &bin_buf),
             "upload relu kernel");

    //
    // Launch
    //
    vx.check(vx_start(vx.dev, bin_buf, arg_buf),
             "start relu kernel");

    //
    // Wait for device
    //
    vx.check(vx_ready_wait(vx.dev, VX_MAX_TIMEOUT),
             "relu wait");

    //
    // Download result
    //
    download_tensor_from_device(vx, output, O_buf);

    //
    // Cleanup
    //
    vx_mem_free(I_buf);
    vx_mem_free(O_buf);
    vx_mem_free(arg_buf);
    vx_mem_free(bin_buf);

    return output;
}
