#include "../cnn_pipeline.h"
#include "../common.h"
#include <iostream>
#include <vortex.h>

// ------------------------------------------------------------
// GPU MaxPool2D Wrapper
// ------------------------------------------------------------
//
// input:  Tensor(C, H, W)
// pool_size: must be 2 (your kernel is fixed 2×2, stride=2)
// kernel_file: pool kernel vxbin (e.g., "pool_kernel.vxbin")
//
// output:
//   Tensor(C, H/2, W/2)
//
// ------------------------------------------------------------

Tensor maxpool2d_gpu(VortexDevice& vx,
                     const Tensor& input,
                     int pool_size,
                     const char* kernel_file)
{
    if (pool_size != 2) {
        std::cerr << "ERROR: This pool GPU kernel only supports pool_size=2\n";
        exit(1);
    }

    int C = input.C;
    int H = input.H;
    int W = input.W;

    int H2 = H / 2;
    int W2 = W / 2;

    // Output tensor
    Tensor output(C, H2, W2);

    //
    // --- Upload input, allocate output ---
    //
    vx_buffer_h I_buf, O_buf;

    uint64_t I_addr = upload_tensor_to_device(vx, input, &I_buf);

    size_t o_nbytes = output.data.size() * sizeof(float);
    vx.check(vx_mem_alloc(vx.dev, o_nbytes, VX_MEM_WRITE, &O_buf),
             "alloc O_buf");
    uint64_t O_addr;
    vx.check(vx_mem_address(O_buf, &O_addr), "O_addr");

    //
    // --- Fill kernel_arg_pool_t exactly like your regression test ---
    //
    kernel_arg_pool_t arg{};
    arg.I_addr = I_addr;
    arg.O_addr = O_addr;
    arg.C      = C;
    arg.H      = H;
    arg.W      = W;

    // Grid: {W_out, H_out, ???}
    // Your kernel only uses blockIdx.x, blockIdx.y, AND loops oc internally.
    arg.grid_dim[0] = W2;
    arg.grid_dim[1] = H2;
    arg.grid_dim[2] = 1;

    // Block dims = 1
    arg.block_dim[0] = 1;
    arg.block_dim[1] = 1;
    arg.block_dim[2] = 1;

    //
    // Upload kernel argument
    //
    vx_buffer_h arg_buf;
    vx.check(vx_upload_bytes(vx.dev, &arg, sizeof(arg), &arg_buf),
             "upload pool args");

    //
    // Upload kernel binary
    //
    vx_buffer_h bin_buf;
    vx.check(vx_upload_kernel_file(vx.dev, kernel_file, &bin_buf),
             "upload pool kernel");

    //
    // Launch kernel
    //
    vx.check(vx_start(vx.dev, bin_buf, arg_buf), "start pool kernel");

    //
    // Wait for device
    //
    vx.check(vx_ready_wait(vx.dev, VX_MAX_TIMEOUT), "pool wait");

    //
    // Download output
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
