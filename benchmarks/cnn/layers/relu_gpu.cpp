#include "../cnn_pipeline.h"
#include "../common.h"
#include <iostream>
#include <vortex.h>

// ------------------------------------------------------------
// GPU ReLU Wrapper (in-place on device buffer)
// ------------------------------------------------------------
//
// input:  Tensor(C, H, W)
// output: Tensor(C, H, W)
// kernel_file: e.g. "layers/relu_kernel.vxbin"
//
// This matches regression/cnn_relu/common.h + kernel:
//   - kernel_arg_relu_t { X_addr, total, grid_dim, block_dim }
//   - kernel_body uses X[idx] in-place
// ------------------------------------------------------------

Tensor relu_gpu(VortexDevice& vx,
                const Tensor& input,
                const char* kernel_file)
{
    int C = input.C;
    int H = input.H;
    int W = input.W;
    int total = C * H * W;

    Tensor output(C, H, W);

    // --- Allocate RW buffer and upload input ---
    vx_buffer_h X_buf;
    size_t nbytes = input.data.size() * sizeof(float);

    vx.check(
        vx_mem_alloc(vx.dev, nbytes, VX_MEM_READ | VX_MEM_WRITE, &X_buf),
        "relu alloc X_buf"
    );

    uint64_t X_addr;
    vx.check(vx_mem_address(X_buf, &X_addr), "relu X_addr");

    vx.check(
        vx_copy_to_dev(X_buf, input.data.data(), 0, nbytes),
        "relu copy input to dev"
    );

    // --- Fill kernel_arg_relu_t ---
    kernel_arg_relu_t arg{};
    arg.X_addr       = X_addr;
    arg.total        = total;
    arg.grid_dim[0]  = total;  // 1D grid: one thread per element
    arg.grid_dim[1]  = 1;
    arg.grid_dim[2]  = 1;
    arg.block_dim[0] = 1;
    arg.block_dim[1] = 1;
    arg.block_dim[2] = 1;

    // --- Upload args ---
    vx_buffer_h arg_buf;
    vx.check(
        vx_upload_bytes(vx.dev, &arg, sizeof(arg), &arg_buf),
        "upload relu args"
    );

    // --- Upload kernel binary ---
    vx_buffer_h bin_buf;
    vx.check(
        vx_upload_kernel_file(vx.dev, kernel_file, &bin_buf),
        "upload relu kernel"
    );

    // --- Launch kernel ---
    vx.check(
        vx_start(vx.dev, bin_buf, arg_buf),
        "start relu kernel"
    );

    // --- Wait for device ---
    vx.check(
        vx_ready_wait(vx.dev, VX_MAX_TIMEOUT),
        "relu wait"
    );

    // --- Download result into output tensor ---
    vx.check(
        vx_copy_from_dev(output.data.data(), X_buf, 0, nbytes),
        "relu download output"
    );

    // --- Cleanup ---
    vx_mem_free(X_buf);
    vx_mem_free(arg_buf);
    vx_mem_free(bin_buf);

    return output;
}
