#include "../cnn_pipeline.h"
#include "../common.h"
#include <iostream>
#include <vortex.h>

Tensor conv2d_gpu(VortexDevice& vx,
                  const Tensor& input,
                  const std::vector<float>& W,
                  const std::vector<float>& B,
                  int C_out,
                  int K,
                  int padding,
                  int stride,
                  const char* kernel_file)
{
    int C_in = input.C;
    int H    = input.H;
    int Wt   = input.W;

    int H_out = (H + 2*padding - K)/stride + 1;
    int W_out = (Wt + 2*padding - K)/stride + 1;

    // Allocate output tensor
    Tensor output(C_out, H_out, W_out);

    // --- Upload input + weights + biases + allocate output
    vx_buffer_h I_buf, W_buf, B_buf, O_buf;

    uint64_t I_addr = upload_tensor_to_device(vx, input, &I_buf);
    uint64_t W_addr = upload_weights_to_device(vx, W, &W_buf);
    uint64_t B_addr = upload_weights_to_device(vx, B, &B_buf);

    size_t o_nbytes = output.data.size() * sizeof(float);
    vx.check(vx_mem_alloc(vx.dev, o_nbytes, VX_MEM_WRITE, &O_buf),
             "alloc O_buf");
    uint64_t O_addr;
    vx.check(vx_mem_address(O_buf, &O_addr), "O_addr");

    // --- Fill kernel_arg_t exactly like regression test ---
    kernel_arg_t arg{};
    arg.I_addr   = I_addr;
    arg.W_addr   = W_addr;
    arg.B_addr   = B_addr;
    arg.O_addr   = O_addr;

    arg.C_in     = C_in;
    arg.C_out    = C_out;
    arg.height   = H;
    arg.width    = Wt;
    arg.padding  = padding;
    arg.stride   = stride;
    arg.H_out    = H_out;
    arg.W_out    = W_out;

    arg.grid_dim[0] = W_out;
    arg.grid_dim[1] = H_out;
    arg.grid_dim[2] = C_out;  // z is not used by kernel, but this matches regression
    arg.block_dim[0] = 1;
    arg.block_dim[1] = 1;
    arg.block_dim[2] = 1;

    arg.use_lmem = 0; // use global memory like in regression host

    // Upload kernel arguments
    vx_buffer_h arg_buf;
    vx.check(vx_upload_bytes(vx.dev, &arg, sizeof(arg), &arg_buf),
             "upload conv args");

    // Upload kernel binary
    vx_buffer_h bin_buf;
    vx.check(vx_upload_kernel_file(vx.dev, kernel_file, &bin_buf),
             "upload conv vxbin");

    // Launch kernel
    vx.check(vx_start(vx.dev, bin_buf, arg_buf), "start conv kernel");

    // Wait for GPU
    vx.check(vx_ready_wait(vx.dev, VX_MAX_TIMEOUT), "conv wait");

    // Download output
    download_tensor_from_device(vx, output, O_buf);

    // Free temp buffers
    vx_mem_free(I_buf);
    vx_mem_free(W_buf);
    vx_mem_free(B_buf);
    vx_mem_free(O_buf);
    vx_mem_free(bin_buf);
    vx_mem_free(arg_buf);

    return output;
}
