#include "../cnn_pipeline.h"
#include "../common.h"
#include <iostream>
#include <vortex.h>

Tensor conv2d_gpu(VortexDevice& vx,
                  const Tensor& input,
                  const std::vector<float>& W,   // original TF layout
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

    Tensor output(C_out, H_out, W_out);

    //------------------------------------------------------------
    // 1. Reorder weights → GEMM layout
    //
    // Original TF/Keras layout: W[ky][kx][ic][oc]
    // Needed layout: W_gemm[oc][ic*K*K + ky*K + kx]
    //------------------------------------------------------------

    const int KK = C_in * K * K;
    std::vector<float> W_gemm(C_out * KK);

    for (int oc = 0; oc < C_out; ++oc) {
        for (int ic = 0; ic < C_in; ++ic) {
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {

                    // TF index: (ky, kx, ic, oc)
                    int tf_idx = ((ky * K + kx) * C_in + ic) * C_out + oc;

                    // GEMM index
                    int gemm_idx = oc * KK + (ic * K * K + ky * K + kx);

                    W_gemm[gemm_idx] = W[tf_idx];
                }
            }
        }
    }

    //------------------------------------------------------------
    // 2. Upload input, weights, biases, allocate output
    //------------------------------------------------------------

    vx_buffer_h I_buf, W_buf, O_buf, B_buf;

    uint64_t I_addr = upload_tensor_to_device(vx, input, &I_buf);
    uint64_t W_addr = upload_weights_to_device(vx, W_gemm, &W_buf);
    uint64_t B_addr = upload_weights_to_device(vx, B, &B_buf);

    size_t o_nbytes = output.data.size() * sizeof(float);
    vx.check(vx_mem_alloc(vx.dev, o_nbytes, VX_MEM_WRITE, &O_buf), "alloc O_buf");

    uint64_t O_addr;
    vx.check(vx_mem_address(O_buf, &O_addr), "O_addr");

    //------------------------------------------------------------
    // 3. Fill kernel_arg_t
    //------------------------------------------------------------

    kernel_arg_t arg{};
    arg.C_in    = C_in;
    arg.C_out   = C_out;
    arg.height  = H;
    arg.width   = Wt;
    arg.padding = padding;
    arg.stride  = stride;
    arg.H_out   = H_out;
    arg.W_out   = W_out;

    arg.I_addr  = I_addr;
    arg.W_addr  = W_addr;
    arg.O_addr  = O_addr;
    arg.B_addr  = B_addr;

    arg.use_lmem = 0;
    arg.use_tcu  = 0;   // CPU FMA path for now

    // Vortex = 2D grid only
    arg.grid_dim[0] = W_out;   // x
    arg.grid_dim[1] = H_out;   // y
    arg.grid_dim[2] = 1;

    arg.block_dim[0] = 1;
    arg.block_dim[1] = 1;
    arg.block_dim[2] = 1;

    //------------------------------------------------------------
    // 4. Upload args + kernel binary
    //------------------------------------------------------------

    vx_buffer_h arg_buf;
    vx.check(vx_upload_bytes(vx.dev, &arg, sizeof(arg), &arg_buf), "upload args");

    vx_buffer_h bin_buf;
    vx.check(vx_upload_kernel_file(vx.dev, kernel_file, &bin_buf), "upload kernel");

    //------------------------------------------------------------
    // 5. Run kernel
    //------------------------------------------------------------

    vx.check(vx_start(vx.dev, bin_buf, arg_buf), "start conv");
    vx.check(vx_ready_wait(vx.dev, VX_MAX_TIMEOUT), "conv wait");

    //------------------------------------------------------------
    // 6. Download output
    //------------------------------------------------------------

    download_tensor_from_device(vx, output, O_buf);

    //------------------------------------------------------------
    // 7. Free buffers
    //------------------------------------------------------------

    vx_mem_free(I_buf);
    vx_mem_free(W_buf);
    vx_mem_free(B_buf);
    vx_mem_free(O_buf);
    vx_mem_free(bin_buf);
    vx_mem_free(arg_buf);

    return output;
}
