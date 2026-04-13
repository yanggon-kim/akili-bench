#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Execution modes
#define MODE_SIMT        0
#define MODE_DENSE_TCU   1
#define MODE_SPARSE_TCU  2

// Kernel dispatch IDs
#define KID_CONV_SIMT    0
#define KID_CONV_TCU     1
#define KID_CONV_SPARSE  2

// Convolution benchmark: C = conv2d(I, W)
//   I shape [C_in × H × W], row-major
//   W shape [C_out × C_in × K × K], row-major (TF layout already flattened
//     into [C_out × (C_in·K·K)] GEMM format on the host before upload)
//   O shape [C_out × H_out × W_out], row-major
//
// TCU path does the im2col on the host so the device sees a plain GEMM:
//   Out_gemm [M_gemm × N_gemm] = W_gemm [M_gemm × K_gemm]
//                              · Icol   [K_gemm × N_gemm]
// where
//   M_gemm = ceil(C_out,          tileM)
//   N_gemm = ceil(H_out * W_out,  tileN)
//   K_gemm = ceil(C_in * K * K,   tileK)
typedef struct {
  uint64_t I_addr;        // fp32 input [C_in × H × W]       (SIMT path)
  uint64_t W_addr;        // fp32 weights [M_gemm × K_gemm]   (SIMT path)
                          //  — or fp16 dense / compressed fp16 (TCU paths)
  uint64_t B_addr;        // TCU only: fp16 Icol [N_gemm × K_gemm] col-major
  uint64_t O_addr;        // fp32 output [M_gemm × N_gemm]
  uint64_t meta_W_addr;   // sparse only: packed metadata for W
  uint64_t kernel_cycles; // written by kernel main()

  uint32_t grid_dim[2];
  uint32_t block_dim[2];

  uint32_t C_in, C_out;
  uint32_t H, W;
  uint32_t K_sz;        // kernel spatial size (e.g. 3)
  uint32_t H_out, W_out;
  uint32_t M_gemm, N_gemm, K_gemm;
  uint32_t kernel_id;
  uint32_t _pad;
} kernel_arg_t;

#endif // _COMMON_H_
