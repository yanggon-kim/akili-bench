#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Dense TCU conv2d via im2col + GEMM.
//   W_gemm [M_gemm × K_gemm] fp16 row-major   (A)
//   Icol   [N_gemm × K_gemm] fp16 col-major   (B)  — host builds via im2col
//   Out    [M_gemm × N_gemm] fp32 row-major   (C)
//   M_gemm = ceil(C_out,          tileM)
//   N_gemm = ceil(H_out * W_out,  tileN)
//   K_gemm = ceil(C_in * K * K,   tileK)
typedef struct {
  uint64_t W_addr;        // fp16 W_gemm
  uint64_t B_addr;        // fp16 Icol (col-major)
  uint64_t O_addr;        // fp32 Out_gemm
  uint64_t kernel_cycles;
  uint32_t grid_dim[2];   // {N_gemm/tileN, M_gemm/tileM}
  uint32_t block_dim[2];
  uint32_t C_in, C_out;
  uint32_t H, W;
  uint32_t K_sz;
  uint32_t H_out, W_out;
  uint32_t M_gemm, N_gemm, K_gemm;
  uint32_t _pad;
} kernel_arg_t;

#endif
