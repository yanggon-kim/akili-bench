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
  uint64_t cycles_addr;   // uint32_t[num_blocks] — per-CTA cycle counts
  uint32_t C_in, C_out;
  uint32_t H, W;
  uint32_t K_sz;
  uint32_t H_out, W_out;
  uint32_t M_gemm, N_gemm, K_gemm;
} kernel_arg_t;

#endif
