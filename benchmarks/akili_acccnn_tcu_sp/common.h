#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Sparse TCU conv2d: 2:4 on weight matrix W_gemm.
//   W_addr    = compressed fp16 W  [M_gemm × K_gemm/2]
//   meta_W    = packed 2:4 metadata for W
//   B_addr    = fp16 Icol (col-major) [N_gemm × K_gemm]
//   O_addr    = fp32 out [M_gemm × N_gemm]
typedef struct {
  uint64_t W_addr;
  uint64_t B_addr;
  uint64_t O_addr;
  uint64_t meta_W_addr;
  uint64_t kernel_cycles;
  uint32_t grid_dim[2];
  uint32_t block_dim[2];
  uint32_t C_in, C_out;
  uint32_t H, W;
  uint32_t K_sz;
  uint32_t H_out, W_out;
  uint32_t M_gemm, N_gemm, K_gemm;
  uint32_t _pad;
} kernel_arg_t;

#endif
