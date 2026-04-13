#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// SIMT conv2d — direct stride-1, valid-padding convolution. No GEMM dims;
// the kernel walks C_in × K × K inside each output (oc, oy, ox) thread.
typedef struct {
  uint64_t I_addr;        // fp32 input   [C_in × H × W]
  uint64_t W_addr;        // fp32 weights [C_out × C_in × K × K]
  uint64_t O_addr;        // fp32 output  [C_out × H_out × W_out]
  uint64_t kernel_cycles; // written by device main()
  uint32_t grid_dim[2];   // {H_out*W_out, C_out}
  uint32_t block_dim[2];
  uint32_t C_in, C_out;
  uint32_t H, W;
  uint32_t K_sz;
  uint32_t H_out, W_out;
  uint32_t _pad;
} kernel_arg_t;

#endif
