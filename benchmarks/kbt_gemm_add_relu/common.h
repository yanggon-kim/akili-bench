#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t grid_dim[3];
  uint32_t block_dim[3];
  uint32_t M;       // batch (rows)
  uint32_t N;       // out_features
  uint32_t K;       // in_features
  uint64_t a_addr;    // input  [M x K]
  uint64_t w_addr;    // weight [K x N]
  uint64_t bias_addr; // bias   [N]
  uint64_t c_addr;    // output [M x N]
} kernel_arg_t;

#endif
