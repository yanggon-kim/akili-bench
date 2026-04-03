#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t grid_dim[3];
  uint32_t block_dim[3];
  uint32_t M;       // batch (rows)
  uint32_t N;       // out_features
  uint32_t K;       // in_features
  float scale_factor;
  float clamp_min;
  float clamp_max;
  uint64_t a_addr;    // input  [M x K]
  uint64_t b_addr;    // weight [K x N]
  uint64_t c_addr;    // output [M] (one scalar per row)
  uint64_t temp_addr; // intermediate [M x N]
} kernel_arg_t;

#endif
