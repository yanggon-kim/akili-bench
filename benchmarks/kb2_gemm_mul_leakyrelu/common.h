#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t M;       // batch (rows of A / rows of C)
  uint32_t N;       // out_features (cols of C)
  uint32_t K;       // in_features (cols of A / rows of W)
  float multiplier;
  float negative_slope;
  uint64_t a_addr;  // input  [M x K]
  uint64_t w_addr;  // weight [K x N]  (transposed)
  uint64_t bias_addr; // bias [N]
  uint64_t c_addr;  // output [M x N]
} kernel_arg_t;

#endif
