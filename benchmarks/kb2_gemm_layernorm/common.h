#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t M;       // batch (rows)
  uint32_t N;       // out_features (hidden dim / cols of output)
  uint32_t K;       // in_features
  float eps;
  uint64_t a_addr;    // input  [M x K]
  uint64_t w_addr;    // weight [K x N]  (matmul weight)
  uint64_t bias_addr; // bias   [N]      (matmul bias)
  uint64_t ln_w_addr; // layernorm weight [N]
  uint64_t ln_b_addr; // layernorm bias   [N]
  uint64_t c_addr;    // output [M x N]
} kernel_arg_t;

#endif
