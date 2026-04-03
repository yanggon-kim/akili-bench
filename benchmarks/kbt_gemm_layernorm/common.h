#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t grid_dim[3];
  uint32_t block_dim[3];
  uint32_t M;
  uint32_t N;
  uint32_t K;
  float    eps;
  uint64_t a_addr;      // A [M x K]
  uint64_t w_addr;      // W [K x N]
  uint64_t bias_addr;   // bias [N]
  uint64_t ln_w_addr;   // layernorm weight [N]
  uint64_t ln_b_addr;   // layernorm bias [N]
  uint64_t c_addr;      // C [M x N] output
} kernel_arg_t;

#endif
