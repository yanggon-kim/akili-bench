#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t M;       // batch (rows)
  uint32_t N;       // hidden_size (output cols)
  uint32_t K;       // input_size
  float scale_factor;
  float clamp_min;
  float clamp_max;
  uint64_t a_addr;    // input  [M x K]
  uint64_t w_addr;    // weight [K x N]
  uint64_t bias_addr; // bias   [N]
  uint64_t c_addr;    // output [M x 1] (logsumexp reduces cols)
} kernel_arg_t;

#endif
