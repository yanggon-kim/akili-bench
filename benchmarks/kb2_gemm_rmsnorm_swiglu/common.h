#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t M;       // batch (rows)
  uint32_t N;       // hidden_dim (output features of gate/up)
  uint32_t K;       // input_dim (features of x, cols of x)
  float eps;
  uint64_t x_addr;     // input   [M x K]
  uint64_t rms_w_addr; // RMSNorm weight [K]
  uint64_t wg_addr;    // gate weight [K x N]
  uint64_t wu_addr;    // up weight   [K x N]
  uint64_t c_addr;     // output  [M x N]
} kernel_arg_t;

#endif
