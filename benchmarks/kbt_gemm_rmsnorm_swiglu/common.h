#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t grid_dim[3];
  uint32_t block_dim[3];
  uint32_t M;
  uint32_t N;
  uint32_t K;
  float    eps;
  uint64_t x_addr;      // x [M x K]
  uint64_t rms_w_addr;  // RMSNorm weight [K]
  uint64_t wg_addr;     // Wg [K x N]
  uint64_t wu_addr;     // Wu [K x N]
  uint64_t norm_addr;   // temp normalized [M x K]
  uint64_t gate_addr;   // temp gate [M x N]
  uint64_t c_addr;      // output [M x N]
} kernel_arg_t;

#endif
