#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t grid_dim[3];
  uint32_t block_dim[3];
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint64_t a_addr;     // x [M x K]
  uint64_t wa_addr;    // Wa [K x N]
  uint64_t ba_addr;    // ba [N]
  uint64_t wb_addr;    // Wb [K x N]
  uint64_t bb_addr;    // bb [N]
  uint64_t gate_addr;  // temp gate [M x N]
  uint64_t c_addr;     // output [M x N]
} kernel_arg_t;

#endif
