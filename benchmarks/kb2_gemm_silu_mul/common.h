#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t M;       // batch (rows)
  uint32_t N;       // out_features
  uint32_t K;       // in_features
  uint64_t a_addr;     // input   [M x K]
  uint64_t wa_addr;    // weight_a (gate) [K x N]
  uint64_t ba_addr;    // bias_a   [N]
  uint64_t wb_addr;    // weight_b (up)   [K x N]
  uint64_t bb_addr;    // bias_b   [N]
  uint64_t c_addr;     // output  [M x N]
} kernel_arg_t;

#endif
