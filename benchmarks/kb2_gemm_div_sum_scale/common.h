#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t M;       // batch (rows)
  uint32_t N;       // hidden_size (cols of W^T)
  uint32_t K;       // input_size (cols of x)
  float scaling_factor;
  uint64_t a_addr;  // input  [M x K]
  uint64_t w_addr;  // weight [K x N]  (already transposed)
  uint64_t c_addr;  // output [M x 1]  (scalar per row)
} kernel_arg_t;

#endif
