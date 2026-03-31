#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t M;       // seq_len (rows of Q, rows of output)
  uint32_t N;       // seq_len (cols of K^T, cols of output)
  uint32_t D;       // head_dim (cols of Q, cols of K)
  uint64_t q_addr;  // Q [M x D]
  uint64_t k_addr;  // K [N x D]  (transposed during dot product)
  uint64_t c_addr;  // output [M x N]
} kernel_arg_t;

#endif
