#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t grid_dim[3];
  uint32_t block_dim[3];
  uint32_t M;   // seq_len query
  uint32_t N;   // seq_len key
  uint32_t D;   // head_dim
  uint64_t q_addr;  // Q [M x D]
  uint64_t k_addr;  // K [N x D]
  uint64_t c_addr;  // C [M x N]
} kernel_arg_t;

#endif
