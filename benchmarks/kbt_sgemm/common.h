#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t grid_dim[3];
  uint32_t block_dim[3];
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint64_t a_addr;  // A [M x K]
  uint64_t b_addr;  // B [K x N]
  uint64_t c_addr;  // C [M x N]
} kernel_arg_t;

#endif
