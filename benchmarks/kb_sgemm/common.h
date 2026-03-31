#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t M;       // rows of A and C
  uint32_t N;       // cols of B and C
  uint32_t K;       // cols of A, rows of B
  uint64_t a_addr;  // A [M x K]
  uint64_t b_addr;  // B [K x N]
  uint64_t c_addr;  // C [M x N]
} kernel_arg_t;

#endif
