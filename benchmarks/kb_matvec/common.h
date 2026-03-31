#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t M;       // rows of A and y
  uint32_t K;       // cols of A, length of x
  uint64_t a_addr;  // A [M x K]
  uint64_t x_addr;  // x [K]
  uint64_t y_addr;  // y [M]
} kernel_arg_t;

#endif
