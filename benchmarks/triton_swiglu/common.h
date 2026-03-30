#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t num_rows;
  uint32_t num_cols;
  uint64_t a_addr;  // input gate
  uint64_t b_addr;  // input value
  uint64_t c_addr;  // output: silu(a) * b
} kernel_arg_t;

#endif
