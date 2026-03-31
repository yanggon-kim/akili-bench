#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t num_rows;
  uint32_t num_cols;
  uint64_t x_addr;  // input  [num_rows x num_cols]
  uint64_t w_addr;  // weight [num_cols]
  uint64_t b_addr;  // bias   [num_cols]
  uint64_t y_addr;  // output [num_rows x num_cols]
  float eps;
} kernel_arg_t;

#endif
