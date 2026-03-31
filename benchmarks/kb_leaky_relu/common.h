#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t num_rows;
  uint32_t num_cols;
  uint64_t x_addr;  // input
  uint64_t y_addr;  // output
  float negative_slope;
} kernel_arg_t;

#endif
