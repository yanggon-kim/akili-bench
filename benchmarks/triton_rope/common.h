#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t num_tokens;  // batch_size * seq_len
  uint32_t head_dim;    // dimension per head (must be even)
  uint32_t num_heads;
  uint64_t x_addr;      // input  [num_tokens × num_heads × head_dim]
  uint64_t y_addr;      // output [num_tokens × num_heads × head_dim]
  float theta_base;     // RoPE base frequency (typically 10000.0)
} kernel_arg_t;

#endif
