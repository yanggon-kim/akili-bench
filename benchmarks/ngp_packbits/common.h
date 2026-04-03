#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t num_tasks;       // num_groups (= N/8)
  float    threshold;
  uint64_t density_addr;    // float[num_groups * 8] input
  uint64_t bitfield_addr;   // uint8_t[num_groups] output
} kernel_arg_t;

#endif
