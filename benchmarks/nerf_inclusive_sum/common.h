#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t num_tasks;       // n_rays
  uint32_t samples_per_ray;
  uint64_t input_addr;      // float[n_rays * samples_per_ray]
  uint64_t output_addr;     // float[n_rays * samples_per_ray]
} kernel_arg_t;

#endif
