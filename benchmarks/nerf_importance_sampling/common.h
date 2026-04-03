#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t num_tasks;   // n_rays
  uint32_t n_cdf;       // CDF values per ray
  uint32_t n_new;       // new samples to generate per ray
  uint64_t cdfs_addr;   // float[n_rays * n_cdf] input
  uint64_t out_addr;    // float[n_rays * n_new] output positions
} kernel_arg_t;

#endif
