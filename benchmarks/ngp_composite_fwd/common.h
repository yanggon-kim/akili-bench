#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t n_rays;
  uint32_t samples_per_ray;
  float    T_thresh;
  uint64_t sigmas_addr;       // float [n_rays * samples_per_ray]
  uint64_t rgbs_addr;         // float [n_rays * samples_per_ray * 3]
  uint64_t deltas_addr;       // float [n_rays * samples_per_ray]
  uint64_t weights_sum_addr;  // float [n_rays]               output
  uint64_t depth_addr;        // float [n_rays]               output
  uint64_t image_addr;        // float [n_rays * 3]           output
} kernel_arg_t;

#endif
