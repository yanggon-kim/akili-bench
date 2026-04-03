#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t num_tasks;   // n_rays
  float    min_near;
  uint64_t rays_o_addr; // float[n_rays*3]
  uint64_t rays_d_addr; // float[n_rays*3]
  uint64_t aabb_addr;   // float[6]
  uint64_t t_mins_addr; // float[n_rays] output
  uint64_t t_maxs_addr; // float[n_rays] output
  uint64_t hits_addr;   // uint32_t[n_rays] output (0 or 1)
} kernel_arg_t;

#endif
