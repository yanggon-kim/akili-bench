#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t num_tasks;   // n_rays
  float    min_near;
  float    aabb0;       // aabb min x
  float    aabb1;       // aabb min y
  float    aabb2;       // aabb min z
  float    aabb3;       // aabb max x
  float    aabb4;       // aabb max y
  float    aabb5;       // aabb max z
  uint64_t rays_o_addr; // float[n_rays*3]
  uint64_t rays_d_addr; // float[n_rays*3]
  uint64_t nears_addr;  // float[n_rays] output
  uint64_t fars_addr;   // float[n_rays] output
} kernel_arg_t;

#endif
