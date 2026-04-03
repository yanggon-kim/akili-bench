#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t num_tasks;     // n_rays
  uint32_t grid_res;      // grid resolution (grid is grid_res^3)
  uint32_t max_samples;   // max output samples per ray
  float    step_size;
  float    aabb0;         // aabb min x
  float    aabb1;         // aabb min y
  float    aabb2;         // aabb min z
  float    aabb3;         // aabb max x
  float    aabb4;         // aabb max y
  float    aabb5;         // aabb max z
  uint64_t rays_o_addr;   // float[n_rays*3]
  uint64_t rays_d_addr;   // float[n_rays*3]
  uint64_t nears_addr;    // float[n_rays]
  uint64_t fars_addr;     // float[n_rays]
  uint64_t grid_addr;     // uint8_t[grid_res^3]
  uint64_t t_starts_addr; // float[n_rays*max_samples] output
  uint64_t t_ends_addr;   // float[n_rays*max_samples] output
  uint64_t counts_addr;   // uint32_t[n_rays] output
} kernel_arg_t;

#endif
