#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t n_rays;
  uint32_t grid_res;     // H
  uint32_t max_steps;
  float    bound;
  float    dt_min;       // precomputed: 2*sqrt(3)/max_steps * bound
  uint64_t rays_o_addr;  // float [n_rays * 3]
  uint64_t rays_d_addr;  // float [n_rays * 3]
  uint64_t nears_addr;   // float [n_rays]
  uint64_t fars_addr;    // float [n_rays]
  uint64_t grid_addr;    // uint8_t [H*H*H / 8]  packed bitfield
  uint64_t xyzs_addr;    // float [n_rays * max_steps * 3]  output
  uint64_t deltas_addr;  // float [n_rays * max_steps]      output
  uint64_t counts_addr;  // uint32_t [n_rays]                output
} kernel_arg_t;

#endif
