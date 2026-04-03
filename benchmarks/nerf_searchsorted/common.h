#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t num_tasks;       // n_rays
  uint32_t n_sorted;        // sorted values per ray
  uint32_t n_queries;       // query values per ray
  uint32_t _pad;            // alignment padding
  uint64_t sorted_addr;     // float[n_rays * n_sorted]
  uint64_t query_addr;      // float[n_rays * n_queries]
  uint64_t ids_left_addr;   // uint32_t[n_rays * n_queries] output
  uint64_t ids_right_addr;  // uint32_t[n_rays * n_queries] output
} kernel_arg_t;

#endif
