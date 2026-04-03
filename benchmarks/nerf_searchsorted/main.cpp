#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <cmath>
#include <vortex.h>
#include "common.h"

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

const char* kernel_file = "kernel.vxbin";
uint32_t n_rays = 32;
uint32_t n_sorted = 16;
uint32_t n_queries = 8;

vx_device_h device = nullptr;
vx_buffer_h sorted_buffer = nullptr;
vx_buffer_h query_buffer = nullptr;
vx_buffer_h ids_left_buffer = nullptr;
vx_buffer_h ids_right_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex NeRF Searchsorted Benchmark." << std::endl;
   std::cout << "Usage: [-k: kernel] [-r rays] [-s sorted] [-q queries] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "r:s:q:k:h")) != -1) {
    switch (c) {
    case 'r': n_rays = atoi(optarg); break;
    case 's': n_sorted = atoi(optarg); break;
    case 'q': n_queries = atoi(optarg); break;
    case 'k': kernel_file = optarg; break;
    case 'h': show_usage(); exit(0); break;
    default: show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(sorted_buffer);
    vx_mem_free(query_buffer);
    vx_mem_free(ids_left_buffer);
    vx_mem_free(ids_right_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  std::srand(42);

  uint32_t sorted_total  = n_rays * n_sorted;
  uint32_t query_total   = n_rays * n_queries;
  uint32_t sorted_size   = sorted_total * sizeof(float);
  uint32_t query_size    = query_total * sizeof(float);
  uint32_t ids_size      = query_total * sizeof(uint32_t);

  std::cout << "Searchsorted: n_rays=" << n_rays
            << " n_sorted=" << n_sorted
            << " n_queries=" << n_queries << std::endl;

  // Generate monotonically increasing sorted values per ray
  std::vector<float> h_sorted(sorted_total);
  for (uint32_t r = 0; r < n_rays; ++r) {
    float val = 0.0f;
    for (uint32_t s = 0; s < n_sorted; ++s) {
      val += 0.1f + 0.9f * static_cast<float>(rand()) / RAND_MAX;
      h_sorted[r * n_sorted + s] = val;
    }
  }

  // Generate query values in valid range per ray
  std::vector<float> h_query(query_total);
  for (uint32_t r = 0; r < n_rays; ++r) {
    float lo = h_sorted[r * n_sorted];
    float hi = h_sorted[r * n_sorted + n_sorted - 1];
    for (uint32_t q = 0; q < n_queries; ++q) {
      h_query[r * n_queries + q] = lo + (hi - lo) * static_cast<float>(rand()) / RAND_MAX;
    }
  }

  RT_CHECK(vx_dev_open(&device));

  kernel_arg.num_tasks  = n_rays;
  kernel_arg.n_sorted   = n_sorted;
  kernel_arg.n_queries  = n_queries;

  RT_CHECK(vx_mem_alloc(device, sorted_size, VX_MEM_READ, &sorted_buffer));
  RT_CHECK(vx_mem_address(sorted_buffer, &kernel_arg.sorted_addr));
  RT_CHECK(vx_mem_alloc(device, query_size, VX_MEM_READ, &query_buffer));
  RT_CHECK(vx_mem_address(query_buffer, &kernel_arg.query_addr));
  RT_CHECK(vx_mem_alloc(device, ids_size, VX_MEM_WRITE, &ids_left_buffer));
  RT_CHECK(vx_mem_address(ids_left_buffer, &kernel_arg.ids_left_addr));
  RT_CHECK(vx_mem_alloc(device, ids_size, VX_MEM_WRITE, &ids_right_buffer));
  RT_CHECK(vx_mem_address(ids_right_buffer, &kernel_arg.ids_right_addr));

  RT_CHECK(vx_copy_to_dev(sorted_buffer, h_sorted.data(), 0, sorted_size));
  RT_CHECK(vx_copy_to_dev(query_buffer, h_query.data(), 0, query_size));

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  std::vector<uint32_t> h_ids_left(query_total, 0), h_ids_right(query_total, 0);
  RT_CHECK(vx_copy_from_dev(h_ids_left.data(), ids_left_buffer, 0, ids_size));
  RT_CHECK(vx_copy_from_dev(h_ids_right.data(), ids_right_buffer, 0, ids_size));

  // CPU reference: binary search per ray per query
  int errors = 0;
  for (uint32_t r = 0; r < n_rays; ++r) {
    uint32_t s_off = r * n_sorted;
    uint32_t q_off = r * n_queries;
    for (uint32_t qi = 0; qi < n_queries; ++qi) {
      float val = h_query[q_off + qi];
      // lower_bound binary search
      uint32_t lo = 0, hi = n_sorted;
      while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (h_sorted[s_off + mid] < val) lo = mid + 1;
        else hi = mid;
      }
      uint32_t exp_left  = (lo > 0) ? (lo - 1) : 0;
      uint32_t exp_right = (lo < n_sorted) ? lo : (n_sorted - 1);

      if (h_ids_left[q_off + qi] != exp_left) {
        if (errors < 10)
          printf("*** error: ray[%u] query[%u] left expected=%u, actual=%u\n",
                 r, qi, exp_left, h_ids_left[q_off + qi]);
        ++errors;
      }
      if (h_ids_right[q_off + qi] != exp_right) {
        if (errors < 10)
          printf("*** error: ray[%u] query[%u] right expected=%u, actual=%u\n",
                 r, qi, exp_right, h_ids_right[q_off + qi]);
        ++errors;
      }
    }
  }

  cleanup();
  if (errors) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
