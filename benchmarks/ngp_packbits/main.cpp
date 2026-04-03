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
uint32_t num_groups = 64;

vx_device_h device = nullptr;
vx_buffer_h density_buffer = nullptr;
vx_buffer_h bitfield_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex NGP Packbits Benchmark." << std::endl;
   std::cout << "Usage: [-k: kernel] [-n num_groups] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:h")) != -1) {
    switch (c) {
    case 'n': num_groups = atoi(optarg); break;
    case 'k': kernel_file = optarg; break;
    case 'h': show_usage(); exit(0); break;
    default: show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(density_buffer);
    vx_mem_free(bitfield_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

// CPU reference: pack bits
static void cpu_packbits(const float* density, float threshold,
                         uint32_t n_groups, uint8_t* bitfield) {
  for (uint32_t g = 0; g < n_groups; ++g) {
    const float* d = density + g * 8;
    uint8_t bits = 0;
    for (uint32_t i = 0; i < 8; ++i) {
      if (d[i] > threshold)
        bits |= (uint8_t)(1 << i);
    }
    bitfield[g] = bits;
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  std::srand(42);

  float threshold = 0.5f;
  uint32_t num_floats = num_groups * 8;
  uint32_t density_size  = num_floats * sizeof(float);
  uint32_t bitfield_size = num_groups * sizeof(uint8_t);

  std::cout << "Packbits: num_groups=" << num_groups
            << " (" << num_floats << " density values)" << std::endl;

  // Generate random density values in [0, 1]
  std::vector<float> h_density(num_floats);
  for (uint32_t i = 0; i < num_floats; ++i)
    h_density[i] = static_cast<float>(rand()) / RAND_MAX;

  RT_CHECK(vx_dev_open(&device));

  kernel_arg.num_tasks = num_groups;
  kernel_arg.threshold = threshold;

  RT_CHECK(vx_mem_alloc(device, density_size, VX_MEM_READ, &density_buffer));
  RT_CHECK(vx_mem_address(density_buffer, &kernel_arg.density_addr));
  RT_CHECK(vx_mem_alloc(device, bitfield_size, VX_MEM_WRITE, &bitfield_buffer));
  RT_CHECK(vx_mem_address(bitfield_buffer, &kernel_arg.bitfield_addr));

  RT_CHECK(vx_copy_to_dev(density_buffer, h_density.data(), 0, density_size));

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // Read back results
  std::vector<uint8_t> h_bitfield(num_groups, 0);
  RT_CHECK(vx_copy_from_dev(h_bitfield.data(), bitfield_buffer, 0, bitfield_size));

  // CPU reference
  std::vector<uint8_t> ref_bitfield(num_groups);
  cpu_packbits(h_density.data(), threshold, num_groups, ref_bitfield.data());

  // Verify byte-by-byte
  int errors = 0;
  for (uint32_t i = 0; i < num_groups; ++i) {
    if (h_bitfield[i] != ref_bitfield[i]) {
      if (errors < 10)
        printf("*** error: group[%d] expected=0x%02x, actual=0x%02x\n",
               i, ref_bitfield[i], h_bitfield[i]);
      ++errors;
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
