#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <cmath>
#include <vortex.h>
#include "common.h"

#define FLOAT_ULP 20  // cross-entropy has more FP error due to exp/log chain

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
uint32_t num_rows = 64;
uint32_t num_classes = 32;

vx_device_h device = nullptr;
vx_buffer_h logits_buffer = nullptr;
vx_buffer_h labels_buffer = nullptr;
vx_buffer_h loss_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex CrossEntropy Benchmark (Triton-style)." << std::endl;
   std::cout << "Usage: [-k: kernel] [-r rows] [-c classes] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "r:c:k:h")) != -1) {
    switch (c) {
    case 'r': num_rows = atoi(optarg); break;
    case 'c': num_classes = atoi(optarg); break;
    case 'k': kernel_file = optarg; break;
    case 'h': show_usage(); exit(0); break;
    default: show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(logits_buffer);
    vx_mem_free(labels_buffer);
    vx_mem_free(loss_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  std::srand(42);

  uint32_t logits_size = num_rows * num_classes * sizeof(float);
  uint32_t labels_size = num_rows * sizeof(int32_t);
  uint32_t loss_size = num_rows * sizeof(float);

  std::cout << "CrossEntropy: rows=" << num_rows << " classes=" << num_classes << std::endl;

  RT_CHECK(vx_dev_open(&device));

  kernel_arg.num_rows = num_rows;
  kernel_arg.num_classes = num_classes;

  RT_CHECK(vx_mem_alloc(device, logits_size, VX_MEM_READ, &logits_buffer));
  RT_CHECK(vx_mem_address(logits_buffer, &kernel_arg.logits_addr));
  RT_CHECK(vx_mem_alloc(device, labels_size, VX_MEM_READ, &labels_buffer));
  RT_CHECK(vx_mem_address(labels_buffer, &kernel_arg.labels_addr));
  RT_CHECK(vx_mem_alloc(device, loss_size, VX_MEM_WRITE, &loss_buffer));
  RT_CHECK(vx_mem_address(loss_buffer, &kernel_arg.loss_addr));

  std::vector<float> h_logits(num_rows * num_classes);
  std::vector<int32_t> h_labels(num_rows);
  std::vector<float> h_loss(num_rows, 0);

  for (uint32_t i = 0; i < num_rows * num_classes; ++i)
    h_logits[i] = 4.0f * static_cast<float>(rand()) / RAND_MAX - 2.0f;
  for (uint32_t i = 0; i < num_rows; ++i)
    h_labels[i] = rand() % num_classes;

  RT_CHECK(vx_copy_to_dev(logits_buffer, h_logits.data(), 0, logits_size));
  RT_CHECK(vx_copy_to_dev(labels_buffer, h_labels.data(), 0, labels_size));
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  RT_CHECK(vx_copy_from_dev(h_loss.data(), loss_buffer, 0, loss_size));

  // Verify
  int errors = 0;
  for (uint32_t r = 0; r < num_rows; ++r) {
    float* row = &h_logits[r * num_classes];
    float max_val = row[0];
    for (uint32_t c = 1; c < num_classes; ++c)
      if (row[c] > max_val) max_val = row[c];
    float sum_exp = 0;
    for (uint32_t c = 0; c < num_classes; ++c)
      sum_exp += expf(row[c] - max_val);
    float expected = -row[h_labels[r]] + logf(sum_exp) + max_val;
    union { float f; int32_t i; } fa, fb;
    fa.f = h_loss[r]; fb.f = expected;
    if (std::abs(fa.i - fb.i) > FLOAT_ULP) {
      if (errors < 10)
        printf("*** error: [%d] expected=%f, actual=%f\n", r, expected, h_loss[r]);
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
