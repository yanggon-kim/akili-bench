#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef TYPE
#define TYPE float   // CNN uses float32
#endif

#include <stdint.h>
#include <vector>

// ------------------------------------------------------------
// Convolution kernel args (must match cnn_conv regression)
// ------------------------------------------------------------
typedef struct {
  // Device pointers
  uint64_t I_addr;   // Input address [C_in][H][W]
  uint64_t W_addr;   // Weight address [C_out][C_in][3][3]
  uint64_t B_addr;   // Bias address [C_out]
  uint64_t O_addr;   // Output address [C_out][H_out][W_out]

  // Convolution parameters
  uint32_t C_in;     // Input channels
  uint32_t C_out;    // Output channels
  uint32_t height;   // Input height
  uint32_t width;    // Input width
  uint32_t padding;  // Padding
  uint32_t stride;   // Stride

  // Derived output dimensions
  uint32_t H_out;
  uint32_t W_out;

  // Execution configuration
  uint32_t grid_dim[3];   // [W_out, H_out, C_out] (z unused)
  uint32_t block_dim[3];  // usually [1,1,1]

  // Flags
  uint32_t use_lmem;      // 1: copy weights into local memory, 0: use device memory
} kernel_arg_t;

// ------------------------------------------------------------
// ReLU kernel args (must match cnn_relu regression)
// In-place: X_addr points to data, total is C*H*W
// ------------------------------------------------------------
typedef struct {
  uint64_t X_addr;        // Activation in/out (in-place)
  uint32_t total;         // Total elements = C×H×W
  uint32_t grid_dim[3];
  uint32_t block_dim[3];
} kernel_arg_relu_t;

// ------------------------------------------------------------
// MaxPool kernel args (must match cnn_pool regression)
// ------------------------------------------------------------
typedef struct {
  uint64_t I_addr;        // input [C][H][W]
  uint64_t O_addr;        // output [C][H/2][W/2]
  uint32_t C;
  uint32_t H;
  uint32_t W;
  uint32_t grid_dim[3];
  uint32_t block_dim[3];
} kernel_arg_pool_t;

// ------------------------------------------------------------
// HOST-SIDE Tensor (CHW layout)
// ------------------------------------------------------------
struct Tensor {
  int C, H, W;
  std::vector<float> data;

  Tensor(int c=0, int h=0, int w=0)
    : C(c), H(h), W(w), data(c*h*w) {}

  inline float& at(int c, int y, int x) {
    return data[c*H*W + y*W + x];
  }
};

#endif // _COMMON_H_
