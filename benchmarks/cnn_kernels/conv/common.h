#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef TYPE
#define TYPE float   // can be overridden at compile time
#endif

#include <stdint.h>

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
  uint32_t padding;  // Padding (same on all sides)
  uint32_t stride;   // Stride (currently assumed square)
  
  // Derived output dimensions
  uint32_t H_out;
  uint32_t W_out;

  // Execution configuration
  uint32_t grid_dim[3];   // [W_out, H_out, C_out]
  uint32_t block_dim[3];  // (not used in Vortex, but included for compatibility)

  // Flags
  uint32_t use_lmem;      // 1: copy weights into local memory, 0: use device memory
} kernel_arg_t;

typedef struct {
    uint64_t X_addr;     // Activation in/out (in-place)
    uint32_t total;      // Total elements = C×H×W
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
} kernel_arg_relu_t;

typedef struct {
    uint64_t I_addr;
    uint64_t O_addr;
    uint32_t C;
    uint32_t H;
    uint32_t W;
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
} kernel_arg_pool_t;


#endif // _COMMON_H_
