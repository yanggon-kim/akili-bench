#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

// Kernel stage IDs — three SIMT kernels dispatched sequentially from host.
#define KID_QK_SIMT   0
#define KID_SOFTMAX   1
#define KID_PV_SIMT   2

typedef struct {
  uint64_t Q_addr;
  uint64_t K_addr;
  uint64_t S_addr;
  uint64_t P_addr;
  uint64_t V_addr;
  uint64_t O_addr;
  uint64_t kernel_cycles;  // written by device main() after spawn
  uint32_t grid_dim[2];
  uint32_t block_dim[2];
  uint32_t N;
  uint32_t d;
  uint32_t kernel_id;
  uint32_t _pad;
} kernel_arg_t;

#endif
