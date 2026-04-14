#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

#define KID_QK_TCU    0   // dense TCU Q·Kᵀ GEMM
#define KID_SOFTMAX   1   // SIMT row-wise softmax on fp32 S
#define KID_PV_TCU    2   // dense TCU P·V GEMM

typedef struct {
  uint64_t Q_addr;
  uint64_t K_addr;
  uint64_t V_addr;
  uint64_t P_addr;
  uint64_t S_addr;
  uint64_t O_addr;
  uint64_t cycles_addr;  // uint32_t[num_blocks_max]
  uint32_t N;
  uint32_t d;
  uint32_t kernel_id;
} kernel_arg_t;

#endif
