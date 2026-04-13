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
  uint64_t Q_addr;   // fp16 row-major [N × d]
  uint64_t K_addr;   // fp16 col-major [N × d]  (K_cm[n*d+k] = K_host[k*N+n])
  uint64_t V_addr;   // fp16 col-major [d × N]  (V_cm[d_idx*N+n] = V_host[n*d+d_idx])
  uint64_t P_addr;   // fp32 (softmax stage) or fp16 (PV TCU stage) — host swaps
  uint64_t S_addr;   // fp32 intermediate
  uint64_t O_addr;   // fp32 output [N × d]
  uint64_t kernel_cycles;
  uint32_t grid_dim[2];
  uint32_t block_dim[2];
  uint32_t N;
  uint32_t d;
  uint32_t kernel_id;
  uint32_t _pad;
} kernel_arg_t;

#endif
