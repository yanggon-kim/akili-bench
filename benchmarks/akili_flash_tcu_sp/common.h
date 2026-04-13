#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

#define KID_QK_SPARSE   0   // sparse TCU Q·Kᵀ  (A = compressed fp16 Q)
#define KID_SOFTMAX     1   // SIMT softmax on fp32 S
#define KID_PV_SPARSE   2   // sparse TCU P·V   (A = compressed fp16 P)

typedef struct {
  uint64_t Q_addr;         // compressed fp16 Q      [N × d/2]
  uint64_t K_addr;         // fp16 K col-major       [N × d]
  uint64_t V_addr;         // fp16 V col-major       [d × N]
  uint64_t P_addr;         // fp32 (softmax) / compressed fp16 (PV sparse)
  uint64_t S_addr;         // fp32 intermediate
  uint64_t O_addr;         // fp32 output
  uint64_t meta_Q_addr;    // packed 2:4 metadata for Q
  uint64_t meta_P_addr;    // packed 2:4 metadata for P
  uint64_t kernel_cycles;
  uint32_t grid_dim[2];
  uint32_t block_dim[2];
  uint32_t N;
  uint32_t d;
  uint32_t kernel_id;
  uint32_t _pad;
} kernel_arg_t;

#endif
