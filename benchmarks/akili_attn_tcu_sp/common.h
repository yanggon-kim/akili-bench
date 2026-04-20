#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

#define KID_QK_SPARSE   0   // sparse TCU Q·Kᵀ  (A = compressed fp16 Q)
#define KID_SOFTMAX     1   // SIMT softmax on fp32 S
#define KID_PV_SPARSE   2   // sparse TCU P·V   (A = compressed fp16 P)
#define KID_PV_DENSE    3   // dense TCU  P·V   (A = full fp16 P, S2 opt-in)

typedef struct {
  uint64_t Q_addr;
  uint64_t K_addr;
  uint64_t V_addr;
  uint64_t P_addr;
  uint64_t S_addr;
  uint64_t O_addr;
  uint64_t meta_Q_addr;
  uint64_t meta_P_addr;
  uint64_t cycles_addr;  // uint32_t[num_blocks_max]
  uint64_t instrs_addr;  // uint32_t[num_blocks_max] per-stage retired instructions
  uint32_t N;
  uint32_t d;
  uint32_t kernel_id;
} kernel_arg_t;

#endif
