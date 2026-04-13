#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

#define KID_QK_SIMT      0
#define KID_SOFTMAX      1
#define KID_PV_SIMT      2
#define KID_FLASH_FUSED  7   // single-launch fused flashattention SIMT (d ≤ 16)

typedef struct {
  uint64_t Q_addr;
  uint64_t K_addr;
  uint64_t V_addr;
  uint64_t O_addr;
  uint64_t S_addr;
  uint64_t P_addr;
  uint64_t kernel_cycles;
  uint32_t grid_dim[2];
  uint32_t block_dim[2];
  uint32_t kernel_id;
  uint32_t seq_len;
  uint32_t head_dim;
  uint32_t block_size_r;
  uint32_t block_size_c;
  uint32_t N;            // padded N
  uint32_t d;            // padded d
  uint32_t _pad;
} kernel_arg_t;

#endif
