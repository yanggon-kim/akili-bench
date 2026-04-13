#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

// Execution modes
#define MODE_SIMT         0
#define MODE_DENSE_TCU    1
#define MODE_SPARSE_TCU   2

// Kernel IDs (which stage + which mode)
#define KID_QK_SIMT      0
#define KID_SOFTMAX      1
#define KID_PV_SIMT      2
#define KID_QK_TCU       3
#define KID_PV_TCU       4
#define KID_QK_SPARSE    5
#define KID_PV_SPARSE    6

typedef struct {
  // 64-bit fields first (for alignment)
  uint64_t Q_addr;
  uint64_t K_addr;
  uint64_t S_addr;
  uint64_t P_addr;
  uint64_t V_addr;
  uint64_t O_addr;
  uint64_t meta_Q_addr;    // sparse metadata for Q (used by KID_QK_SPARSE)
  uint64_t meta_P_addr;    // sparse metadata for P (used by KID_PV_SPARSE)
  uint64_t kernel_cycles;  // written by kernel after vx_spawn_threads
  // 32-bit fields
  uint32_t grid_dim[2];
  uint32_t block_dim[2];
  uint32_t N;
  uint32_t d;
  uint32_t kernel_id;
  uint32_t _pad;           // keep struct size 8-byte aligned
} kernel_arg_t;

#endif
