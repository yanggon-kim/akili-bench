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

// Kernel IDs (match attention's layout + a flash-specific fused entry).
#define KID_QK_SIMT       0
#define KID_SOFTMAX       1
#define KID_PV_SIMT       2
#define KID_QK_TCU        3
#define KID_PV_TCU        4
#define KID_QK_SPARSE     5
#define KID_PV_SPARSE     6
#define KID_FLASH_FUSED   7   // single-launch fused flashattention SIMT

typedef struct {
  // 64-bit fields first
  uint64_t Q_addr;
  uint64_t K_addr;
  uint64_t V_addr;
  uint64_t O_addr;
  uint64_t S_addr;       // intermediate S = Q@K^T (TCU/sparse modes)
  uint64_t P_addr;       // intermediate P = softmax(S) (TCU/sparse modes)
  uint64_t meta_Q_addr;  // sparse metadata for Q (sparse QK)
  uint64_t meta_P_addr;  // sparse metadata for P (sparse PV)
  uint64_t kernel_cycles;
  // 32-bit fields
  uint32_t grid_dim[2];
  uint32_t block_dim[2];
  uint32_t kernel_type;  // 0=SIMT, 1=dense TCU, 2=sparse TCU (for main.cpp)
  uint32_t kernel_id;    // KID_* above — which kernel body to run this launch
  uint32_t seq_len;
  uint32_t head_dim;
  uint32_t block_size_r;
  uint32_t block_size_c;
  uint32_t N;            // padded N (for TCU tile alignment)
  uint32_t d;            // padded d
} kernel_arg_t;

#endif
