#ifndef CNN_SGEMM_COMMON_H
#define CNN_SGEMM_COMMON_H

#include <stdint.h>

//
//  kernel_arg_t
//  This MUST match exactly the memory layout expected inside the
//  Vortex kernel (sgemm kernel using wmma_context).
//
typedef struct {
    // Device pointers
    uint64_t A_addr;   // pointer to matrix A  (M × K)
    uint64_t B_addr;   // pointer to matrix B  (K × N)
    uint64_t C_addr;   // pointer to matrix C  (M × N output)
    uint64_t bias_addr;  // NEW ★ float[M]

    // Matrix dimensions
    uint32_t M;
    uint32_t N;
    uint32_t K;

    uint32_t padM;
    uint32_t padN;
    uint32_t padK;

    // Launch configuration (Vortex supports grid_dim[0:1], block_dim[0])
    uint32_t grid_dim[3];   // ONLY grid_dim[0], grid_dim[1] used
    uint32_t block_dim[3];  // ONLY block_dim[0] used (warp size)
} sgemm_arg_t;


#endif // CNN_SGEMM_COMMON_H
