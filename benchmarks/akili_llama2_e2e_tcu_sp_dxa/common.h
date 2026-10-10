#pragma once

#include <stdint.h>

#ifndef VX_CFG_NUM_THREADS
#define VX_CFG_NUM_THREADS 4
#endif

// Stage count for the DEEP_BUF N-stage software-pipelined DXA kernel
// (matmul_kernel.cpp). Unused unless DEEP_BUF is defined. Matches
// tests/regression/llama2/common.h's default.
#ifndef PIPE_N
#define PIPE_N 4
#endif

typedef struct {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint64_t A_addr;
    uint64_t B_addr;
    uint64_t C_addr;
    uint64_t meta_sp_addr;
} matmul_kernel_args_t;
