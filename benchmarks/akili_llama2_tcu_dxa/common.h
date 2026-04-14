#pragma once

#include <stdint.h>

#ifndef NUM_THREADS
#define NUM_THREADS 4
#endif

typedef struct {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint64_t A_addr;
    uint64_t B_addr;
    uint64_t C_addr;
} matmul_kernel_args_t;
