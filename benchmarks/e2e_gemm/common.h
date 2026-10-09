#pragma once
#include <stdint.h>

#define GEMM_BM 128
#define GEMM_BN 64

typedef struct {
  uint64_t a, a_sc, b, b_sc, c;
  uint32_t M, N, K, num_kt;
} gemm_args_t;
