#pragma once
#include <stdint.h>

typedef struct {
    // pointers
    uint64_t I_addr;   // input image
    uint64_t O_addr;   // im2col output

    // dimensions
    int C_in;
    int H;
    int W;
    int K;
    int stride;
    int padding;

    int H_out;
    int W_out;

    // grid config
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
} im2col_arg_t;
