#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>

// ------------------------------------------------------------
// Global datatype for CNN computation
// ------------------------------------------------------------
typedef float TYPE;   // Fashion-MNIST uses float32 weights and inputs

// ------------------------------------------------------------
// Convolution Kernel Argument Structure
//
// Matches your conv kernel:
//   - C_in = 1 for grayscale Fashion-MNIST
//   - C_out = 8 for your trained model
//   - padding = 0 (Keras "valid")
//   - stride = 1
//   - grid_dim = {W_out, H_out, 1}
// ------------------------------------------------------------
typedef struct {
    uint64_t I_addr;
    uint64_t W_addr;
    uint64_t O_addr;
    uint64_t B_addr;

    int C_in;
    int C_out;
    int height;
    int width;
    int padding;
    int stride;
    int H_out;
    int W_out;

    int use_lmem;      // already existed
    int use_tcu;       // NEW — 0 = CPU FMA, 1 = Tensor Core MMA

    uint32_t grid_dim[3];
    uint32_t block_dim[3];
} kernel_arg_t;

static inline uint16_t float_to_fp16(float x) {
    uint32_t bits = *reinterpret_cast<uint32_t*>(&x);
    uint16_t sign = (bits >> 16) & 0x8000;
    uint16_t mant = (bits >> 13) & 0x03FF;
    uint16_t exp  = (bits >> 23) & 0xFF;

    if (exp == 0)   return sign;              // zero
    if (exp == 255) return sign | 0x7C00;     // inf/NaN

    int new_exp = exp - 127 + 15;

    if (new_exp <= 0) return sign;            // underflow → zero
    if (new_exp >= 31) return sign | 0x7C00;  // overflow → inf

    return sign | (new_exp << 10) | mant;
}


// ------------------------------------------------------------
// MaxPool Kernel Argument Structure
//
// pool_size = 2
// output dims = (C, H/2, W/2)
// ------------------------------------------------------------
typedef struct {
    uint64_t I_addr;      // input feature map
    uint64_t O_addr;      // output feature map

    int32_t C;
    int32_t H;
    int32_t W;

    int32_t grid_dim[3];
    int32_t block_dim[3];
} kernel_arg_pool_t;

// ------------------------------------------------------------
// ReLU Kernel Argument Structure
//
// Applies ReLU to entire tensor elementwise
// ------------------------------------------------------------
typedef struct {
    uint64_t I_addr;  
    uint64_t O_addr;  

    int32_t C;
    int32_t H;
    int32_t W;

    int32_t grid_dim[3]; 
    int32_t block_dim[3];
} kernel_arg_relu_t;

// ------------------------------------------------------------
// HOST-SIDE Tensor struct (not used on device)
// ------------------------------------------------------------
struct Tensor {
    int C, H, W;
    std::vector<float> data;

    Tensor(int c=0, int h=0, int w=0)
        : C(c), H(h), W(w), data(c*h*w) {}

    inline float& at(int c, int y, int x) {
        return data[c*H*W + y*W + x];
    }
};

#endif // COMMON_H
