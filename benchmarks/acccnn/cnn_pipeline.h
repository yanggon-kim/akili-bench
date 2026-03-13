//#pragma once
#include <vector>
#include <stdint.h>
#include <string>
#include <vortex.h>
#include "common.h"

//
// High-level CNN forward declarations
//
// This header defines the shared interfaces for the CNN pipeline:
// Conv → Pool → ReLU → Flatten → FC.

// It also defines a small tensor struct and helper APIs.

// Simple tensor representation (CHW or HWC depending on context)
// We store data in a flat vector<float>.
// struct Tensor {
//     std::vector<float> data;
//     int C;   // channels
//     int H;   // height
//     int W;   // width

//     Tensor() : C(0), H(0), W(0) {}
//     Tensor(int C_, int H_, int W_)
//         : C(C_), H(H_), W(W_),
//           data(C_ * H_ * W_, 0.0f) {}
// };

// --------------------------------------------------------------
// Device wrappers
// --------------------------------------------------------------
class VortexDevice {
public:
    vx_device_h dev;
    VortexDevice();
    ~VortexDevice();
    void check(int err, const char* msg);
};

// --------------------------------------------------------------
// Memory helpers
// --------------------------------------------------------------
uint64_t upload_tensor_to_device(VortexDevice& vx,
                                 const Tensor& t,
                                 vx_buffer_h* buf_out);

uint64_t upload_weights_to_device(VortexDevice& vx,
                                  const std::vector<float>& w,
                                  vx_buffer_h* buf_out);

void download_tensor_from_device(VortexDevice& vx,
                                 Tensor& out,
                                 vx_buffer_h buf);

// --------------------------------------------------------------
// CNN ops
// --------------------------------------------------------------

Tensor conv2d_gpu(VortexDevice& vx,
                  const Tensor& input,
                  const std::vector<float>& W,
                  const std::vector<float>& B,
                  int C_out,
                  int K,
                  int padding,
                  int stride,
                  const char* kernel_file);

Tensor maxpool2d_gpu(VortexDevice& vx,
                     const Tensor& input,
                     int pool_size,
                     const char* kernel_file);

Tensor relu_gpu(VortexDevice& vx,
                const Tensor& input,
                const char* kernel_file);

std::vector<float> fc_cpu(const Tensor& input,
                          const std::vector<float>& W,
                          const std::vector<float>& B);

int argmax(const std::vector<float>& v);

// Softmax (CPU)
std::vector<float> softmax(const std::vector<float>& logits);

//
// Utilities to load .bin weights produced by Python
//
std::vector<float> load_bin_vector(const std::string& path);
Tensor load_bin_tensor(const std::string& path, int C, int H, int W);

